import argparse
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from nvae import NVAE
from thirdparty.adamax import Adamax
from util import utils, datasets
from util.sr_utils import SpectralNormCalculator
from tqdm import tqdm

def main(args):
    # common initialization
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # Get data loaders.
    train_queue, valid_queue, test_queue, train_queue_no_shuffle = datasets.get_loaders(args)

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)
    vae = NVAE(args, arch_instance_nvae)
    vae = vae.cuda()

    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(vae.parameters(), args.distributed)

    vae_optimizer = Adamax(vae.parameters(), args.learning_rate_vae,
                           weight_decay=args.weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)

    # create SN calculator
    sn_calculator = SpectralNormCalculator()
    sn_calculator.add_conv_layers(vae)
    sn_calculator.add_bn_layers(vae)

    grad_scalar = GradScaler(2**10)
    bpd_coeff = utils.get_bpd_coeff(args.dataset)

    # if continue training from a checkpoint
    # useful when training is interrupted.
    checkpoint_file = os.path.join(args.save, 'checkpoint_best.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        vae.load_state_dict(checkpoint['vae_state_dict'])
        vae = vae.cuda()
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        if 'sn_calculator' in checkpoint:   # for backward compatibility
            sn_calculator.load_state_dict(checkpoint['sn_calculator'], torch.device("cuda"))
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        best_score = checkpoint['best_score']
        logging.info('loaded the model at epoch %d.', init_epoch)
    else:
        global_step, epoch, init_epoch, best_score = 0, 0, 0, 1e10


    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            vae_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)
        train_obj, global_step, loss_recon = train_vae(args, train_queue, vae, vae_optimizer, grad_scalar, global_step, warmup_iters,
                                           writer, logging, sn_calculator)

        logging.info('train_loss %f', train_obj)

        logging.info('train_reconst_loss %f', loss_recon)
        writer.add_scalar('train/loss_epoch', train_obj, global_step)

        # TODO: define a save model frequency different than evaluation frequency.
        # generate samples less frequently
        if args.dataset == 'Synthetic':
            num_evaluations = 80
        elif args.dataset == 'MIMIC':
            num_evaluations = 40
        else:
            num_evaluations = 10
        eval_freq = max(args.epochs // num_evaluations, 1)
        if ((epoch + 1) % eval_freq == 0 or epoch == (args.epochs - 1)):
            vae.eval()
            print('#################')
            print('begin validation!')
            valid_neg_log_p, valid_nelbo, imputation = test_vae(valid_queue, vae, num_samples=100, args=args, logging=logging)
            print('#################')

            current_score = valid_nelbo
            logging.info('valid bpd nelbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)
            writer.add_scalar('val/nat_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nat_elbo', valid_nelbo, epoch)

            if args.global_rank == 0 and current_score < best_score:
                best_score = current_score
                logging.info('saving the model.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'grad_scalar': grad_scalar.state_dict(), 'best_score': best_score,
                           'vae_state_dict': vae.state_dict(), 'vae_optimizer': vae_optimizer.state_dict(),
                           'vae_scheduler': vae_scheduler.state_dict(), 'sn_calculator': sn_calculator.state_dict()}
                torch.save(content, checkpoint_file)

    # loading the model at the best score
    # make all nodes wait for rank 0 to finish saving the files
    if args.distributed:
        dist.barrier()

    print('input test dataset for evaluation!!!')
    # Final validation
    print('loading the best model.')
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    init_epoch = checkpoint['epoch']
    epoch = init_epoch
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
    vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
    if 'sn_calculator' in checkpoint:  # for backward compatibility
        sn_calculator.load_state_dict(checkpoint['sn_calculator'], torch.device("cuda"))
    grad_scalar.load_state_dict(checkpoint['grad_scalar'])

    print('loading the best model.')
    print('#################')
    print('begin test dataset testing!')
    valid_neg_log_p, valid_nelbo, imputation = test_vae(test_queue, vae, num_samples=100, args=args, logging=logging)
    print('#################')

    with open("VAE_mae_" + args.dataset + "_imputation.txt", "w") as f:
        f.write(str(imputation))
        f.write('\n')


    logging.info('final valid bpd nelbo %f', valid_nelbo * bpd_coeff)
    logging.info('final valid bpd neg log p %f', valid_neg_log_p * bpd_coeff)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.add_scalar('val/nat_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nat_elbo', valid_nelbo, epoch + 1)
    writer.close()

def test_vae_reconstruction(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    reconst_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()

    total_mae = 0
    total_evaluate_point = 0
    total_rmse = 0

    all_generated_samples = []
    all_target = []
    all_evalpoint = []

    for step, (x, x_gt, mask, ob_mask) in enumerate(
            tqdm(valid_queue, total=len(valid_queue), desc="Validating")
    ):
        x = utils.common_x_operations(x, args.num_x_bits)
        x_gt = utils.common_x_operations(x_gt, args.num_x_bits)
        mask = utils.common_x_operations(mask, args.num_x_bits)
        if mask.shape[2] == 24:
            mask[:, :, :, 7] = 0

        with torch.no_grad():
            nelbo, log_iw, reconst = [], [], []

            B,_, K, L = x.shape
            imputed_samples = torch.zeros(B, num_samples, K, L).to(x.device)
            for k in range(num_samples):
                sigma_mask = torch.tensor(False, dtype=torch.bool)

                logits, all_log_q, all_eps, all_dist = model(x,sigma_mask)
                log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                output = model.decoder_output(logits, sigma_mask)
                recon_loss, mae = utils.reconstruction_loss(output, x_gt, (mask), crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss
                nelbo.append(nelbo_batch)
                reconst.append(mae)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

                ######################
                aa = output.mean()
                aa = aa.median(dim=1).values
                imputed_samples[:, k, :, :] = aa
                #########################

                mae = mae.mean(dim=1)
                mse = mae**2
                total_mae += torch.sum(mae)/100
                total_evaluate_point += torch.sum(mask)/100
                total_rmse += torch.sum(mse)/100

            all_target.append(x_gt.squeeze(1))
            all_evalpoint.append((mask).squeeze(1))
            all_generated_samples.append(imputed_samples)

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            reconst = torch.mean(torch.stack(reconst, dim=1))

            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))

        reconst_avg.update(reconst.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

################################################
    imputation = total_mae / total_evaluate_point
    print(' reconstruction imputation mae is ' + str(imputation))

    imputation_rmse = (total_rmse / total_evaluate_point)**0.5
    print(' reconstruction imputation rmse is ' + str(imputation_rmse))

    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)
    CRPS = calc_quantile_CRPS(
        all_target, all_generated_samples, all_evalpoint, 0, 1.0
    )
    print(' CRPS is ' + str(CRPS))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    utils.average_tensor(reconst_avg.avg, args.distributed)

    logging.info(' test reconstruction imputation mae is : %f, rmse: %f,CRPS:  %f', imputation, imputation_rmse, CRPS)

    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)

    logging.info('val, step: %d, mae: %f', step, reconst_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg, imputation


def train_vae(args, train_queue, model, optimizer, grad_scalar, global_step, warmup_iters, writer, logging, sn_calculator):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    loss_recon = utils.AvgrageMeter()

    model.train()
    total_mae = 0
    total_evaluate_point =0

    for step, (x, x_gt, mask, ob_mask) in enumerate(
            tqdm(train_queue, total=len(train_queue), desc="Training")
    ):
        x = utils.common_x_operations(x, args.num_x_bits)
        x_gt = utils.common_x_operations(x_gt, args.num_x_bits)
        mask = utils.common_x_operations(mask, args.num_x_bits)
        ob_mask = utils.common_x_operations(ob_mask, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            sigma_mask = torch.tensor(False, dtype=torch.bool)
            logits, all_log_q, all_eps, all_dist = model(x, sigma_mask)
            log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
            output = model.decoder_output(logits, sigma_mask)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,

                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff,
                                      args.kl_max_coeff)

            recon_loss, mae = utils.reconstruction_loss(output, x, ob_mask, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            mae = mae.mean(dim=1)

            loss_recon_loss = torch.mean(mae)
            total_mae += torch.sum(mae)/100
            total_evaluate_point += torch.sum(mask)/100


            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = sn_calculator.spectral_norm_parallel()
            bn_loss = sn_calculator.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        loss_recon.update(loss_recon_loss.data, 1)


        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        global_step += 1

    imputation = total_mae / total_evaluate_point
    print('train imputation is ' + str(imputation))

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step, loss_recon.avg

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target
    forecast = forecast

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def test_vae(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    reconst_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()

    total_mae = 0
    total_evaluate_point = 0
    total_rmse = 0

    all_generated_samples = []
    all_target = []
    all_evalpoint = []

    for step, (x, x_gt, mask, ob_mask) in enumerate(
            tqdm(valid_queue, total=len(valid_queue), desc="Validating")
    ):
        x = utils.common_x_operations(x, args.num_x_bits)
        x_gt = utils.common_x_operations(x_gt, args.num_x_bits)
        mask = utils.common_x_operations(mask, args.num_x_bits)
        with torch.no_grad():
            nelbo, log_iw, reconst = [], [], []

            B,_, K, L = x.shape
            imputed_samples = torch.zeros(B, num_samples, K, L).to(x.device)
            for k in range(num_samples):
                sigma_mask = torch.tensor(False, dtype=torch.bool)

                logits, all_log_q, all_eps, all_dist = model(x,sigma_mask)
                log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                output = model.decoder_output(logits, sigma_mask)
                recon_loss, mae = utils.reconstruction_loss(output, x_gt, (1-mask), crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss
                nelbo.append(nelbo_batch)
                reconst.append(mae)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

                ######################
                aa = output.mean()
                aa = aa.median(dim=1).values

                imputed_samples[:, k, :, :] = aa
                #########################

                mae = mae.mean(dim=1)
                mse = mae**2
                total_mae += torch.sum(mae)/100
                total_evaluate_point += torch.sum(1-mask)/100

                total_rmse += torch.sum(mse)/100


            all_target.append(x_gt.squeeze(1))
            all_evalpoint.append((1-mask).squeeze(1))
            all_generated_samples.append(imputed_samples)


            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            reconst = torch.mean(torch.stack(reconst, dim=1))

            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))

        reconst_avg.update(reconst.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

################################################
    imputation = total_mae / total_evaluate_point
    print(' imputation mae is ' + str(imputation))

    imputation_rmse = (total_rmse / total_evaluate_point)**0.5
    print(' imputation rmse is ' + str(imputation_rmse))



    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)
    CRPS = calc_quantile_CRPS(
        all_target, all_generated_samples, all_evalpoint, 0, 1.0
    )
    print(' CRPS is ' + str(CRPS))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    utils.average_tensor(reconst_avg.avg, args.distributed)

    logging.info(' imputation mae is : %f, rmse: %f,CRPS:  %f', imputation, imputation_rmse, CRPS)

    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)

    logging.info('val, step: %d, mae: %f', step, reconst_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg, imputation



def infer_active_variables(train_queue, vae, args, max_iter=None):
    kl_meter = utils.AvgrageMeter()
    vae.eval()
    for step, x in enumerate(train_queue):
        if max_iter is not None and step > max_iter:
            break

        x = utils.common_x_operations(x, args.num_x_bits)
        with autocast(enabled=args.autocast_train):
            # apply vae:
            with torch.set_grad_enabled(False):
                _, all_log_q, all_eps = vae(x)
                all_eps = vae.concat_eps_per_scale(all_eps)
                all_log_q = vae.concat_eps_per_scale(all_log_q)
                log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                kl_meter.update(kl_diag[0], 1)  # only the top scale

    utils.average_tensor(kl_meter.avg, args.distributed)
    return kl_meter.avg > 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nvae-diff/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='Synthetic',
                        choices=['ETT', 'P2012', 'MIMIC', 'Synthetic'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate_vae', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.02,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=1.0,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff', type=float, default=0.7,
                        help='The constant value used for max KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=4,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--log_sig_q_scale', type=float, default=25.,        # we used to use [-5, 5]
                        help='log sigma q is clamped into [-log_sig_q_scale, log_sig_q_scale].')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=2,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=200,
                        help='number of channels in latent variables per group')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=64,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=1,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=64,
                        help='number of channels in decoder')
    parser.add_argument('--channel_mult', nargs='+', type=int,
                        help='channel multiplier per scale', default=[1, 2])
    parser.add_argument('--num_postprocess_blocks', type=int, default=1,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--decoder_dist', type=str, default='normal', choices=['normal', 'dml', 'dl', 'bin'],
                        help='Distribution used in VAE decoder: Normal, Discretized Mix of Logistic,'
                             'Bernoulli, or discretized logistic.')
    parser.add_argument('--progressive_input_vae', type=str, default='none', choices=['none', 'input_skip'],
                        help='progressive type for input')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--autocast_train', action='store_true', default=True,
                        help='This flag enables FP16 in training.')
    parser.add_argument('--autocast_eval', action='store_true', default=True,
                        help='This flag enables FP16 in evaluation.')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node



    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = False
        utils.init_processes(0, size, main, args)

    print('finished')
