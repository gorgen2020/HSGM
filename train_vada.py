import argparse
import torch
import numpy as np
import os

from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from nvae import NVAE
from diffusion_discretized import DiffusionDiscretized
from diffusion_continuous import make_diffusion
try:
    from apex.optimizers import FusedAdam
except ImportError:
    # print("No Apex Available. Using PyTorch's native Adam. Install Apex for faster training.")
    from torch.optim import Adam as FusedAdam
from util.ema import EMA
from util import utils, datasets
from util.sr_utils import SpectralNormCalculator
from evaluate_diffusion import generate_samples_vada, elbo_evaluation, generate_samples_vada_reconstruction
from train_vae import infer_active_variables
from training_obj_joint import train_vada_joint, train_vada_join_frozen
from training_obj_disjoint import train_vada_disjoint
import pickle
import torch


import warnings
warnings.simplefilter("ignore", category=FutureWarning)

def main(args):
    # common initialization
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # Get data loaders.
    train_queue, valid_queue, test_queue, train_queue_no_shuffle = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    # load a pretrained NVAE only for VADA.
    load_vae, load_dae = False, False
    if args.vae_checkpoint != '' or args.vada_checkpoint != '':
        assert not (args.vae_checkpoint != '' and args.vada_checkpoint != ''), 'provide only 1 checkpoint'
        checkpoint_path = args.vada_checkpoint if args.vada_checkpoint != '' else args.vae_checkpoint
        logging.info('loading pretrained vae checkpoint:')
        logging.info(checkpoint_path)
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        stored_args = checkpoint['args']
        utils.override_architecture_fields(args, stored_args, logging)
        load_vae = True and not args.discard_vae_weights
        load_dae = False

    arch_instance_vae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)


    vae = NVAE(args, arch_instance_vae)
    if load_vae:
        logging.info('loading weights from vae checkpoint')
        # vae.load_state_dict(checkpoint['vae_state_dict'])
        vae.load_state_dict(checkpoint['vae_state_dict'], strict=False)
    vae = vae.cuda()
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(vae.parameters(), args.distributed)


############################################################################################
    # Frozen vae encoder parameters
    for param in vae.enc_tower.parameters():
        param.requires_grad = False
    for param in vae.enc_sampler.parameters():
        param.requires_grad = False
    for param in vae.nf_cells.parameters():
        param.requires_grad = False
    for param in vae.eps_conv.parameters():
        param.requires_grad = False


    decoder_params = list(vae.dec_tower.parameters()) + \
                     list(vae.post_process.parameters()) + \
                     list(vae.image_conditional.parameters()) + \
                     [vae.prior_ftr0]

    vae_optimizer = FusedAdam(decoder_params, args.learning_rate_vae, weight_decay=args.weight_decay, eps=1e-3)

    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)
#################################################################################
    # enable mixed prediction for vada
    args.mixed_prediction = True
    num_input_channels = vae.latent_structure()[0]
    dae = utils.get_dae_model(args, num_input_channels)
    if load_dae:
        logging.info('loading weights from dae checkpoint')
        dae.load_state_dict(checkpoint['dae_state_dict'])
    dae = dae.cuda()

    # for VESDE, run one epoch over data and get encodings and estimate sigma2_max based on Song's/Ermon's techniques.
    if args.sde_type == 'vesde':
        assert args.sigma2_min == args.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"
        args = utils.set_vesde_sigma_max(args, vae, train_queue, logging, args.distributed)

    diffusion_cont = make_diffusion(args)
    diffusion_disc = DiffusionDiscretized(args, diffusion_cont.var)

    logging.info('DAE: param size = %fM ', utils.count_parameters_in_M(dae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(dae.parameters(), args.distributed)

    dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae, weight_decay=args.weight_decay, eps=1e-4)
    # add EMA functionality to the optimizer
    dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)

    dae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_dae)

    # create SN calculator
    vae_sn_calculator = SpectralNormCalculator(custom_conv=True)  # NVAE consists of our own custom conv layer classes
    dae_sn_calculator = SpectralNormCalculator(custom_conv=args.custom_conv_dae)   # NCSN++ mode consists of pytorch conv layers
    if args.train_vae:
        vae_sn_calculator.add_conv_layers(vae)
        vae_sn_calculator.add_bn_layers(vae)
    dae_sn_calculator.add_conv_layers(dae)
    dae_sn_calculator.add_bn_layers(dae)

    grad_scalar = GradScaler(2**10)

    # if continue training from a checkpoint
    # useful when training is interrupted.
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        dae.load_state_dict(checkpoint['dae_state_dict'])
        # load dae
        dae = dae.cuda()
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])
        dae_scheduler.load_state_dict(checkpoint['dae_scheduler'])
        if 'dae_sn_calculator' in checkpoint:   # for backward compatibility
            dae_sn_calculator.load_state_dict(checkpoint['dae_sn_calculator'], torch.device("cuda"))
        # load vae
        vae.load_state_dict(checkpoint['vae_state_dict'])
        vae = vae.cuda()
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        if 'vae_sn_calculator' in checkpoint:   # for backward compatibility
            vae_sn_calculator.load_state_dict(checkpoint['vae_sn_calculator'], torch.device("cuda"))
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        best_score_fid = checkpoint['best_score_fid']
        best_score_nll = checkpoint['best_score_nll']
        logging.info('loaded the model at epoch %d.', init_epoch)
    else:
        global_step, epoch, init_epoch, best_score_fid, best_score_nll, best_imputation = 0, 0, 0, 1e10, 1e10, 1e10

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            dae_scheduler.step()
            vae_scheduler.step()

        # remove disabled latent variables by setting their mixing component to a small value
        if epoch == 0 and args.mixed_prediction and args.drop_inactive_var:
            logging.info('inferring active latent variables.')
            is_active = infer_active_variables(train_queue, vae, args, max_iter=1000)
            dae.mixing_logit.data[0, torch.logical_not(is_active), 0, 0] = -15
            dae.is_active = is_active.float().view(1, -1, 1, 1)

        # Logging.
        logging.info('epoch %d', epoch)
        if args.disjoint_training:
            # we may use disjoint training for update q with ema
            assert args.iw_sample_p != args.iw_sample_q or args.update_q_ema, \
                'disjoint training is for the case training objective of p and q are not the same unless q is ' \
                'updated with the EMA parameters.'
            assert args.iw_sample_q in ['ll_uniform', 'll_iw']
            assert args.train_vae, 'disjoint training is used when training both VAE and prior.'

            train_obj, global_step = train_vada_disjoint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                         grad_scalar, global_step, warmup_iters, writer, logging,
                                                         dae_sn_calculator, vae_sn_calculator, args)
        else:
            assert not args.update_q_ema, 'q can be training with EMA parameters of prior in disjoint training only.'
            if epoch < 100:
                print("Begin to train the latent diffusion models!")
                train_obj, global_step = train_vada_joint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                          grad_scalar, global_step, warmup_iters, writer, logging,
                                                          dae_sn_calculator, vae_sn_calculator, args)
                print('dae training, not vae!!')
            else:
                print("Post-train the VAE decoder!")
                train_obj, global_step = train_vada_join_frozen(diffusion_cont, train_queue, diffusion_disc, dae, dae_optimizer, vae, vae_optimizer,
                                                          grad_scalar, global_step, warmup_iters, writer, logging,
                                                          dae_sn_calculator, vae_sn_calculator, args)
                print('vae training, not dae!!')

        logging.info('train_loss %f', train_obj)
        writer.add_scalar('train/loss_epoch', train_obj, global_step)

        ###################################################################
        if epoch == (args.epochs - 1) or (epoch % 10 == 0) & (epoch >= 100):
            print(args.dataset)
            print('evaluation mode begin !!!')
            fast_ode_param = {'ode_eps': args.train_ode_eps, 'ode_solver_tol': args.train_ode_solver_tol}
            dae.eval()
            vae.eval()
            # switch to EMA parameters
            # generate samples
            n = int(np.floor(np.sqrt(min(64, args.batch_size))))  # cannot generate too many samples on big datasets
            num_samples = args.batch_size
            ########################################################
            print('validation result begin !!!')
            valid_result, valid_gt, valid_mask, _, _, _, imputation = generate_samples_vada(args, diffusion_cont, valid_queue,
                                                                                dae, diffusion_disc, vae,
                                                                                num_samples,
                                                                                enable_autocast=args.autocast_eval,
                                                                                prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0,
                                                                                is_train=False)

            if imputation < best_imputation:
                best_imputation = imputation
                print('saving the model.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'grad_scalar': grad_scalar.state_dict(), 'best_score_fid': best_score_fid,
                           'best_score_nll': best_score_nll,
                           'dae_state_dict': dae.state_dict(), 'dae_optimizer': dae_optimizer.optimizer.state_dict(),
                           'dae_scheduler': dae_scheduler.state_dict(), 'vae_state_dict': vae.state_dict(),
                           'vae_optimizer': vae_optimizer.state_dict(), 'vae_scheduler': vae_scheduler.state_dict(),
                           'vae_sn_calculator': vae_sn_calculator.state_dict(),
                           'dae_sn_calculator': dae_sn_calculator.state_dict()}
                torch.save(content, checkpoint_file)


            if epoch == (args.epochs - 1):
                print('loading the best model.')
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

                init_epoch = checkpoint['epoch']
                epoch = init_epoch
                dae.load_state_dict(checkpoint['dae_state_dict'])
                # load dae
                dae = dae.cuda()

                if 'param_groups' in checkpoint.get('dae_optimizer', {}):
                    dae_optimizer.optimizer.load_state_dict(checkpoint['dae_optimizer'])
                else:
                    print("Warning: dae_optimizer param_groups not found. Skipping optimizer state load.")

                dae_scheduler.load_state_dict(checkpoint['dae_scheduler'])
                if 'dae_sn_calculator' in checkpoint:  # for backward compatibility
                    dae_sn_calculator.load_state_dict(checkpoint['dae_sn_calculator'], torch.device("cuda"))
                # load vae
                vae.load_state_dict(checkpoint['vae_state_dict'])
                vae = vae.cuda()
                vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
                vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
                if 'vae_sn_calculator' in checkpoint:  # for backward compatibility
                    vae_sn_calculator.load_state_dict(checkpoint['vae_sn_calculator'], torch.device("cuda"))
                grad_scalar.load_state_dict(checkpoint['grad_scalar'])
                global_step = checkpoint['global_step']
                best_score_fid = checkpoint['best_score_fid']
                best_score_nll = checkpoint['best_score_nll']
                print('loaded the model at epoch ', init_epoch)
                ################################################################
                print('validation result begin !!!')
                valid_result, valid_gt, valid_mask, _, _, _, imputation = generate_samples_vada(args, diffusion_cont,
                                                                                                valid_queue,
                                                                                                dae, diffusion_disc,
                                                                                                vae,
                                                                                                num_samples,
                                                                                                enable_autocast=args.autocast_eval,
                                                                                                prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0,
                                                                                                is_train=False)

                valid_result = torch.cat(valid_result, dim=0)
                valid_result = valid_result.cpu().numpy()
                valid_gt = torch.cat(valid_gt, dim=0)
                valid_gt = valid_gt.cpu().numpy()
                valid_mask = torch.cat(valid_mask, dim=0)
                valid_mask = valid_mask.cpu().numpy()

                np.savez('valid_result' + args.dataset + '.npz', valid_result=valid_result, valid_gt=valid_gt,
                     valid_mask=valid_mask)

                ###############################################################
                print('Reconstruction on test dataset begin !!!')
                num_samples1 = 100
                test_result, test_gt, test_mask, _, _, _, imputation = generate_samples_vada_reconstruction(args, diffusion_cont, test_queue, dae,
                                                                                 diffusion_disc, vae, num_samples1,
                                                                                 enable_autocast=args.autocast_eval,
                                                                                 prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0,
                                                                                 is_train=False)
                with open("latent_mae_Synthetic_reconstruction.txt", "w") as f:
                    f.write(str(imputation))
                    f.write('\n')

                print('Imputation on test dataset begin !!!')
                test_result, test_gt, test_mask, _, _, _, imputation = generate_samples_vada(args, diffusion_cont, test_queue, dae,
                                                                                 diffusion_disc, vae, num_samples1,
                                                                                 enable_autocast=args.autocast_eval,
                                                                                 prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0,
                                                                                 is_train=False)
                with open("latent_mae_Synthetic_imputation.txt", "w") as f:
                    f.write(str(imputation))
                    f.write('\n')

                test_result = torch.cat(test_result, dim=0)
                test_result = test_result.cpu().numpy()
                test_gt = torch.cat(test_gt, dim=0)
                test_gt = test_gt.cpu().numpy()
                test_mask = torch.cat(test_mask, dim=0)
                test_mask = test_mask.cpu().numpy()

                np.savez('test_result' + args.dataset +'.npz', test_result=test_result, test_gt=test_gt,
                     test_mask=test_mask)
                #################################################################
                print('Imputation on training result begin !!!')
                train_result, train_gt, train_mask,  _, _, _, imputation = generate_samples_vada(args, diffusion_cont, train_queue_no_shuffle, dae, diffusion_disc, vae, num_samples,
                                                            enable_autocast=args.autocast_eval,
                                                            prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0, is_train=False)

                train_result = torch.cat(train_result, dim=0)
                train_result = train_result.cpu().numpy()
                train_gt = torch.cat(train_gt, dim=0)
                train_gt = train_gt.cpu().numpy()
                train_mask = torch.cat(train_mask, dim=0)
                train_mask = train_mask.cpu().numpy()

                np.savez('train_result' + args.dataset +'.npz', train_result=train_result, train_gt=train_gt,
                         train_mask=train_mask)

                # restore the original sequences
                if args.dataset == 'Synthetic':
                    predicted_train_result = np.load('train_result' + args.dataset +'.npz')['train_result']
                    predicted_train_result = np.mean(predicted_train_result, axis=1)
                    predicted_train_gt = np.load('train_result' + args.dataset +'.npz')['train_gt']
                    predicted_train_mask = np.load('train_result' + args.dataset +'.npz')['train_mask']

                    predicted_valid_result = np.load('valid_result' + args.dataset +'.npz')['valid_result']
                    predicted_valid_result = np.mean(predicted_valid_result, axis=1)
                    predicted_valid_gt = np.load('valid_result' + args.dataset +'.npz')['valid_gt']
                    predicted_valid_mask = np.load('valid_result' + args.dataset +'.npz')['valid_mask']

                    predicted_test_result = np.load('test_result' + args.dataset +'.npz')['test_result']
                    predicted_test_result = np.mean(predicted_test_result, axis=1)
                    predicted_test_gt = np.load('test_result' + args.dataset +'.npz')['test_gt']
                    predicted_test_mask = np.load('test_result' + args.dataset +'.npz')['test_mask']
                    ############################
                    predicted_train_gt = predicted_train_gt.reshape(-1,4)
                    predicted_valid_gt = predicted_valid_gt.reshape(-1,4)
                    predicted_test_gt = predicted_test_gt.reshape(-1,4)
                    merged_gt = np.concatenate((predicted_train_gt, predicted_valid_gt, predicted_test_gt), axis=0)
                    ###############################################
                    predicted_train_result = predicted_train_result.reshape(-1,4)
                    predicted_valid_result = predicted_valid_result.reshape(-1,4)
                    predicted_test_result = predicted_test_result.reshape(-1,4)
                    merged_vae_data = np.concatenate((predicted_train_result, predicted_valid_result, predicted_test_result), axis=0)
                    ##################################################
                    predicted_train_mask = predicted_train_mask.reshape(-1,4)
                    predicted_valid_mask = predicted_valid_mask.reshape(-1,4)
                    predicted_test_mask = predicted_test_mask.reshape(-1,4)
                    predicted_gt_mask = np.concatenate((predicted_train_mask, predicted_valid_mask, predicted_test_mask), axis=0)
                    #############################################
                    np.savez('latent_output_' + args.dataset + '.npz', total_result=merged_vae_data, total_gt=merged_gt, total_mask=predicted_gt_mask)
                elif args.dataset == 'ETT':
                    train_result = np.load('train_result' + args.dataset +'.npz')['train_result']
                    train_gt = np.load('train_result' + args.dataset +'.npz')['train_gt']
                    train_mask = np.load('train_result' + args.dataset +'.npz')['train_mask']

                    valid_result = np.load('valid_result' + args.dataset +'.npz')['valid_result']
                    valid_gt = np.load('valid_result' + args.dataset +'.npz')['valid_gt']
                    valid_mask = np.load('valid_result' + args.dataset +'.npz')['valid_mask']

                    test_result = np.load('test_result' + args.dataset +'.npz')['test_result']
                    test_gt = np.load('test_result' + args.dataset +'.npz')['test_gt']
                    test_mask = np.load('test_result' + args.dataset +'.npz')['test_mask']
                    ###############################################
                    total_result = np.concatenate((train_result, valid_result, test_result), axis=0)
                    total_gt = np.concatenate((train_gt, valid_gt, test_gt), axis=0)
                    total_mask = np.concatenate((train_mask, valid_mask, test_mask), axis=0)

                    total_result = total_result[:,0,:,0:7] - 4.130494
                    total_gt = total_gt[:,0,:,0:7] - 4.130494
                    total_mask = total_mask[:,0,:,0:7]

                    np.savez('latent_output_' + args.dataset + '.npz', total_result=total_result, total_gt=total_gt, total_mask=total_mask)

                elif args.dataset == 'P2012':
                    path = "./data/physio_missing0.5_seed1_std.pk"
                    with open(path, "rb") as f:
                        observed_values, observed_masks, gt_masks, range_, mean, std = pickle.load(f)

                    indlist = np.arange(observed_values.shape[0])

                    seed = 1
                    np.random.seed(seed)
                    np.random.shuffle(indlist)

                    # 5-fold test
                    start = 0
                    end = (int)(0.2 * observed_values.shape[0])
                    test_index = indlist[start:end]
                    remain_index = np.delete(indlist, np.arange(start, end))

                    np.random.seed(seed)
                    np.random.shuffle(remain_index)
                    num_train = (int)(observed_values.shape[0] * 0.7)
                    train_index = remain_index[:num_train]
                    valid_index = remain_index[num_train:]

                    train_result = np.load('train_result' + args.dataset + '.npz')['train_result']
                    train_gt = np.load('train_result' + args.dataset + '.npz')['train_gt']
                    train_mask = np.load('train_result' + args.dataset + '.npz')['train_mask']

                    valid_result = np.load('valid_result' + args.dataset + '.npz')['valid_result']
                    valid_gt = np.load('valid_result' + args.dataset + '.npz')['valid_gt']
                    valid_mask = np.load('valid_result' + args.dataset + '.npz')['valid_mask']

                    test_result = np.load('test_result' + args.dataset + '.npz')['test_result']
                    test_gt = np.load('test_result' + args.dataset + '.npz')['test_gt']
                    test_mask = np.load('test_result' + args.dataset + '.npz')['test_mask']

                    test_result = test_result[:,0,:,0:observed_values.shape[2]]
                    test_gt = test_gt[:,0,:,0:observed_values.shape[2]]
                    test_mask = test_mask[:,0,:,0:observed_values.shape[2]]

                    valid_result = valid_result[:,0,:,0:observed_values.shape[2]]
                    valid_gt = valid_gt[:,0,:,0:observed_values.shape[2]]
                    valid_mask = valid_mask[:,0,:,0:observed_values.shape[2]]

                    train_result = train_result[:,0,:,0:observed_values.shape[2]]
                    train_gt = train_gt[:,0,:,0:observed_values.shape[2]]
                    train_mask = train_mask[:,0,:,0:observed_values.shape[2]]


                    whole_result = np.concatenate((test_result, train_result, valid_result ), axis=0)

                    whole_gt = np.concatenate((test_gt, train_gt, valid_gt ), axis=0)
                    whole_mask = np.concatenate((test_mask, train_mask, valid_mask ), axis=0)

                    seq = np.concatenate((test_index, train_index, valid_index ), axis=0)

                    whole_result = whole_result[np.argsort(seq)]
                    whole_gt = whole_gt[np.argsort(seq)]
                    whole_mask = whole_mask[np.argsort(seq)]

                    np.savez('latent_output_' + args.dataset + '.npz', total_result=whole_result, total_gt=whole_gt, total_mask=whole_mask)

                elif args.dataset == 'MIMIC':
                    train_result = np.load('train_result' + args.dataset + '.npz')['train_result']
                    train_gt = np.load('train_result' + args.dataset + '.npz')['train_gt']
                    train_mask = np.load('train_result' + args.dataset + '.npz')['train_mask']

                    valid_result = np.load('valid_result' + args.dataset + '.npz')['valid_result']
                    valid_gt = np.load('valid_result' + args.dataset + '.npz')['valid_gt']
                    valid_mask = np.load('valid_result' + args.dataset + '.npz')['valid_mask']

                    test_result = np.load('test_result' + args.dataset + '.npz')['test_result']
                    test_gt = np.load('test_result' + args.dataset + '.npz')['test_gt']
                    test_mask = np.load('test_result' + args.dataset + '.npz')['test_mask']
                    ################################################
                    train_mask = 1 - train_mask

                    train_result = np.mean(train_result, axis=1)
                    train_gt = np.squeeze(train_gt, axis=1)
                    train_mask = np.squeeze(train_mask, axis=1)
                    ################################################
                    valid_mask = 1 - valid_mask

                    valid_result = np.mean(valid_result, axis=1)
                    valid_gt = np.squeeze(valid_gt, axis=1)
                    valid_mask = np.squeeze(valid_mask, axis=1)
                    ##############################################
                    test_mask = 1 - test_mask

                    test_result = np.mean(test_result, axis=1)
                    test_gt = np.squeeze(test_gt, axis=1)
                    test_mask = np.squeeze(test_mask, axis=1)
                    ########################################
                    result = np.concatenate([train_result, valid_result, test_result], axis=0)
                    result_gt = np.concatenate([train_gt, valid_gt, test_gt], axis=0)
                    result_mask = np.concatenate([train_mask, valid_mask, test_mask], axis=0)
                    std = np.load('MIMIC4_mean_std.npz')['std']

                    result = result * std
                    result_gt = result_gt * std

                    np.savez('latent_output_' + args.dataset + '.npz', result=result, total_gt=result_gt,
                             total_mask=result_mask)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

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

    parser.add_argument('--data', type=str, default='/tmp/nvae-diff/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate_vae', type=float, default=1e-4,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-5,
                        help='min learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay factor')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm_vae', type=float, default=0.02,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--epochs', type=int, default=120,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--grad_clip_max_norm', type=float, default=0.,
                        help='The maximum norm used in gradient norm clipping (0 applies no clipping).')
    # Diffusion
    parser.add_argument('--learning_rate_dae', type=float, default=3e-4,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_dae', type=float, default=3e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay_norm_dae', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--custom_conv_dae', action='store_true', default=False,
                        help='Set this argument if conv layers in the SGM prior are custom layers from NVAE.')
    parser.add_argument('--num_channels_dae', type=int, default=288,
                        help='number of initial channels in denosing model')
    parser.add_argument('--num_scales_dae', type=int, default=2,
                        help='number of spatial scales in denosing model')
    parser.add_argument('--num_cell_per_scale_dae', type=int, default=8,
                        help='number of cells per scale')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='dimension used for time embeddings')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='number of diffusion steps')
    #############################################################################
    parser.add_argument('--sigma2_0', type=float, default=0.0,
                        help='initial SDE variance at t=0 (sort of represents Normal perturbation of input data)')
    parser.add_argument('--beta_start', type=float, default=0.1,
                        help='initial beta variance value')
    parser.add_argument('--beta_end', type=float, default=20.0,
                        help='final beta variance value')
    parser.add_argument('--vpsde_power', type=int, default=2,
                        help='vpsde power for power-VPSDE')
    parser.add_argument('--sigma2_min', type=float, default=1e-4,
                        help='initial beta variance value')
    parser.add_argument('--sigma2_max', type=float, default=0.99,
                        help='final beta variance value')
    parser.add_argument('--sde_type', type=str, default='vpsde',
                        choices=['geometric_sde', 'vpsde', 'sub_vpsde', 'vesde'],
                        help='what kind of sde type to use when training/evaluating in continuous manner.')
    ###############################################################################################################
    parser.add_argument('--train_ode_eps', type=float, default=1e-2,
                        help='ODE can only be integrated up to some epsilon > 0.')

    parser.add_argument('--train_ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--eval_ode_eps', type=float, default=1e-5,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--eval_ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--time_eps', type=float, default=1e-2,
                        help='During training, t is sampled in [time_eps, 1.].')
    parser.add_argument('--denoising_stddevs', type=str, default='beta', choices=['learn', 'beta', 'beta_post'],
                        help='enables learning the conditional VAE decoder distribution standard deviations')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout probability applied to the denosing model')

    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='A dir to store fid related files')
    parser.add_argument('--mixing_logit_init', type=float, default=-6,
                        help='The initial logit for mixing coefficient.')
    parser.add_argument('--embedding_type', type=str, choices=['positional', 'fourier'], default='positional',
                        help='Type of time embedding')
    parser.add_argument('--embedding_scale', type=float, default=1000,
                        # 'fourier':16, 'positional':1000, backward compatible: 1.
                        help='Embedding scale that is used for rescaling time')
    # NCSN++
    parser.add_argument('--dae_arch', type=str, default='ncsnpp', choices=['ncsnpp'],
                        help='Switch between different DAE architectures.')
    parser.add_argument('--fir', action='store_true', default=False,
                        help='Enable FIR upsampling/downsampling')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='none', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    # VADA
    parser.add_argument('--vae_checkpoint', type=str, default='/tmp/nvae-diff/expr/exp/checkpoint_best.pt',
                        help='Pretrained VAE checkpoint.')
    parser.add_argument('--vada_checkpoint', type=str, default='',
                        help='Pretrained VADA checkpoint.')
    parser.add_argument('--discard_vae_weights', action='store_true', default=False,
                        help='set true to ignore the vae weights from the checkpoint.')
    parser.add_argument('--discard_dae_weights', action='store_true', default=True,
                        help='set true to ignore the dae weights from the checkpoint.')
    parser.add_argument('--train_vae', action='store_true', default=False,
                        help='set true to train the vae model.')
    #########################################################################
    parser.add_argument('--iw_sample_p', type=str, default='ll_iw', choices=['ll_uniform', 'll_iw',
                        'drop_all_uniform', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw', 'drop_sigma2t_uniform'],
                        help='Specifies the weighting mechanism used for training p (sgm prior) and whether or not to use importance sampling')
    parser.add_argument('--iw_sample_q', type=str, default='ll_iw', choices=['reweight_p_samples', 'll_uniform', 'll_iw'],
                        help='Specifies the weighting mechanism used for training q (vae) and whether or not to use importance sampling. '
                             'reweight_p_samples indicates reweight the t samples generated for the prior as done in Algorithm 3.')
    parser.add_argument('--iw_subvp_like_vp_sde', action='store_true', default=False,
                        help='Only relevant when using Sub-VPSDE. When true, use VPSDE-based IW distributions.')
    #########################################################################
    parser.add_argument('--no_autograd_jvp', action='store_true', default=False,
                        help='Set to true to use backward() instead of grad(). '
                             'Suitable for models with gradient checkpointing.')
    parser.add_argument('--apply_sqrt2_res', action='store_true', default=False,
                        help='Enable mixing residual cells with 1/sqrt(2).')
    parser.add_argument('--drop_inactive_var', action='store_true', default=False,
                        help='Drops inactive latent variables.')
    parser.add_argument('--skip_final_eval', action='store_true', default=False,
                        help='set true to skip the final eval.')
    parser.add_argument('--disjoint_training', action='store_true', default=False,
                        help='When p (sgm prior) and q (vae) have different objectives, trains them in two separate forward calls (Algorithm 2).')
    parser.add_argument('--update_q_ema', action='store_true', default=False,
                        help='Enables updating q with EMA parameters of prior.')
    # second stage VADA KL annealing
    parser.add_argument('--cont_kl_anneal', action='store_true', default=False,
                        help='If true, we continue KL annealing using below setup when training.')
    parser.add_argument('--kl_anneal_portion_vada', type=float, default=0.1,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion_vada', type=float, default=0.0,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff_vada', type=float, default=0.7,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff_vada', type=float, default=1.,
                        help='The constant value used for max KL coeff')
    parser.add_argument('--kl_balance_vada', action='store_true', default=False,
                        help='If true, we use KL balancing during VADA KL annealing.')
    # DDP.
    parser.add_argument('--autocast_train', action='store_true', default=False,
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
