import torch
import numpy as np
from util import utils
from torch.cuda.amp import autocast
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def train_vada_join_frozen(diffusion_cont, train_queue, diffusion, dae, dae_optimizer, vae, vae_optimizer, grad_scalar, global_step,
                     warmup_iters, writer, logging, dae_sn_calculator, vae_sn_calculator, args):
    """ This function implements Algorithm 1, 2, 3 together from the LSGM paper. If you are trying to understand
    how this function works for the first time, I would suggest checking training_obj_disjoint.py that implements
    Algorithm 3 in a slightly simpler way. """


    dae.eval()
    vae.train(True)
    for param in vae.parameters():
        param.requires_grad = True
        # Frozen encoder parameters
    for param in vae.enc_tower.parameters():
        param.requires_grad = False
    for param in vae.enc_sampler.parameters():
        param.requires_grad = False
    for param in vae.nf_cells.parameters():
        param.requires_grad = False
    ######################################
    alpha_i = utils.kl_balancer_coeff(num_scales=vae.num_latent_scales,
                                      groups_per_scale=vae.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()


    for step, (x, x_gt, mask, ob_mask) in enumerate(train_queue):
        x = utils.common_x_operations(x, args.num_x_bits)
        x_gt = utils.common_x_operations(x_gt, args.num_x_bits)
        mask = utils.common_x_operations(mask, args.num_x_bits)
        ob_mask = utils.common_x_operations(ob_mask, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in vae_optimizer.param_groups:
                param_group['lr'] = lr

        vae_optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            sigma_mask = torch.tensor(False, dtype=torch.bool)
            logits, all_log_q, all_eps, all_dist = vae(x, sigma_mask)
            log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
            ############################################################################
            eps2 = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale

            noise_list = [torch.randn_like(tensor, device='cuda') for tensor in all_eps]
            for ii in range(len(all_dist)):
                all_eps[ii] = all_dist[ii].sample_diff(0)

            eps = vae.concat_eps_per_scale(all_eps)[0]# prior is applied at the top scale
            noisee = vae.concat_eps_per_scale(noise_list)[0]  #

            remaining_neg_log_p_total, _ = utils.cross_entropy_normal(
                all_eps[vae.num_groups_per_scale:])
            num_samples = x.shape[0]
            enable_autocast=args.autocast_eval
            shape = [dae.num_input_channels, dae.input_size, dae.input_size]
            vae_temp = 1.0

            temp = 0.3
            # initialize sample
            x_noisy_size = eps.shape
            x_noisy = noisee * temp
            ################################################
            t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                diffusion_cont.iw_quantities(args.batch_size, args.time_eps, 'll_iw_denoise', args.iw_subvp_like_vp_sde)

            x_noisy = m_t_q[0] * eps + torch.sqrt(var_t_q[0]) * x_noisy

            eps1, nfe, time_ode_solve = diffusion_cont.sample_model_ode(x, ob_mask, dae, num_samples, shape, 1e-2, 1e-2, enable_autocast, temp, x_noisy)

            decomposed_eps = vae.decompose_eps(eps1)
            decomposed_eps2 = vae.decompose_eps(eps2)

            sigma_mask = torch.tensor(False, dtype=torch.bool)
            image = vae.sample(num_samples, vae_temp, sigma_mask,  decomposed_eps2, decomposed_eps, enable_autocast)
            # for ETT and synthetic dataset, pay attention
            normalized_samples = (x - image) * ob_mask
            normalized_samples = normalized_samples.mean(dim=1)  # Compute mean over the second dimension

            vae_recon_loss = torch.sum(torch.abs(normalized_samples))
            output = vae.decoder_output(logits, sigma_mask)
            kl_coeff = utils.kl_coeff(global_step, 0.0001 * args.num_total_iter,
                                      0.0001 * args.num_total_iter, 0.0001,
                                      0.7)

            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = vae_recon_loss
            loss = torch.mean(nelbo_batch)
            norm_loss = vae_sn_calculator.spectral_norm_parallel()
            bn_loss = vae_sn_calculator.batchnorm_loss()
            # get spectral regularization coefficient (lambda)

            wdn_coeff = 0.02
            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        # loss_recon.update(loss_recon_loss.data, 1)
        grad_scalar.scale(loss).backward()
        utils.average_gradients(vae.parameters(), args.distributed)
        grad_scalar.step(vae_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

    if (global_step + 1) % 100 == 0:
        if (global_step + 1) % 10000 == 0:  # reduced frequency
            n = int(np.floor(np.sqrt(x.size(0))))
            x_img = x[:n * n]
            output_img = output.mean()
            output_img = output_img[:n * n]
            x_tiled = utils.tile_image(x_img, n)
            output_tiled = utils.tile_image(output_img, n)
            in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
            in_out_tiled = utils.unsymmetrize_image_data(in_out_tiled)
            writer.add_image('reconstruction', in_out_tiled, global_step)

        # norm
        writer.add_scalar('train/norm_loss', norm_loss, global_step)
        writer.add_scalar('train/bn_loss', bn_loss, global_step)
        writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

        utils.average_tensor(nelbo.avg, args.distributed)
        logging.info('train %d %f', global_step, nelbo.avg)
        writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
        writer.add_scalar('train/lr', vae_optimizer.state_dict()[
            'param_groups'][0]['lr'], global_step)
        writer.add_scalar('train/nelbo_iter', loss, global_step)
        writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)

        writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
        total_active = 0
        for i, kl_diag_i in enumerate(kl_diag):
            utils.average_tensor(kl_diag_i, args.distributed)
            num_active = torch.sum(kl_diag_i > 0.1).detach()
            total_active += num_active

            # kl_ceoff
            writer.add_scalar('kl/active_%d' % i, num_active, global_step)
            writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
            writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
        writer.add_scalar('kl/total_active', total_active, global_step)
    global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def train_vada_joint(train_queue, diffusion, dae, dae_optimizer, vae, vae_optimizer, grad_scalar, global_step,
                     warmup_iters, writer, logging, dae_sn_calculator, vae_sn_calculator, args):
    """ This function implements Algorithm 1, 2, 3 together from the LSGM paper. If you are trying to understand
    how this function works for the first time, I would suggest checking training_obj_disjoint.py that implements
    Algorithm 3 in a slightly simpler way. """
    args.train_vae = False
    alpha_i = utils.kl_balancer_coeff(num_scales=vae.num_latent_scales,
                                      groups_per_scale=vae.groups_per_scale, fun='square')
    tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema = start_meters()

    dae.train(True)
    vae.train(False)

    for step, (x, x_gt, mask, ob_mask)in enumerate(train_queue):
        # warm-up lr
        update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer)
        x = utils.common_x_operations(x, args.num_x_bits)
        x_gt = utils.common_x_operations(x_gt, args.num_x_bits)
        mask = utils.common_x_operations(mask, args.num_x_bits)
        ob_mask = utils.common_x_operations(ob_mask, args.num_x_bits)
        # print(mask.shape)
        # dae_optimizer.zero_grad()
        dae_optimizer.optimizer.zero_grad()

        # vae_optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            # apply vae:
            with torch.set_grad_enabled(args.train_vae):
                with torch.no_grad():
                    sigma_mask = torch.tensor(False, dtype=torch.bool)
                    logits, all_log_q, all_eps, all_dist = vae(x, sigma_mask)
                    output = vae.decoder_output(logits, sigma_mask)
                    log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                    recon_loss, mae = utils.reconstruction_loss(output, x, ob_mask, crop=vae.crop_output)
                    balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)

                    # prior is applied at the top scale
                    noise_list = [torch.randn_like(tensor, device='cuda') for tensor in all_eps]
                    for ii in range(len(all_dist)):
                        all_eps[ii] = all_dist[ii].sample_diff(0)

            eps = vae.concat_eps_per_scale(all_eps)[0]
            noisee = vae.concat_eps_per_scale(noise_list)[0] * 0.3

            # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
            # in case we want to train q (vae) with another batch using a different sampling scheme for times t
            t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_q, args.iw_subvp_like_vp_sde)
            eps_t_q = diffusion.sample_q(eps, noisee, var_t_q, m_t_q, t_q)

            # run the score model
            # eps_t_q.detach()
            pred_params = dae(eps_t_q, t_q, x, ob_mask)
            # params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)
            l2_term = torch.square(pred_params - noisee)

            # l2_term_p, l2_term_q = torch.chunk(l2_term, chunks=2, dim=0)
            p_objective = torch.sum(l2_term, dim=[1, 2, 3])
            cross_entropy_per_var = l2_term
            cross_entropy_per_var += diffusion.cross_entropy_const(args.time_eps)
            all_neg_log_p = vae.decompose_eps(cross_entropy_per_var)
            kl_all_list, kl_vals_per_group, kl_diag_list = utils.kl_per_group_vada(all_log_q, all_neg_log_p)

            # kl coefficient
            if args.cont_kl_anneal:
                kl_coeff = utils.kl_coeff(step=global_step,
                                          total_step=args.kl_anneal_portion_vada * args.num_total_iter,
                                          constant_step=args.kl_const_portion_vada * args.num_total_iter,
                                          min_kl_coeff=args.kl_const_coeff_vada,
                                          max_kl_coeff=args.kl_max_coeff_vada)
            else:
                kl_coeff = 1.0

            # nelbo loss with kl balancing
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all_list, kl_coeff, kl_balance=args.kl_balance_vada, alpha_i=alpha_i)
            nelbo_loss = balanced_kl + recon_loss

            # compute regularization terms
            regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = vae_regularization(args, vae_sn_calculator)
            regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff = dae_regularization(args, dae_sn_calculator)

            # regularization = regularization_p + regularization_q
            q_loss = torch.mean(nelbo_loss) + regularization_p + regularization_q   # vae loss
            p_loss = torch.mean(p_objective)                   # sgm prior loss

            # backpropagate q_loss for vae and update vae params, if trained
            if args.train_vae:
                grad_scalar.scale(q_loss).backward(
                    retain_graph=utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q))
                utils.average_gradients(vae.parameters(), args.distributed)
                if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                    grad_scalar.unscale_(vae_optimizer)
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=args.grad_clip_max_norm)
                grad_scalar.step(vae_optimizer)

            # if we use different p and q objectives or are not training the vae, discard gradients and backpropagate p_loss
            if utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q) or not args.train_vae:
                if args.train_vae:
                    # discard current gradients computed by weighted loss for VAE
                    dae_optimizer.zero_grad()

                # compute gradients with unweighted loss
                grad_scalar.scale(p_loss).backward()

            # update dae parameters
            utils.average_gradients(dae.parameters(), args.distributed)
            if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                grad_scalar.unscale_(dae_optimizer)
                torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=args.grad_clip_max_norm)
            grad_scalar.step(dae_optimizer)

            # update grade scalar
            grad_scalar.update()
        # Bookkeeping!
        # update average meters
        tr_loss_meter.update(p_loss.data, 1)


        if (global_step + 1) % 200 == 0:
            writer.add_scalar('train/lr_dae', dae_optimizer.optimizer.state_dict()['param_groups'][0]['lr'])
            writer.add_scalar('train/lr_vae', vae_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/p_loss', p_loss - regularization_p, global_step)
            writer.add_scalar('train/norm_loss_dae', dae_norm_loss, global_step)
            writer.add_scalar('train/bn_loss_dae', dae_bn_loss, global_step)
            writer.add_scalar('train/norm_coeff_dae', dae_wdn_coeff, global_step)

        global_step += 1

    # write at the end of epoch
    epoch_logging(args, writer, global_step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema)

    utils.average_tensor(tr_loss_meter.avg, args.distributed)
    return tr_loss_meter.avg, global_step


def vae_regularization(args, vae_sn_calculator):
    regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = 0., 0., 0., args.weight_decay_norm_vae
    if args.train_vae:
        vae_norm_loss = vae_sn_calculator.spectral_norm_parallel()
        vae_bn_loss = vae_sn_calculator.batchnorm_loss()
        regularization_q = (vae_norm_loss + vae_bn_loss) * vae_wdn_coeff

    return regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff


def dae_regularization(args, dae_sn_calculator):
    dae_wdn_coeff = args.weight_decay_norm_dae
    dae_norm_loss = dae_sn_calculator.spectral_norm_parallel()
    dae_bn_loss = dae_sn_calculator.batchnorm_loss()
    regularization_p = (dae_norm_loss + dae_bn_loss) * dae_wdn_coeff

    return regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff


def update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer):
    if global_step < warmup_iters:
        lr = args.learning_rate_dae * float(global_step) / warmup_iters
        for param_group in dae_optimizer.param_groups:
            param_group['lr'] = lr

        if args.train_vae:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in vae_optimizer.param_groups:
                param_group['lr'] = lr


def start_meters():
    tr_loss_meter = utils.AvgrageMeter()
    vae_recon_meter = utils.AvgrageMeter()
    vae_kl_meter = utils.AvgrageMeter()
    vae_nelbo_meter = utils.AvgrageMeter()
    kl_per_group_ema = utils.AvgrageMeter()
    return tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema


def epoch_logging(args, writer, step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema):
    utils.average_tensor(vae_recon_meter.avg, args.distributed)
    utils.average_tensor(vae_kl_meter.avg, args.distributed)
    utils.average_tensor(vae_nelbo_meter.avg, args.distributed)
    utils.average_tensor(tr_loss_meter.avg, args.distributed)
    utils.average_tensor(kl_per_group_ema.avg, args.distributed)

    writer.add_scalar('epoch/vae_recon', vae_recon_meter.avg, step)
    writer.add_scalar('epoch/vae_kl', vae_kl_meter.avg, step)
    writer.add_scalar('epoch/vae_nelbo', vae_nelbo_meter.avg, step)
    writer.add_scalar('epoch/total_loss', tr_loss_meter.avg, step)
