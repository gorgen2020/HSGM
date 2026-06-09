import numpy as np
import torch
from timeit import default_timer as timer

from util import utils
from torch.cuda.amp import autocast
from diffusion_continuous import DiffusionBase
from diffusion_discretized import DiffusionDiscretized
import pickle
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def create_generator_vada(dae, diffusion, vae, batch_size, num_total_samples, enable_autocast, ode_param_dict, prior_var, temp, vae_temp):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    shape = [dae.num_input_channels, dae.input_size, dae.input_size]
    for i in range(num_iters):
        with torch.no_grad():
            if ode_param_dict is None:
                eps = diffusion.run_denoising_diffusion(dae, batch_size, shape, temp, enable_autocast, is_image=False, prior_var=prior_var)
            else:
                eps, _, _ = diffusion.sample_model_ode(dae, batch_size, shape, ode_param_dict['ode_eps'],
                                                         ode_param_dict['ode_solver_tol'], enable_autocast, temp)
            decomposed_eps = vae.decompose_eps(eps)
            image = vae.sample(batch_size, vae_temp, decomposed_eps, enable_autocast)
        yield image.float()


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

def generate_samples_vada(args, diffusion_cont, val_queue, dae, diffusion, vae, num_samples, enable_autocast, ode_eps=None, ode_solver_tol=None,
                          ode_sample=False, prior_var=1.0, is_train=False, temp=1.0, vae_temp=1.0, noise=None):
    shape = [dae.num_input_channels, dae.input_size, dae.input_size]

    total_mae = 0
    total_evaluate_point = 0
    total_rmse = 0


    with torch.no_grad():
        if ode_sample:
            assert isinstance(diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
            assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
            assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
            start = timer()
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(dae, num_samples, shape, ode_eps, ode_solver_tol, enable_autocast, temp, noise)

            decomposed_eps = vae.decompose_eps(eps)
            image = vae.sample(num_samples, vae_temp, decomposed_eps, enable_autocast)
            end = timer()
            sampling_time = end - start
            # average over GPUs
            nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
            sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
            time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
            utils.average_tensor(nfe_torch, True)
            utils.average_tensor(sampling_time_torch, True)
            utils.average_tensor(time_ode_solve_torch, True)

        else:
            assert isinstance(diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
            assert noise is None, 'Noise is not used in ancestral sampling.'
            #########################################################################
            vae.eval()
            dae.eval()
            valid_result = []
            valid_gt = []
            valid_mask = []

            all_generated_samples = []
            all_target = []
            all_evalpoint = []

            cc = 0
            with torch.no_grad():
                # for step, x in enumerate(val_queue):
                for step, (x, x_gt, mask, ob_mask) in enumerate(tqdm(val_queue, total=len(val_queue))):

                    # print('Batch number is ' + str(cc))
                    cc = cc + 1
                    num_samples1 = x.shape[0]
                    x = utils.common_x_operations(x, 8)
                    x_gt = utils.common_x_operations(x_gt, 8)
                    mask = utils.common_x_operations(mask, 8)
                    image_ls = []

                    B, _, K, L = x.shape
                    imputed_samples = torch.zeros(B, num_samples, K, L).to(x.device)
                    for k in range(num_samples):
                        with autocast(enabled=True):
                            # apply vae:
                            sigma_mask = torch.tensor(False, dtype=torch.bool)
                            if is_train==False:
                                logits, all_log_q, all_eps, all_dist = vae(x, sigma_mask)
                            else:
                                logits, all_log_q, all_eps, all_dist = vae(x_gt, sigma_mask)


                            eps2 = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                            remaining_neg_log_p_total, _ = utils.cross_entropy_normal(
                                all_eps[vae.num_groups_per_scale:])

                            ############################################################################
                            noise_list = [torch.randn_like(tensor, device='cuda') for tensor in all_eps]
                            for ii in range(len(all_dist)):
                                all_eps[ii] = all_dist[ii].sample_diff(0)

                            eps = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                            noisee = vae.concat_eps_per_scale(noise_list)[0]  # prior is applied at the top scale
                            #####################################################
                            start = timer()
                            # eps1 = diffusion.run_denoising_diffusion(noisee, diffusion_cont, mask, x, args, eps, dae, num_samples, shape, temp, enable_autocast, is_image=False, prior_var=prior_var)
                            temp = 0.3
                            # initialize sample
                            x_noisy = noisee * temp
                            ################################################
                            t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                                diffusion_cont.iw_quantities(args.batch_size, args.time_eps, 'll_iw_denoise',
                                                             args.iw_subvp_like_vp_sde)

                            x_noisy = m_t_q[0] * eps + torch.sqrt(var_t_q[0]) * x_noisy

                            if is_train==False:
                                eps1, nfe, time_ode_solve = diffusion_cont.sample_model_ode(x, mask, dae, num_samples1,
                                                                                        shape, 1e-3, 1e-3,
                                                                                        enable_autocast, temp, x_noisy)
                            else:
                                eps1, nfe, time_ode_solve = diffusion_cont.sample_model_ode(x_gt, mask, dae, num_samples1,
                                                                                        shape, 1e-3, 1e-3,
                                                                                        enable_autocast, temp, x_noisy)
                            decomposed_eps = vae.decompose_eps(eps1)
                            decomposed_eps2 = vae.decompose_eps(eps2)
                            sigma_mask = torch.tensor(False, dtype=torch.bool)
                            image = vae.sample(num_samples1, vae_temp, sigma_mask, decomposed_eps2,  decomposed_eps, enable_autocast)
                            image_ls.append(image)

                            #################################
                            aa = image.median(dim=1).values
                            imputed_samples[:, k, :, :] = aa
                            #######################################

                    result = torch.stack(image_ls, dim=0).mean(dim=0)
                    mae = torch.abs(result - x_gt) * (1 - mask)
                    mae = torch.mean(mae, dim=1)
                    mse = mae ** 2

                    total_mae += torch.sum(mae) / 100
                    total_rmse += torch.sum(mse) / 100


                    total_evaluate_point += torch.sum((1 - mask)) / 100
                    valid_result.append(result)
                    valid_gt.append(x_gt)
                    valid_mask.append(mask)
                    #####################
                    all_target.append(x_gt.squeeze(1))
                    all_evalpoint.append((1 - mask).squeeze(1))
                    all_generated_samples.append(imputed_samples)
                    ##############################

                imputation = total_mae / total_evaluate_point
                print('Imputation mae is ' + str(imputation))

                imputation_rmse = (total_rmse / total_evaluate_point) ** 0.5
                print('Imputation rmse is ' + str(imputation_rmse))

                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                with open(
                         "./test_generated_outputs_nsample" + str(1000) + ".pk", "wb"
                ) as f:
                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                        ],
                        f,
                    )


                CRPS = calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, 0, 1.0
                )
                print(' CRPS is ' + str(CRPS))

            end = timer()
            sampling_time = end - start
            # average over GPUs
            nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
            sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
            time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
            utils.average_tensor(nfe_torch, True)
            utils.average_tensor(sampling_time_torch, True)
            utils.average_tensor(time_ode_solve_torch, True)

    return valid_result, valid_gt, valid_mask, nfe_torch, time_ode_solve_torch, sampling_time_torch, imputation



def generate_samples_vada_reconstruction(args, diffusion_cont, val_queue, dae, diffusion, vae, num_samples, enable_autocast, ode_eps=None, ode_solver_tol=None,
                          ode_sample=False, prior_var=1.0, is_train=False, temp=1.0, vae_temp=1.0, noise=None):
    shape = [dae.num_input_channels, dae.input_size, dae.input_size]

    total_mae = 0
    total_evaluate_point = 0
    total_rmse = 0


    with torch.no_grad():
        if ode_sample:
            assert isinstance(diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
            assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
            assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
            start = timer()
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(dae, num_samples, shape, ode_eps, ode_solver_tol, enable_autocast, temp, noise)

            decomposed_eps = vae.decompose_eps(eps)
            image = vae.sample(num_samples, vae_temp, decomposed_eps, enable_autocast)
            end = timer()
            sampling_time = end - start
            # average over GPUs
            nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
            sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
            time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
            utils.average_tensor(nfe_torch, True)
            utils.average_tensor(sampling_time_torch, True)
            utils.average_tensor(time_ode_solve_torch, True)

        else:
            assert isinstance(diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
            assert noise is None, 'Noise is not used in ancestral sampling.'
            #########################################################################
            vae.eval()
            dae.eval()
            valid_result = []
            valid_gt = []
            valid_mask = []

            all_generated_samples = []
            all_target = []
            all_evalpoint = []

            cc = 0
            with torch.no_grad():
                # for step, x in enumerate(val_queue):
                for step, (x, x_gt, mask, ob_mask) in enumerate(tqdm(val_queue, total=len(val_queue))):
                    # print('Batch number is ' + str(cc))
                    cc = cc + 1
                    num_samples1 = x.shape[0]
                    x = utils.common_x_operations(x, 8)
                    x_gt = utils.common_x_operations(x_gt, 8)
                    mask = utils.common_x_operations(mask, 8)
                    image_ls = []
                    if mask.shape[2]==24:
                        mask[:,:,:,7] = 0

                    B, _, K, L = x.shape
                    imputed_samples = torch.zeros(B, num_samples, K, L).to(x.device)
                    for k in range(num_samples):
                        with autocast(enabled=True):
                            # apply vae:
                            sigma_mask = torch.tensor(False, dtype=torch.bool)
                            if is_train==False:
                                logits, all_log_q, all_eps, all_dist = vae(x, sigma_mask)
                            else:
                                logits, all_log_q, all_eps, all_dist = vae(x_gt, sigma_mask)


                            eps2 = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                            remaining_neg_log_p_total, _ = utils.cross_entropy_normal(
                                all_eps[vae.num_groups_per_scale:])

                            ############################################################################
                            noise_list = [torch.randn_like(tensor, device='cuda') for tensor in all_eps]
                            for ii in range(len(all_dist)):
                                all_eps[ii] = all_dist[ii].sample_diff(0)

                            eps = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                            noisee = vae.concat_eps_per_scale(noise_list)[0]  # prior is applied at the top scale
                            #####################################################
                            start = timer()
                            # eps1 = diffusion.run_denoising_diffusion(noisee, diffusion_cont, mask, x, args, eps, dae, num_samples, shape, temp, enable_autocast, is_image=False, prior_var=prior_var)
                            temp = 0.3
                            # initialize sample
                            x_noisy = noisee * temp
                            ################################################
                            t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                                diffusion_cont.iw_quantities(args.batch_size, args.time_eps, 'll_iw_denoise',
                                                             args.iw_subvp_like_vp_sde)

                            x_noisy = m_t_q[0] * eps + torch.sqrt(var_t_q[0]) * x_noisy

                            if is_train==False:
                                eps1, nfe, time_ode_solve = diffusion_cont.sample_model_ode(x, mask, dae, num_samples1,
                                                                                        shape, 1e-3, 1e-3,
                                                                                        enable_autocast, temp, x_noisy)
                            else:
                                eps1, nfe, time_ode_solve = diffusion_cont.sample_model_ode(x_gt, mask, dae, num_samples1,
                                                                                        shape, 1e-3, 1e-3,
                                                                                        enable_autocast, temp, x_noisy)
                            decomposed_eps = vae.decompose_eps(eps1)
                            decomposed_eps2 = vae.decompose_eps(eps2)
                            sigma_mask = torch.tensor(False, dtype=torch.bool)
                            image = vae.sample(num_samples1, vae_temp, sigma_mask, decomposed_eps2,  decomposed_eps, enable_autocast)
                            image_ls.append(image)

                            #################################
                            aa = image.median(dim=1).values
                            imputed_samples[:, k, :, :] = aa
                            #######################################

                    result = torch.stack(image_ls, dim=0).mean(dim=0)
                    mae = torch.abs(result - x_gt) * (mask)
                    mae = torch.mean(mae, dim=1)
                    mse = mae ** 2

                    total_mae += torch.sum(mae) / 100
                    total_rmse += torch.sum(mse) / 100


                    total_evaluate_point += torch.sum((mask)) / 100
                    valid_result.append(result)
                    valid_gt.append(x_gt)
                    valid_mask.append(mask)
                    #####################
                    all_target.append(x_gt.squeeze(1))
                    all_evalpoint.append((mask).squeeze(1))
                    all_generated_samples.append(imputed_samples)
                    ##############################

                imputation = total_mae / total_evaluate_point
                print('reconstruction Imputation mae is ' + str(imputation))

                imputation_rmse = (total_rmse / total_evaluate_point) ** 0.5
                print('reconstruction Imputation rmse is ' + str(imputation_rmse))

                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)
                CRPS = calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, 0, 1.0
                )
                print('reconstruction CRPS is ' + str(CRPS))

            end = timer()
            sampling_time = end - start
            # average over GPUs
            nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
            sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
            time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
            utils.average_tensor(nfe_torch, True)
            utils.average_tensor(sampling_time_torch, True)
            utils.average_tensor(time_ode_solve_torch, True)

    return valid_result, valid_gt, valid_mask, nfe_torch, time_ode_solve_torch, sampling_time_torch, imputation






def elbo_evaluation(val_queue, diffusion, dae, args, vae=None, max_step=None, ode_eval=False, ode_param_dict=None,
                    num_samples=1, num_inner_samples=1, report_std=False):
    nelbo_avg, neg_log_p_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

    reconst_mae = utils.AvgrageMeter()

    if ode_eval:
        # Note that we are currently not averaging the NFE counter over different GPUs! Doesn't seem very important,
        # though, as NFEs mainly matter  for sampling not NLL calculation.
        nfe_counter_avg = utils.AvgrageMeter()

    if ode_eval and num_inner_samples > 1 and report_std:
        stddev_avg = utils.AvgrageMeter()
        stderror_avg = utils.AvgrageMeter()

    dae.eval()
    vae.eval()
    with torch.no_grad():
        for step, (x, x_gt, mask) in enumerate(val_queue):
        # for step, x in enumerate(val_queue):
            # we avoid computing ELBO on the whole dataset
            if max_step is not None and step >= max_step:
                break

            x = utils.common_x_operations(x, args.num_x_bits)
            x_gt = utils.common_x_operations(x_gt, 8)
            mask = utils.common_x_operations(mask, 8)


            nelbo, log_iw = [], []
            mae_ls = []
            for k in range(num_samples):
                with autocast(enabled=args.autocast_eval):
                    # apply vae:
                    logits, all_log_q, all_eps = vae(x)
                    eps = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                    remaining_neg_log_p_total, _ = utils.cross_entropy_normal(all_eps[vae.num_groups_per_scale:])
                    output = vae.decoder_output(logits)
                    vae_recon_loss, mae = utils.reconstruction_loss(output, x, (1-mask), crop=vae.crop_output)
                    neg_vae_entropy = utils.sum_log_q(all_log_q)
                    vae_loss = vae_recon_loss + neg_vae_entropy + remaining_neg_log_p_total


                # computing prior likelihood outside of autocast as the inner functions have their own autocast
                if ode_eval:
                    assert isinstance(diffusion, DiffusionBase), 'ODE-based NLL evaluation requires cont. diffusion!'
                    assert ode_param_dict is not None
                    nelbo_prior, nfe, stddev_batch, stderror_batch = diffusion.compute_ode_nll(
                        dae=dae, eps=eps, ode_eps=ode_param_dict['ode_eps'],
                        ode_solver_tol=ode_param_dict['ode_solver_tol'], enable_autocast=args.autocast_eval,
                        no_autograd=args.no_autograd_jvp, num_samples=num_inner_samples, report_std=report_std)

                    nfe_counter_avg.update(nfe, x.size(0))
                    if num_inner_samples > 1 and report_std:
                        assert stddev_batch is not None and stderror_batch is not None
                        stddev_avg.update(stddev_batch, x.size(0))
                        stderror_avg.update(stderror_batch, x.size(0))
                else:
                    assert isinstance(diffusion, DiffusionDiscretized), 'Regular NLL evaluation requires disc. diffusion!'
                    assert num_inner_samples == 1, 'inner_samples more than one is not implemented'
                    nelbo_prior = diffusion.compute_nelbo(dae, eps, enable_autocast=args.autocast_eval, is_image=False,
                                                          prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0)

                nelbo_k = nelbo_prior + vae_loss
                nelbo.append(nelbo_k)
                log_iw.append(-nelbo_k)   # we can use nelbo as the KL is computed using a sample based objective.

                mae_ls.append(mae)


            # IW estimation of log prob
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))
            loss_iw = torch.mean(-log_p)
            neg_log_p_avg.update(loss_iw.data, x.size(0))

            # Multi-sample estimation of nelbo
            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            loss_nelbo = torch.mean(nelbo)
            nelbo_avg.update(loss_nelbo.data, x.size(0))


            # Multi-sample estimation of nelbo
            mae_ls = torch.mean(torch.stack(mae_ls, dim=1))
            loss_mae_ls = torch.mean(mae_ls)
            reconst_mae.update(loss_mae_ls.data, x.size(0))




    if ode_eval and num_inner_samples > 1 and report_std:
        utils.average_tensor(stddev_avg.avg, args.distributed)
        utils.average_tensor(stderror_avg.avg, args.distributed)
        stddev_avg_return = stddev_avg.avg
        stderror_avg_return = stderror_avg.avg
    else:
        stddev_avg_return = None
        stderror_avg_return = None

    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    utils.average_tensor(nelbo_avg.avg, args.distributed)

    utils.average_tensor(reconst_mae.avg, args.distributed)


    nfes = nfe_counter_avg.avg if ode_eval else None
    return nelbo_avg.avg, neg_log_p_avg.avg, nfes, stddev_avg_return, stderror_avg_return, reconst_mae.avg


