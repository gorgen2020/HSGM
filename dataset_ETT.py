import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import h5py
from scipy.interpolate import interp1d



def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


class ETT_Dataset(Dataset):
    def __init__(self, eval_length=24, use_index_list=None, missing_ratio=0.5, seed=0, missing_pattern='block', use_latent_diffusion_imputation=True):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        # file_path = 'ETTm_datasets_05.h5'
        # # 0 is CSDI, 1 is latent impute, 2 is ground truth mode

        SEED = 1
        rng = np.random.default_rng(SEED)

        if missing_pattern == 'block':
            file_path = './data/ETTm1_seqlen24_05masked_with_ground_truth.h5'
            with h5py.File(file_path, 'r') as f:
                train_group = f['train']
                X_train = train_group['X'][:]
                X_ob_mask = (~np.isnan(X_train)).astype(np.float32)

                X_val = f['val']['X'][:]
                # X_val_missing_mask = f['val']['missing_mask'][:]

                X_test = f['test']['X'][:]
                # X_test_missing_mask = f['test']['missing_mask'][:]

            eval = 1.0 - sample_mask(shape=((3861 + 959 + 983) * 24, 7), p=0.015, p_noise=0.1, min_seq=6,
                                     max_seq=12 * 2, rng=rng)
            eval = eval.reshape(-1, 24, 7)

            X_ob_mask = eval[0:3861]

            X_train = X_train * X_ob_mask
            X_val_missing_mask = eval[3861:3861 + 959]
            X_test_missing_mask = eval[3861 + 959:]


            if use_latent_diffusion_imputation:
                total_result = np.load('latent_output_' + 'ETT' + '.npz')['total_result']
                total_gt = np.load('latent_output_' + 'ETT' + '.npz')['total_gt']
                X_train = np.where(X_ob_mask == 0, total_result[0:3861], X_train)
                X_ob_mask = np.ones_like(X_ob_mask)

        elif missing_pattern == 'point':
            file_path = './data/ETTm_datasets_05.h5'
            with h5py.File(file_path, 'r') as f:
                train_group = f['train']

                X_train = train_group['X'][:]
                X_ob_mask = (~np.isnan(X_train)).astype(np.float32)

                # 读取 val
                X_val = f['val']['X'][:]
                X_val_hat = f['val']['X_hat'][:]
                X_val_missing_mask = f['val']['missing_mask'][:]
                # X_val_indicating_mask = f['val']['indicating_mask'][:]

                # 读取 test
                X_test = f['test']['X'][:]
                X_test_hat = f['test']['X_hat'][:]
                X_test_missing_mask = f['test']['missing_mask'][:]
                # X_test_indicating_mask = f['test']['indicating_mask'][:]

            if use_latent_diffusion_imputation:
                total_result = np.load('latent_output_' + 'ETT' + '.npz')['total_result']
                total_gt = np.load('latent_output_' + 'ETT' + '.npz')['total_gt']
                X_train = np.where(X_ob_mask == 0, total_result[0:3861], X_train)
                X_ob_mask = np.ones_like(X_ob_mask)

        #########################################################################
        self.observed_values = np.concatenate((X_train, X_val, X_test), axis=0)
        self.observed_masks = np.concatenate(
            (X_ob_mask, np.ones_like(X_val_missing_mask), np.ones_like(X_test_missing_mask)), axis=0)
        self.gt_masks = np.concatenate((X_ob_mask, X_val_missing_mask, X_test_missing_mask), axis=0)
#########################################################################
        print()

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.5, missing_pattern='block', use_latent_diffusion_imputation=True):

    # only to obtain total length of dataset
    dataset = ETT_Dataset(missing_ratio=missing_ratio, seed=seed, missing_pattern=missing_pattern, use_latent_diffusion_imputation=use_latent_diffusion_imputation)
    indlist = np.arange(len(dataset))

    num_train = 3861
    num_val = 3861 + 959
    train_index = indlist[:num_train]
    valid_index = indlist[num_train:num_val]
    test_index = indlist[num_val:]


    dataset = ETT_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, missing_pattern=missing_pattern, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = ETT_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, missing_pattern=missing_pattern, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = ETT_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, missing_pattern=missing_pattern, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
