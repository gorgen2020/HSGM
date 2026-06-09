import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d
import pickle

class MIMIC_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.5, seed=0, use_latent_diffusion_imputation=True):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        ########################################################################
        train_data = np.load('./data/MIMIC/processed_train_data.npz')['data']
        test_data = np.load('./data/MIMIC/processed_test_data.npz')['data']
        self.observed_values = np.concatenate([train_data,test_data], axis=0)

        ########## generate the mask ##################
        self.observed_masks = ~np.isnan(self.observed_values)

        masks = self.observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False

        self.gt_masks = masks.reshape(self.observed_masks.shape).astype("float32")
        self.observed_values = np.nan_to_num(self.observed_values)
        self.observed_masks = self.observed_masks.astype("float32")





        count = np.sum(self.observed_values < 0)
        print('negative count:' + str(count))

        # set negative anomaly points as 0
        self.observed_values[self.observed_values < 0] = 0


        # std normalization
        tmp_values = self.observed_values.reshape(-1, 8)
        tmp_masks = self.observed_masks.reshape(-1, 8)
        mean = np.zeros(8)
        std = np.zeros(8)
        for k in range(8):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        self.observed_values = ( (self.observed_values) / std * self.observed_masks)

        # set extraordinary big values limitation as 5
        count = np.sum(self.observed_values > 7)
        print('extraordinary big values number:' + str(count))

        self.observed_values[self.observed_values > 7] = 5

        self.observed_values = (
                (self.observed_values - mean / std) * self.observed_masks
        )



        if use_latent_diffusion_imputation:
            aaa = np.load('latent_output_' + 'MIMIC' + '.npz')
            latent_imputation = np.load('latent_output_' + 'MIMIC' + '.npz')['result']
            latent_gt = np.load('latent_output_' + 'MIMIC' + '.npz')['total_gt']

            latent_imputation = (latent_imputation - mean) / std
            latent_gt = (latent_gt - mean) / std

            original_mask = 1 - self.observed_masks
            self.observed_masks = self.observed_masks + original_mask
            self.gt_masks = self.gt_masks + original_mask
            self.observed_values = np.where(original_mask == 1, latent_imputation, self.observed_values)

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


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.5, use_latent_diffusion_imputation=True):

    # only to obtain total length of dataset
    dataset = MIMIC_Dataset(missing_ratio=missing_ratio, seed=seed, use_latent_diffusion_imputation=use_latent_diffusion_imputation)

    indlist = np.arange(len(dataset))

    # 5-fold test
    start = 24903
    end = 31129
    train_index = indlist[0:start]

    valid_index = indlist[start:end]
    test_index = indlist[end:]

    dataset = MIMIC_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = MIMIC_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = MIMIC_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, use_latent_diffusion_imputation=use_latent_diffusion_imputation
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
