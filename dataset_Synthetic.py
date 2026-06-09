import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Synthetic_Dataset(Dataset):
    def __init__(self, eval_length=10, mode="train", val_len=0.1, test_len=0.2, missing_pattern='point',
                 target_strategy='random', missing_ratio=0.5, use_latent_diffusion_imputation=True):


        self.eval_length = eval_length
        self.target_strategy = target_strategy
        self.mode = mode
        self.train_mean = 0
        self.train_std = 1
        T = 500

        train_set = np.load('./data/Synthetic/train_orignial_synthetic.npz')['train_data_raw_gt']
        train_set_mask = np.load('./data/Synthetic/train_orignial_synthetic.npz')['train_data_raw_gt_mask']

        valid_set = np.load('./data/Synthetic/valid_orignial_synthetic.npz')['valid_data_gt']
        valid_set_mask = np.load('./data/Synthetic/valid_orignial_synthetic.npz')['valid_data_raw_gt_mask']

        test_set = np.load('./data/Synthetic/test_orignial_synthetic.npz')['test_data_gt']
        test_set_mask = np.load('./data/Synthetic/test_orignial_synthetic.npz')['test_data_raw_gt_mask']


        X = np.concatenate((train_set, valid_set, test_set), axis=0)
        X_noisy = X.reshape(-1,4)
        mask = np.concatenate((train_set_mask, valid_set_mask, test_set_mask), axis=0)
        mask = mask.reshape(-1, 4)
        # split the dataset
        val_start = int((1 - val_len - test_len) * T)
        test_start = int((1 - test_len) * T)

        if use_latent_diffusion_imputation:
            total_result = np.load('latent_output_' + 'Synthetic' + '.npz')['total_result']
            # total_gt = np.load('latent_output_' + 'Synthetic' + '.npz')['gt']
            # MMask = np.load('latent_output_' + 'Synthetic' + '.npz')['gt_mask']

            X_interpolation = np.where(mask == 0, total_result, X_noisy)
            ob_mask = np.ones_like(mask)
        else:
            ob_mask = mask
            X_interpolation = mask * X_noisy

        gt_mask = mask



        # create data for batch
        self.use_index = []
        self.cut_length = []

        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = X_interpolation[:val_start]

        elif mode == 'valid':
            self.observed_mask = np.ones_like(ob_mask)[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = X_noisy[val_start: test_start]

        elif mode == 'test':
            self.observed_mask = np.ones_like(ob_mask)[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = X_noisy[test_start:]

        current_length = len(self.observed_mask) - eval_length + 1

        if mode == "test":
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(
                0, 0 + eval_length * n_sample, eval_length
            )
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [eval_length - len(self.observed_data) % eval_length]
        elif mode != "test":
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]
        ob_mask = self.observed_mask[index: index + self.eval_length]
        ob_mask_t = torch.tensor(ob_mask).float()
        gt_mask = self.gt_mask[index: index + self.eval_length]


        s = {
            "observed_data": ob_data,
            "observed_mask": ob_mask,
            "gt_mask": gt_mask,
            "hist_mask": ob_mask,
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

        return s


    def __len__(self):
        return len(self.use_index)



def get_dataloader(batch_size, device, val_len=0.1, test_len=0.2, missing_pattern='block'
                  , num_workers=4, target_strategy='random', missing_ratio=0.5, use_latent_diffusion_imputation=True):
    dataset = Synthetic_Dataset(mode="train", val_len=val_len, test_len=test_len, missing_pattern=missing_pattern,
                             target_strategy=target_strategy, missing_ratio=missing_ratio, use_latent_diffusion_imputation=use_latent_diffusion_imputation)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dataset_test = Synthetic_Dataset(mode="test", val_len=val_len, test_len=test_len, missing_pattern=missing_pattern,
                                   target_strategy=target_strategy, missing_ratio=missing_ratio, use_latent_diffusion_imputation=use_latent_diffusion_imputation)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dataset_valid = Synthetic_Dataset(mode="valid", val_len=val_len, test_len=test_len, missing_pattern=missing_pattern,
                                   target_strategy=target_strategy, missing_ratio=missing_ratio, use_latent_diffusion_imputation=use_latent_diffusion_imputation)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    scaler = torch.from_numpy(np.array(1)).to(device).float()
    mean_scaler = torch.from_numpy(np.array(0)).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
