# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import urllib
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
# from torch._utils import _accumulate
import h5py

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pickle
import torchcde



def _accumulate(iterable):
    total = 0
    result = []
    for x in iterable:
        total += x
        result.append(total)
    return result


class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args.data, args.distributed, args.batch_size)

class CombinedDataset(Dataset):
    def __init__(self, data_dataset, data_gt, mask_dataset, mask_ob, transform=None):
        self.data_dataset = data_dataset
        self.data_gt = data_gt
        self.mask_dataset = mask_dataset
        self.mask_ob = mask_ob
        self.transform = transform

    def __len__(self):
        return len(self.data_dataset)

    def __getitem__(self, idx):
        data_item = self.data_dataset[idx]
        data_gt = self.data_gt[idx]
        mask_item = self.mask_dataset[idx]
        mask_ob = self.mask_ob[idx]


        if self.transform:
            if mask_item.shape[0] != 24:
                mask_item_tensor = torch.from_numpy(mask_item)  # 转 tensor
                data_gt_tensor = torch.from_numpy(data_gt)  # 转 tensor
                itp_data = torch.where(mask_item_tensor == 0, float('nan'), data_gt_tensor).to(torch.float32)
                itp_data = torchcde.linear_interpolation_coeffs(
                    itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                data_item = itp_data.numpy()

            if isinstance(data_item, np.ndarray):
                data_item = Image.fromarray(data_item)

            if isinstance(data_gt, np.ndarray):
                data_gt = Image.fromarray(data_gt)

            if isinstance(mask_item, np.ndarray):
                mask_item = mask_item.astype(np.float32)
                mask_ob = mask_ob.astype(np.float32)

            data_item = self.transform(data_item)
            data_gt_item = self.transform(data_gt)
            mask_item = self.transform(mask_item)
            mask_ob = self.transform(mask_ob)

        return data_item, data_gt_item, mask_item, mask_ob


def _data_transforms(binarize):
    T = [transforms.ToTensor()]
    if binarize:
        T.append(Binarize())

    train_transform = transforms.Compose(T)
    valid_transform = transforms.Compose(T)
    test_transform = transforms.Compose(T)
    return train_transform, valid_transform, test_transform

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

def get_loaders_eval(dataset, root, distributed, batch_size, augment=True, drop_last_train=True, shuffle_train=True,
                     binarize_binary_datasets=False):
    if dataset == 'ETT':
        binarize_binary_datasets = False
        train_transform, valid_transform, test_transform = _data_transforms(binarize_binary_datasets)

        SEED = 1
        rng = np.random.default_rng(SEED)
        missing_pattern = 'block'

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

            eval =1.0 - sample_mask(shape=((3861+959+983)*24, 7), p=0.015, p_noise=0.1, min_seq=6, max_seq=12 * 2, rng=rng)
            eval = eval.reshape(-1,24,7)

            X_ob_mask = eval[0:3861]

            X_train = X_train * X_ob_mask
            X_val_missing_mask  = eval[3861:3861+959]
            X_test_missing_mask =  eval[3861+959:]
            print()

        else:
            file_path = './data/ETTm_datasets_05.h5'
            with h5py.File(file_path, 'r') as f:
                train_group = f['train']
                X_train = train_group['X'][:]
                X_ob_mask = (~np.isnan(X_train)).astype(np.float32)

                X_val = f['val']['X'][:]
                X_val_missing_mask = f['val']['missing_mask'][:]

                X_test = f['test']['X'][:]
                X_test_missing_mask = f['test']['missing_mask'][:]

                X_train = np.nan_to_num(X_train, nan=0.0)

        # combine all the values and make them positive
        observed_values = np.concatenate((X_train, X_val, X_test), axis=0) + 4.130494

        observed_masks = np.concatenate(
            (np.ones_like(X_ob_mask), np.ones_like(X_val_missing_mask), np.ones_like(X_test_missing_mask)), axis=0)
        gt_masks = np.concatenate((X_ob_mask, X_val_missing_mask, X_test_missing_mask), axis=0)

        ########## 1D interpolation for missing #############
        whole_data = observed_values * gt_masks
        whole_data = whole_data.reshape(whole_data.shape[0] * whole_data.shape[1], -1)

        whole_mask = gt_masks
        whole_mask = 1 - whole_mask.reshape(whole_mask.shape[0] * whole_mask.shape[1], -1)

        y_hat = []
        total_mask = whole_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(whole_data[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq1 = np.transpose(np.array(y_hat))

        observed_data_interpolation = data_seq1.reshape(observed_values.shape[0], observed_values.shape[1], -1)
        ################################################################################
        np.savez('raw_data_ETT.npz', observed_data_interpolation=observed_data_interpolation,
                 observed_values=observed_values,
                 gt_masks=gt_masks, observed_masks=observed_masks)

        #################### split the dataset as SAITS model ###########################################################################
        indlist = np.arange(observed_data_interpolation.shape[0])

        num_train = 3861
        num_val = 3861 + 959
        train_index = indlist[:num_train]
        valid_index = indlist[num_train:num_val]
        test_index = indlist[num_val:]

        train_data_raw_gt = observed_values[train_index]
        train_data_raw = observed_data_interpolation[train_index]
        train_data_raw_gt_mask = gt_masks[train_index]
        train_data_raw_ob_mask = observed_masks[train_index]

        last_feature = train_data_raw[:, :, [-1]]
        train_data_raw = np.concatenate([train_data_raw, last_feature], axis=-1)


        last_feature = train_data_raw_gt[:, :, [-1]]
        train_data_raw_gt = np.concatenate([train_data_raw_gt, last_feature], axis=-1)


        train_data_raw_gt_mask = np.pad(train_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=1).astype(np.float32)
        train_data_raw_ob_mask = np.pad(train_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=0).astype(np.float32)



        np.savez('train_orignial_ETT.npz', train_data_raw=train_data_raw, train_data_raw_gt=train_data_raw_gt,
                 train_data_raw_gt_mask=train_data_raw_gt_mask, train_data_raw_ob_mask=train_data_raw_ob_mask)

        train_combined_dataset = CombinedDataset(train_data_raw, train_data_raw_gt, train_data_raw_gt_mask,
                                                 train_data_raw_ob_mask,
                                                 transform=train_transform)
        #############################################################################
        valid_data_gt = observed_values[valid_index]
        valid_data_raw = observed_data_interpolation[valid_index]
        valid_data_raw_gt_mask = gt_masks[valid_index]
        valid_data_raw_ob_mask = observed_masks[valid_index]

        last_feature = valid_data_raw[:, :, [-1]]
        valid_data_raw = np.concatenate([valid_data_raw, last_feature], axis=-1)

        last_feature = valid_data_gt[:, :, [-1]]
        valid_data_gt = np.concatenate([valid_data_gt, last_feature], axis=-1)

        valid_data_raw_gt_mask = np.pad(valid_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=1).astype(np.float32)
        valid_data_raw_ob_mask = np.pad(valid_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=0).astype(np.float32)

        np.savez('valid_orignial_ETT.npz', valid_data_raw=valid_data_raw, valid_data_gt=valid_data_gt,
                 valid_data_raw_gt_mask=valid_data_raw_gt_mask, valid_data_raw_ob_mask=valid_data_raw_ob_mask)

        valid_combined_dataset = CombinedDataset(valid_data_raw, valid_data_gt, valid_data_raw_gt_mask,
                                                 valid_data_raw_ob_mask,
                                                 transform=valid_transform)

        #########################################################
        test_data_gt = observed_values[test_index]
        test_data_raw = observed_data_interpolation[test_index]
        test_data_raw_gt_mask = gt_masks[test_index]
        test_data_raw_ob_mask = observed_masks[test_index]

        last_feature = test_data_raw[:, :, [-1]]
        test_data_raw = np.concatenate([test_data_raw, last_feature], axis=-1)

        last_feature = test_data_gt[:, :, [-1]]
        test_data_gt = np.concatenate([test_data_gt, last_feature], axis=-1)


        test_data_raw_gt_mask = np.pad(test_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                       constant_values=1).astype(np.float32)
        test_data_raw_ob_mask = np.pad(test_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                       constant_values=0).astype(np.float32)

        np.savez('test_orignial_ETT.npz', test_data_raw=test_data_raw, test_data_gt=test_data_gt,
                 test_data_raw_ob_mask=test_data_raw_ob_mask,
                 test_data_raw_gt_mask=test_data_raw_gt_mask)

        test_combined_dataset = CombinedDataset(test_data_raw, test_data_gt, test_data_raw_gt_mask,
                                                test_data_raw_ob_mask,
                                                transform=test_transform)

    elif dataset == 'Synthetic':
        eval_length = 10
        val_len = 0.1
        test_len = 0.2
        missing_rate = 0.5
        T = 500
        train_transform, valid_transform, test_transform = _data_transforms(binarize_binary_datasets)
        U = np.array([
            [1.0, 1.0, -2.0, -2.0],
            [0.4, 1.0, 2.0, -1.0],
            [-0.3, 2.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0, 0.5]
        ])

        np.random.seed(0)
        time = np.sort(np.random.rand(T))

        V = np.vstack([
            10 * time,
            np.sin(20 * np.pi * time),
            np.cos(40 * np.pi * time),
            np.sin(60 * np.pi * time)
        ])

        X = U @ V

        # generating mask
        mask = np.zeros_like(X, dtype=int)
        num_points = X.size
        observed_indices = np.random.choice(num_points, int(num_points * missing_rate), replace=False)
        np.put(mask, observed_indices, 1)


        mask = mask.T
        val_start = int((1 - val_len - test_len) * T)
        test_start = int((1 - test_len) * T)

####################################################
        X_noisy = X

        min = 1
        max = 10

        X_noisy = (X_noisy.T ) / max + min
        X = (X.T ) / max + min

        train_data_raw = X_noisy[0:val_start]
        train_data_raw_gt = X[0:val_start]
        train_data_raw_gt_mask = mask[0:val_start]
        train_data_raw_ob_mask = np.ones_like(mask)[0:val_start]

        train_data_raw = train_data_raw.reshape(-1, eval_length, 4)

        train_data_raw_gt = train_data_raw_gt.reshape(-1, eval_length, 4)

        train_data_raw_gt_mask = train_data_raw_gt_mask.reshape(-1, eval_length, 4)

        train_data_raw_ob_mask = train_data_raw_ob_mask.reshape(-1, eval_length, 4)

        ########################################################################################
        np.savez('train_orignial_synthetic.npz', train_data_raw=train_data_raw, train_data_raw_gt=train_data_raw_gt,
                 train_data_raw_gt_mask=train_data_raw_gt_mask, train_data_raw_ob_mask=train_data_raw_ob_mask)

        train_combined_dataset = CombinedDataset(train_data_raw, train_data_raw_gt, train_data_raw_gt_mask, train_data_raw_ob_mask,
                                                 transform=train_transform)
        #############################################################################
        # valid dataset
        valid_data_raw = X_noisy[val_start:test_start]
        valid_data_gt = X[val_start:test_start]
        valid_data_raw_gt_mask = mask[val_start:test_start]
        valid_data_raw_ob_mask = mask[val_start:test_start]

        valid_data_raw = valid_data_raw.reshape(-1, eval_length, 4)

        valid_data_gt = valid_data_gt.reshape(-1, eval_length, 4)


        valid_data_raw_gt_mask = valid_data_raw_gt_mask.reshape(-1, eval_length, 4)

        valid_data_raw_ob_mask = valid_data_raw_ob_mask.reshape(-1, eval_length, 4)

        np.savez('valid_orignial_synthetic.npz', valid_data_raw=valid_data_raw, valid_data_gt=valid_data_gt,
                 valid_data_raw_gt_mask=valid_data_raw_gt_mask, valid_data_raw_ob_mask=valid_data_raw_ob_mask)

        valid_combined_dataset = CombinedDataset(valid_data_raw, valid_data_gt, valid_data_raw_gt_mask, valid_data_raw_ob_mask,
                                                 transform=valid_transform)

        # test dataset
        test_data_raw = X_noisy[test_start:]
        test_data_gt = X[test_start:]
        test_data_raw_gt_mask = mask[test_start:]
        test_data_raw_ob_mask = mask[test_start:]

        test_data_raw = test_data_raw.reshape(-1, eval_length, 4)


        test_data_gt = test_data_gt.reshape(-1, eval_length, 4)

        test_data_raw_gt_mask = test_data_raw_gt_mask.reshape(-1, eval_length, 4)

        test_data_raw_ob_mask = test_data_raw_ob_mask.reshape(-1, eval_length, 4)

        np.savez('test_orignial_synthetic.npz', test_data_raw=test_data_raw, test_data_gt=test_data_gt,
                 test_data_raw_gt_mask=test_data_raw_gt_mask,test_data_raw_ob_mask=test_data_raw_ob_mask)

        #########################################################
        test_combined_dataset = CombinedDataset(test_data_raw, test_data_gt, test_data_raw_gt_mask,test_data_raw_ob_mask,
                                                 transform=test_transform)


    elif dataset == 'P2012':
        train_transform, valid_transform, test_transform = _data_transforms(binarize_binary_datasets)

        path = "./data/physio_missing0.5_seed1_std.pk"
        with open(path, "rb") as f:
            observed_values, observed_masks, gt_masks, range_, mean, std = pickle.load(f)

        whole_data = observed_values
        whole_data = whole_data.reshape(whole_data.shape[0] * whole_data.shape[1], -1)

        whole_mask = observed_masks
        whole_mask = 1 - whole_mask.reshape(whole_mask.shape[0] * whole_mask.shape[1], -1)

        y_hat = []
        total_mask = whole_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(whole_data[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq1 = np.transpose(np.array(y_hat))

        observed_values = data_seq1.reshape(observed_values.shape[0], observed_values.shape[1], -1)

        ###################################################################################
        observed_data_interpolation = observed_values * gt_masks

        np.savez('raw_data_P2012.npz', observed_data_interpolation=observed_data_interpolation,
                 observed_values=observed_values,
                 gt_masks=gt_masks, observed_masks=observed_masks)

        observed_values = np.load('raw_data_P2012.npz')['observed_values']
        observed_data_interpolation = np.load('raw_data_P2012.npz')['observed_data_interpolation']
        gt_masks = np.load('raw_data_P2012.npz')['gt_masks']
        observed_masks = np.load('raw_data_P2012.npz')['observed_masks']

        ############### split dataset as CSDI ################################################################################
        indlist = np.arange(observed_data_interpolation.shape[0])
        seed = 1
        np.random.seed(seed)
        np.random.shuffle(indlist)

        # 5-fold test
        start = 0
        end = (int)(0.2 * observed_data_interpolation.shape[0])
        test_index = indlist[start:end]
        remain_index = np.delete(indlist, np.arange(start, end))

        np.random.seed(seed)
        np.random.shuffle(remain_index)
        num_train = (int)(observed_data_interpolation.shape[0] * 0.7)
        train_index = remain_index[:num_train]
        valid_index = remain_index[num_train:]

        #########################################################
        train_data_raw_gt = observed_values[train_index]
        train_data_raw = observed_data_interpolation[train_index]
        train_data_raw_gt_mask = gt_masks[train_index]
        train_data_raw_ob_mask = observed_masks[train_index]

        train_data_raw = np.pad(train_data_raw, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        train_data_raw_gt = np.pad(train_data_raw_gt, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        train_data_raw_gt_mask = np.pad(train_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=1)
        train_data_raw_ob_mask = np.pad(train_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=0)

        np.savez('train_orignial_P2012.npz', train_data_raw=train_data_raw, train_data_raw_gt=train_data_raw_gt,
                 train_data_raw_gt_mask=train_data_raw_gt_mask, train_data_raw_ob_mask=train_data_raw_ob_mask)

        train_combined_dataset = CombinedDataset(train_data_raw, train_data_raw_gt, train_data_raw_gt_mask,
                                                 train_data_raw_ob_mask,
                                                 transform=train_transform)
        #############################################################################
        valid_data_gt = observed_values[valid_index]
        valid_data_raw = observed_data_interpolation[valid_index]
        valid_data_raw_gt_mask = gt_masks[valid_index]
        valid_data_raw_ob_mask = observed_masks[valid_index]

        valid_data_raw = np.pad(valid_data_raw, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        valid_data_gt = np.pad(valid_data_gt, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        valid_data_raw_gt_mask = np.pad(valid_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=1)
        valid_data_raw_ob_mask = np.pad(valid_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                        constant_values=0)

        np.savez('valid_orignial_P2012.npz', valid_data_raw=valid_data_raw, valid_data_gt=valid_data_gt,
                 valid_data_raw_gt_mask=valid_data_raw_gt_mask, valid_data_raw_ob_mask=valid_data_raw_ob_mask)

        valid_combined_dataset = CombinedDataset(valid_data_raw, valid_data_gt, valid_data_raw_gt_mask,
                                                 valid_data_raw_ob_mask,
                                                 transform=valid_transform)

        #########################################################
        test_data_gt = observed_values[test_index]
        test_data_raw = observed_data_interpolation[test_index]
        test_data_raw_gt_mask = gt_masks[test_index]
        test_data_raw_ob_mask = observed_masks[test_index]

        test_data_raw = np.pad(test_data_raw, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        test_data_gt = np.pad(test_data_gt, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        test_data_raw_gt_mask = np.pad(test_data_raw_gt_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                       constant_values=1)
        test_data_raw_ob_mask = np.pad(test_data_raw_ob_mask, ((0, 0), (0, 0), (0, 1)), mode='constant',
                                       constant_values=0)

        np.savez('test_orignial_P2012.npz', test_data_raw=test_data_raw, test_data_gt=test_data_gt,
                 test_data_raw_ob_mask=test_data_raw_ob_mask,
                 test_data_raw_gt_mask=test_data_raw_gt_mask)

        test_combined_dataset = CombinedDataset(test_data_raw, test_data_gt, test_data_raw_gt_mask,
                                                test_data_raw_ob_mask,
                                                transform=test_transform)
        #########################################################
    elif dataset == 'MIMIC':
        train_transform, valid_transform, test_transform = _data_transforms(binarize_binary_datasets)
        missing_ratio = 0.5
        seed = 1
        np.random.seed(seed)  # seed for ground truth choice
        ########################################################################
        train_data = np.load('./data/MIMIC/processed_train_data.npz')['data']
        test_data = np.load('./data/MIMIC/processed_test_data.npz')['data']
        observed_values = np.concatenate([train_data, test_data], axis=0)

        ########## generate the mask ##################
        observed_masks = ~np.isnan(observed_values)
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False

        gt_masks = masks.reshape(observed_masks.shape).astype("float32")
        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")



        count = np.sum(observed_values < 0)
        print('negative count:' + str(count))

        # set negative anomaly points as 0
        observed_values[observed_values < 0] = 0


        # std normalization
        tmp_values = observed_values.reshape(-1, 8)
        tmp_masks = observed_masks.reshape(-1, 8)
        mean = np.zeros(8)
        std = np.zeros(8)
        for k in range(8):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        observed_values = ( (observed_values) / std * observed_masks)

        # set extraordinary big values limitation as 5
        count = np.sum(observed_values > 7)
        print('extraordinary big values number:' + str(count))

        observed_values[observed_values > 7] = 5
        np.savez('MIMIC4_mean_std.npz', mean=mean, std=std)

        gt_masks = observed_masks - gt_masks
        gt_masks = 1 - gt_masks

        whole_data = observed_values
        whole_data = whole_data.reshape(whole_data.shape[0] * whole_data.shape[1], -1)
        whole_mask = observed_masks
        whole_mask = 1 - whole_mask.reshape(whole_mask.shape[0] * whole_mask.shape[1], -1)

        y_hat = []
        total_mask = whole_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(whole_data[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        data_seq1 = np.transpose(np.array(y_hat))

        observed_values = data_seq1.reshape(observed_values.shape[0], observed_values.shape[1], -1)

        ###################################################################################
        observed_data_interpolation = observed_values * gt_masks

        np.savez('raw_data_MIMIC.npz', observed_data_interpolation=observed_data_interpolation,
                 observed_values=observed_values,
                 gt_masks=gt_masks, observed_masks=observed_masks)

        observed_values = np.load('raw_data_MIMIC.npz')['observed_values']
        observed_data_interpolation = np.load('raw_data_MIMIC.npz')['observed_data_interpolation']
        gt_masks = np.load('raw_data_MIMIC.npz')['gt_masks']
        observed_masks = np.load('raw_data_MIMIC.npz')['observed_masks']
        ##################### split dataset as medfusion code ##########################################################################
        indlist = np.arange(observed_values.shape[0])

        # 5-fold test
        start = 24903
        end = 31129
        train_index = indlist[0:start]
        valid_index = indlist[start:end]
        test_index = indlist[end:]

        #########################################################
        train_data_raw_gt = observed_values[train_index]
        train_data_raw = observed_data_interpolation[train_index]
        train_data_raw_gt_mask = gt_masks[train_index]
        train_data_raw_ob_mask = observed_masks[train_index]


        np.savez('train_orignial_MIMIC.npz', train_data_raw=train_data_raw, train_data_raw_gt=train_data_raw_gt,
                 train_data_raw_gt_mask=train_data_raw_gt_mask, train_data_raw_ob_mask=train_data_raw_ob_mask)

        train_combined_dataset = CombinedDataset(train_data_raw, train_data_raw_gt, train_data_raw_gt_mask,
                                                 train_data_raw_ob_mask,
                                                 transform=train_transform)
        #############################################################################
        valid_data_gt = observed_values[valid_index]
        valid_data_raw = observed_data_interpolation[valid_index]
        valid_data_raw_gt_mask = gt_masks[valid_index]
        valid_data_raw_ob_mask = observed_masks[valid_index]


        np.savez('valid_orignial_MIMIC.npz', valid_data_raw=valid_data_raw, valid_data_gt=valid_data_gt,
                 valid_data_raw_gt_mask=valid_data_raw_gt_mask, valid_data_raw_ob_mask=valid_data_raw_ob_mask)

        valid_combined_dataset = CombinedDataset(valid_data_raw, valid_data_gt, valid_data_raw_gt_mask,
                                                 valid_data_raw_ob_mask,
                                                 transform=valid_transform)

        #########################################################
        test_data_gt = observed_values[test_index]
        test_data_raw = observed_data_interpolation[test_index]
        test_data_raw_gt_mask = gt_masks[test_index]
        test_data_raw_ob_mask = observed_masks[test_index]


        np.savez('test_orignial_MIMIC.npz', test_data_raw=test_data_raw, test_data_gt=test_data_gt,
                 test_data_raw_ob_mask=test_data_raw_ob_mask,
                 test_data_raw_gt_mask=test_data_raw_gt_mask)

        test_combined_dataset = CombinedDataset(test_data_raw, test_data_gt, test_data_raw_gt_mask,
                                                test_data_raw_ob_mask,
                                                transform=test_transform)
        #########################################################


    train_sampler, valid_sampler, test_sampler = None, None, None

    train_queue = torch.utils.data.DataLoader(
        train_combined_dataset, batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle_train,
        sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=drop_last_train)

    valid_queue = torch.utils.data.DataLoader(
        valid_combined_dataset, batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    test_queue = torch.utils.data.DataLoader(
        test_combined_dataset, batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler, pin_memory=True, num_workers=1, drop_last=False)

    train_queue_no_shuffle = torch.utils.data.DataLoader(
        train_combined_dataset, batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, test_queue, train_queue_no_shuffle


def random_split_dataset(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)

    indices = torch.randperm(sum(lengths), generator=g)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]





