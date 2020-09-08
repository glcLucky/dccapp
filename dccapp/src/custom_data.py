import os.path as osp
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import scipy.io as sio
import numpy as np
import h5py

class DCCPT_data(data.Dataset):
    """Custom dataset loader for Pretraining SDAE"""

    def __init__(self, root, filename, h5=False):
        self.root_dir = root

        if h5:
            data = h5py.File(osp.join(root, '{}.h5'.format(filename)), 'r')
        else:
            data = sio.loadmat(osp.join(root, '{}.mat'.format(filename)), mat_dtype=True)
        self.train_data = data['X'][:].astype(np.float32)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        data = self.train_data[item]

        return data

class DCCFT_data(data.Dataset):
    """ Custom dataset loader for the finetuning stage of DCC"""

    def __init__(self, pairs, data, samp_weights):
        self._pairs = pairs
        self._data = data
        self._sampweight = samp_weights

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, inds):
        current_pairs = self._pairs[inds].copy()
        data_ind, mapping = np.unique(current_pairs[:, :2].astype(int), return_inverse=True)
        current_pairs[:, :2] = mapping.reshape((len(inds), 2))

        # preparing weights for pairs as well as sample weight for weighted reconstruction error
        weights_blob = current_pairs[:, 2].astype(np.float32)

        sampweight_blob = np.zeros(len(data_ind), dtype=np.float32)
        sampweight_blob[...] = np.bincount(mapping) / self._sampweight[data_ind]

        # preparing data
        if len(self._data.shape) > 2:
            data_blob = np.zeros((len(data_ind), self._data.shape[1], self._data.shape[2], self._data.shape[3]), dtype=np.float32)
        else:
            data_blob = np.zeros((len(data_ind), self._data.shape[1]), dtype=np.float32)
        data_blob[...] = self._data[data_ind]

        assert np.isfinite(data_blob).all(), 'Nan found in data'
        assert np.isfinite(weights_blob).all(), 'Nan found in weights'
        assert np.isfinite(sampweight_blob).all(), 'Nan found in sample weights'

        return data_blob, weights_blob, sampweight_blob, current_pairs[:, :2].astype(int), data_ind


class DCCSampler(object):
    """Custom Sampler is required. This sampler prepares batch by passing list of
    data indices instead of running over individual index as in pytorch sampler"""
    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size