from __future__ import print_function
import os
import h5py
import numpy as np
import argparse
import scipy.io as sio
from config import get_data_dir

# python 3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Note that just like in RCC & RCC-DR, the graph is built on original data.
# Once the features are extracted from the pretrained SDAE,
# they are merged along with the mkNN graph data into a single file using this module.
parser = argparse.ArgumentParser(
    description='This module is used to merge graph and extracted features into single file')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--version', '-v', type=str, default='default', help='the version of pretrain config')
parser.add_argument('--graph', dest='g', help='path to the graph file', default=None, type=str)
parser.add_argument('--features', dest='feat', help='path to the feature file', default=None, type=str)
parser.add_argument('--out', dest='out', help='path to the output file', default=None, type=str)
parser.add_argument('--h5', dest='h5', action='store_true', help='to store as h5py file')


def main(args):
    datadir = get_data_dir(args.db)

    featurefile = os.path.join(datadir, args.version, args.feat)
    graphfile = os.path.join(datadir, args.version, args.g)
    outputfile = os.path.join(datadir, args.version, args.out)
    if os.path.isfile(featurefile) and os.path.isfile(graphfile):

        if args.h5:
            data0 = h5py.File(featurefile, 'r')
            data1 = h5py.File(graphfile, 'r')
            data2 = h5py.File(outputfile + '.h5', 'w')
        else:
            fo = open(featurefile, 'rb')
            data0 = pickle.load(fo)
            data1 = sio.loadmat(graphfile)
            fo.close()

        joined_data = {'X': data0['data'][:].astype(np.float32),
                       'Z': data0['Z'][:].astype(np.float32),
                       'w': data1['w'][:].astype(np.float32)}

        if args.h5:
            data2.create_dataset('X', data=data0['data'][:].astype(np.float32))
            data2.create_dataset('Z', data=data0['Z'][:].astype(np.float32))
            data2.create_dataset('w', data=data1['w'][:].astype(np.float32))
            data0.close()
            data1.close()
            data2.close()
        else:
            sio.savemat(outputfile + '.mat', joined_data)
        return joined_data
    else:
        if not os.path.isfile(featurefile):
            print("{}: NOT FOUND!".format(featurefile))
        if not os.path.isfile(graphfile):
            print("{}: NOT FOUND!".format(graphfile))        
        raise FileNotFoundError


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
