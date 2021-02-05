import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from matplotlib import cm
import seaborn as sns
sns.set(font_scale=3, rc={'text.usetex' : False})
sns.set_theme()
sns.set_style('whitegrid')
import glob
import shutil
import math
import copy

import models
import random

import torch.optim
import torch
import argparse
import utils

from sklearn.linear_model import LogisticRegression

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-o', type=str, help='output root for the results')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--lr_update', type=int, default=0, help='update for the learning rate in epochs')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('-R', '--remove', type=int, default=100, help='the number of neurons to remove at each layer')
    # parser_model = parser.add_mutually_exclusive_group(required=True)
    # parser_model.add_argument('--model', help='path of the model to separate')
    # parser_model.add_argument('--directory', help='path of the directory where results are')
    # parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    # parser_model.add_argument('--csv', help='path of the previous saved csv file')
    # parser_model.add_argument('--width', type=int, help='width for a random network')
    parser.add_argument('--reset_random', action='store_true', help='randomly reset the model')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth', type=int, help='the depth for init network')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--steps', type=int, default=20, help='The number of steps to take')
   # parser.add_argument('--log_mult', type=int, default=1, help='The log2 of the mulltiplicative factor')
    parser.add_argument('--infile', '-A',  nargs='*', help='the directories to process')
    parser.add_argument('--outfile', '-B', nargs='*', help='the directories to process')
    parser.add_argument('--root', help='the root of the experiment to operate on')
    parser.add_argument('--table_format', choices=["wide", "long"], default="long")
    parser.add_argument('--rename', help='rename the directory')
    parser.add_argument('--name', help=' name for the directory')
    parser.add_argument('dirs', nargs='*', help='list of directories to process')

    parser.add_argument('--force', action='store_true', default=False, help='overwrites existing folder')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()
    table_format = args.table_format

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    def get_parent(path):
        return os.path.basename(os.path.dirname(path))


    if args.dirs is not None:

        quant = pd.DataFrame()
        index_ext=[0]
        for d in args.dirs:

            f = os.path.join(d, "eval_copy.pth")
            chkpt_f = torch.load(f, map_location=device)
            quant_f = chkpt_f["quant"]
            ndraw = index_ext[-1]
            index_ext = ndraw + quant_f.index
            quant_f.index = index_ext

            quant = pd.concat([quant, quant_f])

        chkpt = copy.copy(chkpt_f)
        chkpt['args'].steps=ndraw
        chkpt['args'].name = args.name
        chkpt['quant'] = quant
        torch.save(chkpt, os.path.join(os.path.dirname(d.rstrip(os.sep)), args.name, 'eval_copy.pth'))


    else
        for d_A, d_B in zip(args.infile, args.outfile):

            f_A = os.path.join(args.root, d_A, "eval_copy.pth")  # all the saved results
            f_B = os.path.join(args.root, d_B, "eval_copy.pth")  # all the saved results
            # unique_ids = set(list(map(get_parent, lst_file)))
            chkpt_A = torch.load(f_A, map_location=device)
            chkpt_B = torch.load(f_B, map_location=device)

            quant_A = chkpt_A['quant']
            quant_B = chkpt_B['quant']
            chkpt_C = copy.copy(chkpt_B)
            index_ext = quant_A.index[-1] + quant_B.index
            quant_B.index = index_ext

            quant_C = pd.concat([quant_A, quant_B])
            chkpt_C['quant'] = quant_C

            if args.rename :

                d_C = os.path.join(args.root, args.rename)
                # shutil.move(d_B, d_C)
            else:
                d_C = os.path.join(args.root, f'{d_A}-{d_B}')
            # chkpt_B['args'].name = args.rename

            chkpt_C['args'].name = os.path.basename(d_C)
            os.makedirs(d_C, exist_ok=args.force)
            f_C = os.path.join(d_C, "eval_copy.pth")

            torch.save(chkpt_C, f_C)


