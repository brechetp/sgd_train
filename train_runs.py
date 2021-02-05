import torch
import numpy as np
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import shutil

import models
import random

import torch.optim
import torch
import argparse
import utils
import subprocess
import glob

#from torchvision import models, datasets, transforms
from os.path import join as join_path

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x

def main(argv):
    SLURM='slurm'
    TEMPLATE=f'{SLURM}/template.sbatch'

    for name in argv.names:  # for the different experiments
        logs = [f for f in glob.glob(join_path(name, "**", 'logs.txt'), recursive=True)]

        for f in logs:
            # each file is a setting
            dirname, basename = f.split()


            path_checkpoints = os.path.join(dirname, 'checkpoints')
            os.makedirs(path_checkpoints, exist_ok=True)
            if os.path.isfile(join_path(dirname, 'checkpoint.pth')):  # on the first average need to make a folder
                shutil.move(join_path(dirname, 'checkpoint.pth'), join_path(path_checkpoints, 'checkpoint-r1.pth'))
                id_run=2
            previous_runs = len(os.listdir(path_checkpoints))>0

            chkpts, id_run = utils.get_id_run(path_checkpoints)
            id_run += 1


            id_batch = '-'.join(s.replace('-', '')[:3] for s in dirname.split(os.sep())[-3:])
            fname = join_path(SLURM, '{}.sbatch'.format(id_batch))
            shutil.copyfile(TEMPLATE, fname)

            with open(fname, 'a') as fbatch:

                for idx in range(id_run, id_run+argv.nruns):

                    print(f"srun python train_mnist.py {dirname} --average --id_run {id_run}", file=fbatch)
            proc = subprocess.Popen('sbatch', fname)
            try:
                outs, errs = proc.communicate()
            except:
                proc.kill()
