import utils

import torch
import torchvision.utils as vutils
import os
import numpy as np
import math
from subprocess import Popen, PIPE
import argparse
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()
import shutil
import pdb
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


def np_to_pd(stats):
    '''convert a dict of numpy arrays into a pandas dataframe'''
    index=pd.Index(stats.pop('epochs'), name='epoch')
    #df = pd.concat( [pd.DataFrame(val, index=index) for val in stats.values()],axis=1)  # create a big frame  for train and test
    name_columns = []
    names = []
    df = None
    for key,val in stats.items():
        if type(val) is list and len(val) == len(index):
            df_new = pd.DataFrame(val, index=index)
            name_columns +=  len(df_new.columns) * [key]
            names += [key]
            if df is None:
                df = df_new
            else:
                df = pd.concat([df, df_new], axis=1)

    #index=pd.MultiIndex.from_product([['train', 'test'], ['zo', 'ce']], names=['dataset', 'type'])
    N = len(df.columns) //2
    #names = ['dataset', 'type'] name_columns is None else name_columns
    #columns=pd.MultiIndex.from_arrays([ N*['train'] + N*['test'], df.columns], names=names)  # set the different indexes for the array
    names=['stat', 'try']
    columns=pd.MultiIndex.from_arrays([name_columns, df.columns], names=names)  # set the different indexes for the array
    #df.values = {'loss': df.values}
    df.columns = columns
    return df

def eval_model_df(df, output_path):
    '''Plot the model with pandas dataframe'''

    df_reset = df.reset_index()
    df_plot = pd.melt(df_reset, id_vars='epoch')
    g = sns.relplot(
        data = df_plot,
        col='stat',
        #hue='dataset',
        x='epoch',
        y='value',
        kind='line',
    )
    plt.savefig(fname=os.path.join(output_path, 'losses.pdf'))


def eval_model(stats, output_path):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(stats['epochs'], stats['loss_train']['zo'], label='Train')
    if  'loss_test' in stats.keys():
        ax.plot(stats['epochs'], stats['loss_test']['zo'],   label='Test')
    #            yerr=stats_acc['loss_train']['zo'][:, :epoch].std(axis=0),
    #            label='Train')
    #ax.errorbar(stats['epochs'], stats_acc['loss_test']['zo'][:, :epoch].mean(axis=0),
    #            yerr=stats_acc['loss_test']['zo'][:, :epoch].std(axis=0),
    ax.legend()
    ax.set_ylabel('Error')
    ax.set_xlabel('Epoch')
    ax.set_title('Classification error')
    ax.set_yscale('log')
    plt.savefig(fname=os.path.join(output_path, 'zero_one_loss.pdf'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stats['epochs'], stats['loss_train']['ce'], label='Train')
    ax.plot(stats['epochs'], stats['loss_test']['ce'],   label='Test')
    #ax.errorbar(stats['epochs'], stats_acc['loss_train']['ce'][:, :epoch].mean(axis=0),
    #        yerr=stats_acc['loss_train']['ce'][:, :epoch].std(axis=0),
    #        label='Train')
    #ax.errorbar(stats['epochs'], stats_acc['loss_test']['ce'][:, :epoch].mean(axis=0),
    #        yerr=stats_acc['loss_test']['ce'][:, :epoch].std(axis=0),
    #        label='Test')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (nats)')
    ax.set_title('Cross-entropy loss')
    ax.set_yscale('linear')
    plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss.pdf'))

    #fig=plt.figure()
    #plt.plot(stats['epochs'], stats['lr'], label='lr')
    #plt.legend()
    #plt.savefig(fname=os.path.join(path_output, 'lr.pdf'))

    plt.close('all')

def eval_lin_df(df, output_path):
    '''process the dataframe from a linear separation result'''


    fig=plt.figure()
    df_reset = df.reset_index()
    df_plot = pd.melt(df_reset, id_vars='epoch')
    g = sns.relplot(
        data = df_plot,
        col='stat',
        #hue='dataset',
        x='epoch',
        y='value',
        kind='line',
        col_wrap=4,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
    )
    plt.savefig(fname=os.path.join(output_path, 'losses_try.pdf'))


def eval_lin(stats, output_path):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(stats['epochs'], stats['loss_train'])

    if  'loss_test' in stats.keys():
        ax.plot(stats['epochs'], stats['loss_test'])
    #            yerr=stats_acc['loss_train']['zo'][:, :epoch].std(axis=0),
    #            label='Train')
    #ax.errorbar(stats['epochs'], stats_acc['loss_test']['zo'][:, :epoch].mean(axis=0),
    #            yerr=stats_acc['loss_test']['zo'][:, :epoch].std(axis=0),
    ax.set_ylabel('loss (nats)')
    ax.set_xlabel('Epoch')
    ax.set_title('Cross-entropy loss')
    ax.set_yscale('log')
    plt.savefig(fname=os.path.join(output_path, 'loss.pdf'))

    if 'loss_hidden_train' in stats.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(stats['epochs'], stats['loss_hidden_train'], marker='o')

        ax.legend([f'Layer {i}' for i in range(1, 1+stats['loss_hidden_train'][0].shape[0])])

        #ax.errorbar(stats['epochs'], stats_acc['loss_train']['ce'][:, :epoch].mean(axis=0),
        #        yerr=stats_acc['loss_train']['ce'][:, :epoch].std(axis=0),
        #        label='Train')
        #ax.errorbar(stats['epochs'], stats_acc['loss_test']['ce'][:, :epoch].mean(axis=0),
        #        yerr=stats_acc['loss_test']['ce'][:, :epoch].std(axis=0),
        #        label='Test')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss (nats)')
        ax.set_title('Cross-entropy loss at intermediate layers')
        ax.set_yscale('linear')
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss_hidden_train.pdf'))

    if  'loss_hidden_test' in stats.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(stats['epochs'], stats['loss_hidden_test'])
        #ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        ax.legend([f'Layer {i}' for i in range(1, 1+stats['loss_hidden_test'][0].shape[0])])
        ax.set_title('Cross-entropy loss for the layers')
        ax.set_yscale('linear')
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss_hidden_test.pdf'))
    #fig=plt.figure()
    #plt.plot(stats['epochs'], stats['lr'], label='lr')
    #plt.legend()
    #plt.savefig(fname=os.path.join(path_output, 'lr.pdf'))

    plt.close('all')


def main(argv):
    '''find the different models saved in argv.root and write the images to argv.output'''


    names = argv.exp_names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    for name in names:  # for the different experiments
        file_lst = [f for f in glob.glob(os.path.join(name, "**", "checkpoint*.pth"), recursive=True)]
        for f in file_lst:
            dirname, basename = os.path.split(f)
            log_fname = [lf for lf in glob.glob(os.path.join(dirname, '*.txt'), recursive=False) if 'log' in lf]
            if not log_fname:
                log_fname += [lf for lf in glob.glob(os.path.join(os.path.dirname(dirname), '*.txt'), recursive=False) if 'log' in lf]
            log_fname = log_fname[0]  # should be alone
            try:
                if not os.path.isfile(log_fname):
                        raise ValueError
                #  rootname = os.path.join(opt.path, 'imgs')
                #args = [
                #        ('batch_size', int),
                #        ('epochs', int),
                #        ('dataset', str),
                #        ('dataroot', str),
                #        ('output_root', str),
                #        ('name', str),
                #    ('learning_rate', float),
                #    ('loss', str),
                #    ('width', int),
                #        ]
                #d_args = utils.parse_log_file(log_fname, *args)
                checkpoint = torch.load(f, map_location=device)
                args = checkpoint['args']
                batch_size = args.batch_size
                #dataset = args.dataset
                #dataroot = args.dataroot

                #archis = utils.parse_archi(log_fname)
                #transform = utils.parse_transform(log_fname)

                #train_ds, test_ds, num_chs = utils.get_dataset(dataset, dataroot, transform=transform)

                #train_loader,train_size,\
                #    val_loader, val_size,\
                #    test_loader, test_size = utils.get_dataloader(train_ds, test_ds, batch_size, ss_factor=0.1, download=False)


                epoch = checkpoint['epochs']
                stats = checkpoint['stats']

                df = np_to_pd(stats)

                output_path =    os.path.join(dirname, 'eval', f'e-{epoch:03d}')
                if os.path.exists(output_path) and not argv.force:
                    continue
                os.makedirs(output_path, exist_ok=True)

                if 'lin' in basename:
                    #eval_lin(stats, output_path)
                    eval_lin_df(df, output_path)
                else:
                    #eval_model(stats, output_path)
                    eval_model_df(df, output_path)
                plt.close('all')



            except BaseException as e:
                print(f'Error {str(e)} at line {e.__traceback__.tb_lineno} for file {f}')
                continue

def preprocess_stats(stats):

    '''return a numpy 2d array with correct type'''

    names =  list(stats.keys())
    d = len(stats)
    formats = d* ['f8']
    dtype = dict(names=names, formats=formats)


    Ns = [v.shape[0] for v in stats.values() if v is not None]
    same_size = [n == m for (n, m) in zip(Ns[:-1], Ns[1:])]

    assert all(same_size)
    N = Ns[0]
    #subarray = np.zeros((d,N), dtype=np.float64)

    nan_array = np.nan* np.empty((N,))
    for k, v in stats.items():
        if v is None:
            stats[k] = nan_array

    subarray  = np.array(list(stats.values()), dtype=np.float64).T.copy().view(dtype)

    return subarray



def save_meta(meta_npz_fname, stats, args):
    '''accumulate the obeservation in a meta file'''


    #N_train = train_array.shape[1]
    #N_test = test_array.shape[1]
    loss_train, loss_test = stats['loss_train']['zo'][-1], stats['loss_test']['zo'][-1]
    num_parameters = stats['num_parameters']
    new_entry_train = np.array([(num_parameters, args.__dict__, loss_train)], dtype=[('num_parameters', np.int32 ), ('args', dict), ('loss', np.float32)])
    new_entry_test = np.array([(num_parameters, args.__dict__, loss_test)], dtype=[('num_parameters', np.int32), ('args', dict), ('loss', np.float32)])
    if os.path.isfile(meta_npz_fname):
        meta_data = np.load(meta_npz_fname, allow_pickle=True)

        meta_train = meta_data['train']
        meta_test = meta_data['test']
        meta_nparam = meta_train['num_parameters']
        #names = meta_train['label'].squeeze()
        if num_parameters in meta_nparam:
            # we found a duplicate with the same number of parameters
            idx = np.where(num_parameters == meta_nparam)[0]
            meta_train[idx] = new_entry_train
            meta_test[idx] = new_entry_test
            if len(idx) >=2:
                print('Warning, more than one ducplicate of {} in {}'.format(num_parameters, meta_npz_fname))
        else:

            meta_train = np.vstack((meta_train, new_entry_train))
            meta_test = np.vstack((meta_test, new_entry_test))
    else:
        meta_train = new_entry_train
        meta_test =  new_entry_test
    #meta_train[exp_name] =
    np.savez_compressed(meta_npz_fname, train=meta_train, test=meta_test)

    return





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='script for generating and saving images')
    parser.add_argument('exp_names',  nargs='*', help='the experiment names to resume')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-f', '--force', default=False, action='store_true', help='force the computation again')
    parser.add_argument('--save_meta', action='store_true', help='saves data for meta comparison')
    #  parser.add_argument('--dry_run', action='store_true', help='dry run mode (do not call sbatch, only copy the files)')
    #  parser.add_argument('--start_idx', type=int, default=1, help='the number of index to start from')
    #  parser.add_argument('--config', help='the default configuration to start from')
    #  parser.add_argument('--batch_template',  default='slurm/scripts/template.sbatch', help='the template sbatch file')
    #  parser.add_argument('script', help='the training script to use')
    #  parser.add_argument('--force_resume', action='store_true', default=False, help='if true, we resume even runs already resumes')
    #  parser.add_argument('--no-touch_resume', action='store_false', dest='touch_resume',  help='if true, we resume even runs already resumes')
    #  iter_parser  = parser.add_mutually_exclusive_group(required=False)
    filter_list = parser.add_mutually_exclusive_group(required=False)
    filter_list.add_argument('--whitelist', help='whitelisting the suffix')
    filter_list.add_argument('--blacklist', help='blacklisting the suffix')
    argv = parser.parse_args()

    main(argv)
