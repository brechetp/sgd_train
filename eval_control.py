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
import seaborn as sns
sns.set_theme()
sns.set_style('whitegrid')
import glob

import math

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

def select_min(df):

    n_layers = len(quant.columns.levels[0])
    #columns = quant.columns.name
    indices = np.zeros(n_layers, dtype=int)
    Idx = pd.IndexSlice

    for idx in range(n_layers):
        # replace NaN with 0
        val_min = df.loc[:, (idx, 'error')].min()
        mask = df.loc[:, (idx, 'error')] == val_min
        indices[idx] = df.loc[mask, (idx, 'loss')].idxmin()  # if several min, take the min of them
        # the indices for the try that has the minimum training
        # error at the epoch epoch

    # remove the column index 'try'
    cols = pd.MultiIndex.from_product(df.columns.levels, names=df.columns.names)  # all but the try
    df_min = pd.DataFrame(columns=cols, index=[1])
    df_min.index.name = 'step'

    for idx in range(n_layers):
        # select the try that has the minimum training error at the
        # last epoch (at index indices[idx])
        df_min.loc[1, Idx[idx, :]] = df.loc[indices[idx],Idx[idx, :]]#.droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('layer') == 'last'] = df.xs(('last', idx_last), axis=1, level=[2, 3], drop_level=False).droplevel('try', axis=1)
    #df_min.loc[:, df_min.columns.get_level_values('stat') == 'err'] *= 100
    #df_min = df_min.loc[pd.IndexSlice[:, df_min.columns.get_level_values('layer').isin(range(1, n_layers+1))]]

    #if not df.loc[epoch,  ('train', 'err', 1, indices[0] )] == 0:
    #    print('Not separated!', dirname)
    #else:
    return df_min

    #    print('Separated!', dirname)

def process_checkpoint(checkpoint):
    '''Read and process a previously computed result stored inside a checkpoint'''

    quant = checkpoint['quant']
    args = checkpoint['args']
    process_df(quant, args.path_output)
    return

def process_df(quant, dirname, args=None, args_model=None, save=True):

    idx = pd.IndexSlice
    n = len(quant.columns.levels)
    losses = quant.xs('loss', level=n-1, axis=1)
    cols_error = idx[:, :, 'error'] if n == 3 else idx[:, 'error']
    quant.loc[:, cols_error] *= 100  # in %
    N_L = len(quant.columns.levels[n-2])
    errors = quant.loc[:, cols_error]

    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))

    losses.to_csv(os.path.join(dirname, 'losses.csv'))
    errors.to_csv(os.path.join(dirname, 'errors.csv'))

    losses.describe().to_csv(os.path.join(dirname, 'losses_describe.csv'))
    errors.describe().to_csv(os.path.join(dirname, 'errors_describe.csv'))

    #fig=plt.figure()
    f, axes = plt.subplots(1, 2, figsize=[10., 5.])
    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='steps')
    bp1 = sns.boxplot(
        data = df_plot.query('layer > 0 & stat == "error"'),
        #col='log_mult',
        #hue='set',
        #col='stat',
        # row='stat',
        #col='log_mult',
        x='layer',
        y='value',
        ax=axes[0],
        #kind='line',
        #ylabel='%',
        #ci=100,
        #col_wrap=2,
        #facet_kws={
        #    'sharey': False,
        #    'sharex': True
        #}
    )
    axes[0].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[quant.loc[1, (0, "error")]], color="red")
    axes[0].set_title("Error")
    axes[0].set_ylabel("error (%)")

    bp2 = sns.boxplot(
        data = df_plot.query('layer >0 & stat =="loss"'),
        x="layer",
        y="value",
        ax=axes[1]
    )
    axes[1].plot(np.linspace(0, N_L, num=N_L)-1, (N_L)*[quant.loc[1, (0, "loss")]], color="red", label="full network")
    axes[1].set_title("Loss")
    axes[1].set_ylabel("loss")
    #plt.legend()
    f.legend()
    f.subplots_adjust(top=0.85, left=0.10)
    plt.savefig(fname=os.path.join(dirname, 'boxplot.pdf'))

    rp = sns.relplot(
        data = df_plot.query('layer > 0'),
        #col='log_mult',
        #hue='set',
        col='stat',
        # row='stat',
        #col='log_mult',
        x='layer',
        y='value',
        #ax=axes[0],
        kind='line',
        legend="full",
        #ylabel='%',
        #ci=100,
        #col_wrap=2,
        facet_kws={
            'sharey': False,
            'sharex': True,
            'legend_out':True,
        }
    )


    rp.axes[0,0].set_ylabel("error (%)")
    rp.axes[0,1].set_ylabel("loss")

    rp.axes[0,0].plot(np.linspace(1, N_L-1, num=N_L), (N_L)*[quant.loc[1, (0, "error")]], color="red")
    rp.axes[0,1].plot(np.linspace(1, N_L-1, num=N_L), (N_L)*[quant.loc[1, (0, "loss")]], color="red", label="full network")
    #rp.axes[0,1].legend()
    #plt.legend()
    rp.fig.legend()
    #rp.fig.subplots_adjust(top=0.9, left=1/rp.axes.shape[1] * 0.1)
    rp.fig.subplots_adjust(top=0.85, left=0.10)
    if args_model is not None and args is not None:
       removed = "width / {}".format(args.fraction) if hasattr(args, 'fraction') and args.fraction is not None else args.remove
       rp.fig.suptitle('ds = {}, width = {}, removed = {}, steps = {}'.format(args_model.dataset, args_model.width, removed, args.ntry))
    #rp.set(yscale='log')
    #rp.set(ylabel='%')
    plt.savefig(fname=os.path.join(dirname, 'relplot.pdf'))

    only_min = select_min(quant)  # select the draw with minimum error
    only_min_plot = pd.melt(only_min.reset_index(), id_vars='step')



    m = sns.relplot(data=only_min_plot.query('layer > 0'),
    #m = only_min_plot.plot(x='layer', kind='line', y='value')
        col='stat',
        x='layer',
        y='value',
        kind='scatter',
        facet_kws={
            'sharey': False,
            'sharex': True
        }
    )
    plt.savefig(fname=os.path.join(dirname, 'plot_min.pdf'))
    return

def process_csv(file_csv):
    '''Read and process a previously computed result stored inside a checkpoint'''

    idx = pd.IndexSlice
    quant = pd.read_csv(file_csv, header=[0,1], index_col=0)
    process_df(quant, os.path.dirname(file_csv))
    return

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Evaluating a copy of a classifier with removed units')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-o', type=str, help='output root for the results')
    parser.add_argument('--name', default='eval-copy', type=str, help='the name of the experiment')
    parser.add_argument('--vary_name', nargs='*', default=['depth', 'width'], help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--lr_update', type=int, default=0, help='update for the learning rate in epochs')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('-R', '--remove', type=int, default=100, help='the number of neurons to remove at each layer')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--directory', help='path of the directory where results are')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser_model.add_argument('--csv', help='path of the previous saved csv file')
    parser_model.add_argument('--width', type=int, help='width for a random network')
    parser.add_argument('--reset_random', action='store_true', help='randomly reset the model')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth', type=int, help='the depth for init network')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.add_argument('--steps', type=int, default=20, help='The number of steps to take')
   # parser.add_argument('--log_mult', type=int, default=1, help='The log2 of the mulltiplicative factor')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:  # process the  previous computation
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            #args.nepochs = nepochs
            #cont = True  # continue the computation
            process_checkpoint(checkpoint)
            sys.exit(0)
        except RuntimeError:
            print('Could not load the model')

    elif args.csv is not None:  # process the csv
        process_csv(args.csv)
        sys.exit(0)

    elif args.directory is not None: # proces the whole directory with results from different mult factors

        try:
            chkpt_model = torch.load(os.path.join(os.path.dirname(args.directory.rstrip(os.sep)), "checkpoint.pth"), map_location=device)
        except:
            chkpt_model = torch.load(os.path.join(args.directory, "eval_copy.pth"), map_location=device)

        args_model = chkpt_model['args']
        N_L = args_model.depth
        N_T = args.steps
        layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
        stats = ['loss', 'error']
        #tries = np.arange(1, 1+args.ntry)  # the different tries


        lst_file = glob.glob(os.path.join(args.directory, "eval_copy*.pth"), recursive=False)
        Idx = pd.IndexSlice

        if len(lst_file)  > 1:  # more than one file to glue together
            #index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')
            #index = pd.Index(np.arange(1, N_T+1), name='steps')
            df_bundle = pd.DataFrame()
            #df_bundle = pd.DataFrame(columns=columns, dtype=float)
            epochs = {}

            df_bundle.sort_index(axis=1, inplace=True)  # sort for quicker access

            for filename in lst_file:

                #match = regex_entry.search(file_entry)
                #if match is None:
                #    continue
                checkpoint = torch.load(filename, map_location=device)
                args_copy = checkpoint['args']
                log_mult = args_copy.log_mult#int(match.groups()[0])
                #columns=pd.MultiIndex.from_product([[log_mult], layers, stats], names=names)
                #epoch = checkpoint['epochs']
                quant = checkpoint['quant'].loc[:, (layers, slice(None))]
                C = len(quant.columns)
                quant.columns = pd.MultiIndex.from_arrays([C*[log_mult], quant.columns.get_level_values(0), quant.columns.get_level_values(1)],
                                                        names=['log_mult'] + quant.columns.names
                                                        )

                df_bundle = pd.concat([df_bundle, quant], ignore_index=False, axis=1)
                #df_bundle.loc[:, (log_mult, layers, 'loss')] = quant.xs('loss', level=1, axis=1)
                #df_bundle.loc[Idx[:, (log_mult, layers, 'error')]] = quant.xs('error', level=1, axis=1)
                #epochs[idx_entry] = epoch

        else:
            # no need to bundle

            names=['layer', 'stat']
            filename = lst_file[0]
            checkpoint = torch.load(filename, map_location=device)
            args_copy = checkpoint['args']
            #log_mult = args_copy.log_mult#int(match.groups()[0])
            #columns=pd.MultiIndex.from_product([[log_mult], layers, stats], names=names)
            #epoch = checkpoint['epochs']
            quant = checkpoint['quant']
            df_bundle = quant


        df_bundle = df_bundle.sort_index(level=0, axis=1)
        process_df(df_bundle, args.directory)
        sys.exit(0)


    elif args.width is not None:
        # new network with random init
        checkpoint = dict()

        dataset = args.dataset
        args.output_root = utils.get_output_root(args)
        args.name = utils.get_name(args)

        path_output = os.path.join(args.output_root, args.name)
        train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                            dataroot=args.dataroot,
                                                                 )

        train_loader, size_train,\
            val_loader, size_val,\
            test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None, pin_memory=True)

        num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
        imsize = next(iter(train_loader))[0].size()[1:]
        input_dim = imsize[0]*imsize[1]*imsize[2]


        min_width = max_width = args.width
        model = models.classifiers.FCNHelper(num_layers=args.depth,
                                            input_dim=input_dim,
                                            num_classes=num_classes,
                                            min_width=min_width,
                                            max_width=max_width,
                                            shape='square',
                                             )

    else:
        checkpoint = dict()

        try:
            checkpoint_model = torch.load(args.model, map_location=device)  # checkpoint is a dictionnary with different keys
        except RuntimeError as e:
            print('Error loading the model at {}'.format(e))

        args_model = checkpoint_model['args']  # restore the previous arguments
        imresize = checkpoint_model.get('imresize', None)







    # if args.vary_name is not None:
        # name = ''
        # for field in args.vary_name:
            # if field in args:
                # arg = args.__dict__[field]
                # if isinstance(arg, bool):
                    # dirname = f'{field}' if arg else f'no-{field}'
                # else:
                    # val = str(arg)
                    # key = ''.join(c[0] for c in field.split('_'))
                    # dirname = f'{key}-{val}'
                # name = os.path.join(name, dirname)
        # args.name = name if name else args.name

        if args.reset_random:
            args.name += '-reset'
        path_output = os.path.join(args_model.output_root, args_model.name, args.name)
    # Logs
        log_fname = os.path.join(args_model.output_root, args_model.name, 'logs.txt')
        train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                            dataroot=args_model.dataroot,
                                                                imresize =imresize,
                                                                )

        train_loader, size_train,\
            val_loader, size_val,\
            test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args_model.batch_size, ss_factor=1, size_max=args_model.size_max, collate_fn=None, pin_memory=True)

        archi = utils.parse_archi(log_fname)
        model = utils.construct_FCN(archi)

        if not args.reset_random:
            try:
                model.load_state_dict(checkpoint_model['model'])
            except RuntimeError as e:
                print("Can't load mode (error {})".format(e))




    os.makedirs(path_output, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(path_output, 'logs_eval.txt'), 'w')
    else:
        logs = sys.stdout
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    #imresize = (256, 256)
    #imresize=(64,64)]

    #model = models.cnn.CNN(1)




    #min_width = int(args.coefficient *math.sqrt(size_train)+0.5)
    #max_width = int(3*args.coefficient *math.sqrt(size_train)+0.5)
    #model = models.classifiers.FCN3(input_dim=input_dim, num_classes=num_classes, min_width=min_width, max_width=max_width)




    model.requires_grad_(False)
    model.eval()

    model.to(device)
    #if 'classifier' in checkpoint.keys():
    #    classifier.load_state_dict(checkpoint['classifier'])

    #num_parameters = utils.num_parameters(classifier)
    num_layers = 1
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    #print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(classifier.size_out), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)






    #print('Linear classifier: {}'.format(str(classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    #parameters = classifier.network.parameters()
    #if len(args.learning_rate) <= 2:
    #    lr_range = 2*args.learning_rate if len(args.learning_rate) == 1 else args.learning_rate
        #lrs = np.linspace(lr_range[0], lr_range[1], classifier.n_layers)
   #     lrs = np.geomspace(lr_range[0], lr_range[1], classifier.n_layers)
    #elif len(args.learning_rate) == classifier.n_layers:
    #    lrs  = args.learning_rate
   # else:
   #     raise ValueError('Parameter learning_rate not understood (n_layers ={}, #lr = {})'.format(classifier.n_layers, len(args.learning_rate)))

   # param_list = [{'params': param, 'lr': lr} for param, lr in zip(parameters, lrs)]


    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    def zero_one_loss(x, targets):
        ''' x: BxC
        targets: Bx1

        returns: err of size 1
        '''
        return  (x.argmax(dim=1)!=targets).float().mean(dim=0)

    #mse_loss = nn.MSELoss()
    #ce_loss_check = nn.CrossEntropyLoss(reduction='none')

    def ce_loss(input, target):
        '''Batch cross entropy loss

        input: BxC output of the linear model
        target: Bx1: the target classes

        output: B the loss for each try
        '''


        B, C = input.size()
        cond = input.gather(1,target.view(-1, 1)).squeeze(1)  # Bx1
        output = - cond + input.logsumexp(dim=-1)
        return output

    #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

    #sets = ['train', 'test']
    N_L = utils.count_hidden_layers(model.main)
    layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
    #log_mult = np.arange(1, N_L+1)
    stats = ['loss', 'error']
    #tries = np.arange(1, 1+args.ntry)  # the different tries

    names=['layer', 'stat']
    columns=pd.MultiIndex.from_product([layers, stats], names=names)
    #index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')
    index = pd.Index(np.arange(1, args.steps+1), name='steps')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    quant.sort_index(axis=1, inplace=True)  # sort for quicker access

    #if 'quant' in checkpoint.keys():
    #    quant.update(checkpoint['quant'])



    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes


    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, optimizer, lr_scheduler, epoch#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                #'stats': stats,
            'quant': quant,
                'args': args,
            #'log_mult': args.log_mult,
          #  'args_model': args_model,
                #'optimizer': optimizer.state_dict(),
                #'epochs': epoch,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint

    def save_checkpoint(fname=None, checkpoint=None):
        '''Save checkpoint to disk'''

        global path_output

        if fname is None:
            fname = os.path.join(path_output, 'eval_copy.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)






    #ones = torch.ones(args.ntry, device=device, dtype=dtype)

    def eval_epoch(classifier, dataloader):


        classifier.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_train = 0
        err_train = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):


                x = x.to(device)
                y = y.to(device)
                out_class = classifier(x)  # BxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB
                err = zero_one_loss(out_class, y)  # T
                err_train = (idx * err_train + err.detach().cpu().numpy()) / (idx+1)  # mean error
                loss_train = (idx * loss_train + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)

        return loss_train, err_train


    loss_0, error_0 = eval_epoch(model, train_loader)  # original loss and error of the model
    print(f'loss: {loss_0}, error: {error_0}', file=logs, flush=True)
    #stats_0 = pd.DataFrame(columns=['loss', 'error'])
    #stats_0['loss'] = loss_0
    #stats_0['error'] = error_0
    #stats_0.to_csv(os.path.join(path_output, 'original.csv'))
    quant.loc[1, [0, 'loss']] = loss_0
    quant.loc[1, [0, 'error']] = error_0


    #mult = 2**args.log_mult
    for t in range(1, args.steps+1):
        for l in range(1, N_L+1):

            mult = 2**(N_L+1-l)
            classifier = models.classifiers.ClassifierCopyFCN(model,l,mult).to(device)
            loss, error = eval_epoch(classifier, train_loader)


            quant.loc[pd.IndexSlice[t, (l, 'loss')]] =  loss
            quant.loc[pd.IndexSlice[t, (l, 'error')]] =  error


            print('mult: {}, t: {}, l: {}, loss: {}, error: {}'.format(mult, t, l, loss, error), file=logs, flush=(l==N_L))


    quant.to_csv(os.path.join(path_output, 'quant.csv'))
    save_checkpoint()
    process_df(quant, path_output)
        #end = err_train.max(axis=1).nonzero()[0].max() + 1  # requires _all_ the tries to be 0 to stop the computation, 1 indexed
        #if args.end_layer is not None:
        #    end = min(end, args.end_layer)

        #ones = ones[:end, :]
        #optimizer.param_groups,new_params_disc  = optimizer.param_groups[:end], optimizer.param_groups[end:]  # trim the parameters accordingly




        #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(



