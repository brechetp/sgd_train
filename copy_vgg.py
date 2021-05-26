import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()
sns.set_style('whitegrid')
import glob
from scipy.optimize import minimize, minimize_scalar

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
    #losses = quant.xs('loss', level=n-1, axis=1)
    #cols_error = idx[:, :, 'error'] if n == 3 else idx[:, 'error']
    col_order = ["stat", "set", "layer"]
    if quant.columns.names != col_order:
        # the order is
        # perform pivot
        quant = pd.melt(quant.reset_index(), id_vars="steps").pivot(index="steps", columns=col_order, values="value")

    cols_error = idx['error', :, :]
    quant.loc[:, cols_error] *= 100  # in %
    N_L = len(quant.columns.unique(level="layer")) -1 # number of hidden layers
    errors = quant["error"]
    losses = quant["loss"]
    #errors = quant["error"]


    if save:
        quant.to_csv(os.path.join(dirname, 'quant.csv'))

    losses.to_csv(os.path.join(dirname, 'losses.csv'))
    errors.to_csv(os.path.join(dirname, 'errors.csv'))

    losses.describe().to_csv(os.path.join(dirname, 'losses_describe.csv'))
    errors.describe().to_csv(os.path.join(dirname, 'errors_describe.csv'))

    # rel_error = pd.DataFrame()
    # rel_losses = pd.DataFrame()
    # for W in quant.columns.levels[2]:  # for each width
        # idx_col = (errors.columns.get_level_values("layer") > 0) & (errors.columns.get_level_values("width") == W)
        # rel_error = pd.concat([rel_error, abs(errors.loc[:, idx_col] - errors[0][W][1]) / errors[0][W][1]], axis=1, ignore_index=False)
        # rel_losses = pd.concat([rel_losses,  abs(losses.loc[:, idx_col] - losses[0][W][1]) / losses[0][W][1]], axis=1, ignore_index=False)

    #rel_error_plot = pd.melt(rel_error.reset_index(), id_vars="steps")#, id_vars="steps")
    #rel_losses_plot = pd.melt(rel_losses.min(axis=0).reset_index(), id_vars="layer")#, id_vars="steps")

    df_reset = quant.reset_index()
    df_plot = pd.melt(df_reset, id_vars='steps')#.query("layer>0")
    df_plot_no_0 = df_plot.query('layer>0')
    df_plot_0 = df_plot.query('layer==0')
    mp = sns.relplot(
        #data=rel_losses.min(axis=0).to_frame(name="loss"),
        data=df_plot_no_0.pivot(index="steps", columns=col_order).min(axis=0).to_frame(name="value"),
        #hue="width",
        row="stat",
        col_order=["train", "test"],
        col="set",
        x="layer",
        y="value",
        kind='line',
        legend="full",
        palette=sns.color_palette(n_colors=N_L),
        #style='layer',
        markers=True,
        facet_kws={
            'sharey': False,
            'sharex': True
        }
        #y="value",
    )

    #plt.figure()

    mp.axes[0,0].set_ylabel("error (%)")
    #mp.axes[0,1].set_ylabel("error (%)")
    mp.axes[0,0].set_title("Train Error")
    mp.axes[0,0].plot(np.linspace(1, N_L, num=N_L), N_L*[quant.loc[1, ("error", "train", 0)]], color="red", label="full model")
    mp.axes[0,1].set_title("Test Error")
    mp.axes[0,1].plot(np.linspace(1, N_L, num=N_L), N_L*[quant.loc[1, ("error", "test", 0)]], color="red", label="full model" )

    mp.axes[1,0].set_title("Train Loss")
    mp.axes[1,0].set_ylabel("loss")
    mp.axes[1,0].plot(np.linspace(1, N_L, num=N_L), N_L*[quant.loc[1, ("loss", "train", 0)]], color="red", label="full model")
    mp.axes[1,1].set_title("Test Loss")
    mp.axes[1,1].plot(np.linspace(1, N_L, num=N_L), N_L*[quant.loc[1, ("loss", "test", 0)]], color="red", label="full model")
    for ax in mp.axes.ravel():
        ax.legend()

    #mp.axes[1,1].set_ylabel("loss")

    plt.savefig(fname=os.path.join(dirname, "min_quant.pdf"))

    #lp = rel_losses.min(axis=0).plot(kind='line', hue='width', x='layer')
    # losses_plot = pd.melt(losses.reset_index(), id_vars="steps").query("layer>0")
    # lp_loss = sns.lineplot(
        # #data=rel_losses.min(axis=0).to_frame(name="loss"),
        # data=losses_plot,
        # #hue="layer",
        # x="layer",
        # y="value",
        # legend="full",
        # #palette=sns.color_palette(n_colors=N_L-1),
        # #style='layer',
        # markers=True,
        # #y="value",
    # )
    # lp_loss.axes.set_ylabel("loss")
    # lp_loss.axes.set_title("Loss")
    # lp_loss.plot(np.linspace(1, N_L, num=N_L), (N_L)*[losses.loc[1,0]], color="red", label="full model")
    # lp_loss.legend()
    # #rp.axes[0,1].set_ylabel("loss")

    # #plt.figure()

    # #lp = rel_losses.min(axis=0).plot(kind='line', hue='width', x='layer')

    # sns.lineplot(
        # #data=rel_losses.min(axis=0).to_frame(name="loss"),
        # data=losses_plot.pivot(index="steps", columns=["layer"]).min(axis=0).to_frame(name="loss"),
        # #hue="layer",
        # x="layer",
        # y="loss",
        # legend="full",
        # ax=lp_loss,
         # color="green",
        # label="min",
        # #palette=sns.color_palette(n_colors=N_L-1),
        # #style='layer',
        # markers=True,
        # #y="value",
    # )
    # # lp_loss_min.axes.set_ylabel("loss")
    # # lp_loss_min.axes.set_title("Min loss")

    # #lp_loss_min.plot(np.linspace(1, N_L, num=N_L), (N_L)*[losses.loc[1,0]],  color="red", label="full model")
    # #lp_loss_min.legend()
    # #rp.axes[0,1].set_ylabel("loss")
    # plt.savefig(fname=os.path.join(dirname, "loss.pdf"))
    # #plt.savefig(fname=os.path.join(dirname, "loss_min.pdf"))

    # plt.figure()
    # errors_plot = pd.melt(errors.reset_index(), id_vars="steps").query("layer>0")

    # lp_error = sns.lineplot(
        # #data=rel_error.min(axis=0).to_frame(name="error"),
        # data=errors_plot,
        # #hue="layer",
        # x="layer",
        # y="value",
        # legend="full",
        # #palette=sns.color_palette(n_colors=N_L-1),
        # #style='layer',
        # markers=True,
        # #y="value",
    # )
    # lp_error.axes.set_ylabel("error (%)")
    # lp_error.axes.set_title("Error")
    # #rp.axes[0,1].set_ylabel("error")
    # lp_error.plot(np.linspace(1, N_L, num=N_L), (N_L)*[errors.loc[1,0]], color="red", label="full model")
    # lp_error.legend()


    # rp.axes[0,0].set_ylabel("error (%)")
    # rp.axes[0,1].set_ylabel("loss")

    # rp.axes[0,0].plot(np.linspace(1, N_L-1, num=N_L), (N_L)*[quant.loc[1, (0, "error")]], color="red")
    # rp.axes[0,1].plot(np.linspace(1, N_L-1, num=N_L), (N_L)*[quant.loc[1, (0, "loss")]], color="red", label="full network")
    # #rp.axes[0,1].legend()
    # #plt.legend()
    # rp.fig.legend()
    # #rp.fig.subplots_adjust(top=0.9, left=1/rp.axes.shape[1] * 0.1)
    # rp.fig.subplots_adjust(top=0.85, left=0.10)
    # if args_model is not None and args is not None:
       # removed = "width / {}".format(args.fraction) if hasattr(args, 'fraction') and args.fraction is not None else args.remove
       # rp.fig.suptitle('ds = {}, width = {}, removed = {}, steps = {}'.format(args_model.dataset, args_model.width, removed, args.ntry))
    # #rp.set(yscale='log')
    # #rp.set(ylabel='%')
    # plt.savefig(fname=os.path.join(dirname, 'relplot.pdf'))

    # only_min = select_min(quant)  # select the draw with minimum error
    # only_min_plot = pd.melt(only_min.reset_index(), id_vars='step')



    #m = sns.relplot(data=only_min_plot.query('layer > 0'),
    #m = only_min_plot.plot(x='layer', kind='line', y='value')
        # col='stat',
        # x='layer',
        # y='value',
        # kind='scatter',
        # facet_kws={
            # 'sharey': False,
            # 'sharex': True
        # }
    # )
    # plt.savefig(fname=os.path.join(dirname, 'plot_min.pdf'))
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
    parser_model.add_argument('--model', nargs='*', help='path of the model to separate')
    parser_model.add_argument('--root_model', nargs='*', help='path of the model to separate')
    #parser_model.add_argument('--model_directory', help='root of the models to separate (same plot)')
    parser_model.add_argument('--directory', nargs='*', help='path of the directory where results are')
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
    parser.add_argument('--optim_mult', action="store_true", help="flag to search for best multiplication factor")
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
                        # deprecated ?

        def get_parent(path):
            return os.path.basename(os.path.dirname(path))

        for directory in args.directory:

            lst_file = glob.glob(os.path.join(directory, "**", "eval_copy.pth"), recursive=True)  # all the saved results
            #unique_ids = set(list(map(get_parent, lst_file)))

            #for uid in unique_ids:
            #    id_lst_file = glob.glob(os.path.join(directory, "**", uid, "eval_copy.pth"), recursive=True)
            for f in lst_file:

                df_bundle = pd.DataFrame()
                directory = os.path.dirname(f)
                try:
                    chkpt_model = torch.load(os.path.join(os.path.dirname(os.path.dirname(f)), "checkpoint.pth"), map_location=device)
                except:
                    chkpt_model = torch.load(f, map_location=device)

                args_model = chkpt_model['args']
                N_L = args_model.depth
                N_T = args.steps
                layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
                stats = ['loss', 'error']
                sets = ['train', 'test']
                names=['set', 'layer', 'stat']
                #tries = np.arange(1, 1+args.ntry)  # the different tries


                #lst_file = glob.glob(os.path.join(args.directory, "eval_copy*.pth"), recursive=False)
                Idx = pd.IndexSlice


                # no need to bundle

                names=['layer', 'stat']
                filename = lst_file[0]
                checkpoint = torch.load(f, map_location=device)
                args_copy = checkpoint['args']
                #log_mult = args_copy.log_mult#int(match.groups()[0])
                #columns=pd.MultiIndex.from_product([[log_mult], layers, stats], names=names)
                #epoch = checkpoint['epochs']
                quant = checkpoint['quant']
                df_bundle = quant
                if 'df_mult' in checkpoint.keys():
                    df_mult = checkpoint['df_mult']
                    df_mult.describe().to_csv(os.path.join(directory, 'mult_describe.csv'))


            df_bundle = df_bundle.sort_index(level=0, axis=1)
            #process_df(df_bundle, directory)
        sys.exit(0)




    else:  # args.model should not be None
        if args.root_model is not None:

            lst_models = [glob.glob(os.path.join(rm, '**', 'checkpoint.pth'), recursive=True) for rm in args.root_model]
        elif args.model is not None:
            lst_models = [args.model]
        else:
            raise NotImplementedError

        for m in [m for lst in lst_models for m in lst]:
            checkpoint = dict()

            try:
                checkpoint_model = torch.load(m, map_location=torch.device('cpu'))  # checkpoint is a dictionnary with different keys
            except RuntimeError as e:
                print('Error loading the model at {}'.format(e))

            args_model = checkpoint_model['args']  # restore the previous arguments

            #model = utils.construct_FCN(archi)
            NUM_CLASSES = utils.get_num_classes(args_model.dataset)
            model,input_size  = models.pretrained.initialize_model(args_model.model, pretrained=False, freeze=True, num_classes=NUM_CLASSES)
            model.n_layers = utils.count_hidden_layers(model)

            imresize = checkpoint_model.get('imresize', input_size)








            if args.reset_random:
                args.name += '_reset'
            if args.optim_mult:
                args.name += '_optim-mult'
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

            #archi = utils.parse_archi(log_fname)



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
        print('Model: {}'.format(str(model)), file=logs)

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
        #print('Image dimension: {}'.format(imsize), file=logs)

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

        # def zero_one_loss_prime(x, targets):
            # ''' x: BxC
            # targets: Bx1

            # returns: err of size 1
            # '''
            # return  (x.argmax(axis=1)!=targets).float().mean(axis=0)

        def zero_one_loss(x, targets):
            ''' x: BxC
            targets: Bx1

            returns: err of size 1
            '''
            return  (x.argmax(axis=1)!=targets).float().mean(axis=0)

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
            output = - cond + input.logsumexp(axis=-1)
            return output

        def ce_loss_prime(input, target, M):
            '''Batch cross entropy loss

            input: BxC output of the linear model
            target: Bx1: the target classes

            output: B the prime loss for each try
            '''


            B, C = input.size()
            m_y = M * input.cpu().numpy()
            cond = input.gather(1,target.view(-1, 1)).squeeze(1)  # Bx1
            output = - cond + (np.exp(input * (m_y))).sum(axis=-1) / (np.exp(m_y)).sum(axis=-1)
            return output

        def ce_loss_hess(y, target, M):
            '''Batch cross entropy loss

            input: BxC output of the linear model
            target: Bx1: the target classes

            output: B the prime loss for each try
            '''


            B, C = y.size()
            y = y.cpu().numpy()
            y_2 = np.power(y, 2)
            m_y = M * y
            exp_m_y = np.exp(m_y)
            sum_exp_m_y = exp_m_y.sum(axis=1)
            y_exp_m_y = y * exp_m_y
            y_2_exp_m_y = y_2 * exp_m_y
            num = y_2_exp_m_y.sum(axis=1) * sum_exp_m_y - np.power(y_exp_m_y.sum(axis=1), 2)
            denom = np.power(sum_exp_m_y, 2)
            #cond = y.gather(1,target.view(-1, 1)).squeeze(1)  # Bx1
            output = num  / denom
            return output

        #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

        #sets = ['train', 'test']
        N_L = model.n_layers
        layers = np.arange(1, N_L+1)#classifier.n_layers)  # the different layers, forward order
        #log_mult = np.arange(1, N_L+1)
        stats = ['loss', 'error']
        #tries = np.arange(1, 1+args.ntry)  # the different tries

        names=['layer', 'stat', 'set']
        sets = ['train', 'test']
        columns=pd.MultiIndex.from_product([layers, stats, sets], names=names)
        #index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')
        index = pd.Index(np.arange(1, args.steps+1), name='steps')
        quant = pd.DataFrame(columns=columns, index=index, dtype=float)

        quant.sort_index(axis=1, inplace=True)  # sort for quicker access

        df_mult = pd.DataFrame(columns=[layers], index=index, dtype=float)

        #if 'quant' in checkpoint.keys():
        #    quant.update(checkpoint['quant'])



        #classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes


        def get_checkpoint():
            '''Get current checkpoint'''
            global model, stats, quant, df_mult, args, optimizer, lr_scheduler, epoch#, params_discarded, end

            #optimizer.param_groups = optimizer.param_groups + params_discarded

            checkpoint = {
                    'classifier': classifier.state_dict(),
                    #'stats': stats,
                'quant': quant,
                'df_mult': df_mult,
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






        def eval_class_mult(out_class, mult, only_loss=False):


            #classifier.eval()
            X, Y = out_class
            shape = X[0].shape
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_tot = 0
            err_tot = 0
            #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

            with torch.no_grad():
                for idx, (x, y) in enumerate(zip(X, Y)):
                    x = x.to(device)
                    y = y.to(device)
                    loss = ce_loss(x, y)  # LxTxB
                    loss_tot = (idx * loss_tot + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                    if not only_loss:
                        err = zero_one_loss(x, y)
                        err_tot = (idx * err_tot + err.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean err
                    # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                    #break


            if only_loss:
                return loss_tot
            else:
                return loss_tot, err_tot
        #ones = torch.ones(args.ntry, device=device, dtype=dtype)

        def eval_epoch(classifier, dataloader, with_error=True):


            classifier.eval()
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_tot = 0
            err_tot = 0
            #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

            with torch.no_grad():
                for idx, (x, y) in enumerate(dataloader):


                    x = x.to(device)
                    y = y.to(device)
                    out_class = classifier(x)  # BxC,  # each output for each layer
                    loss = ce_loss(out_class, y)  # LxTxB
                    loss_tot = (idx * loss_tot + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                    if with_error:
                        err = zero_one_loss(out_class, y)  # T
                        err_tot = (idx * err_tot + err.detach().cpu().numpy()) / (idx+1)  # mean error
                    # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                    #break


            if with_error:
                return loss_tot, err_tot
            else:
                return loss_tot

        def eval_epoch_prime(classifier, dataloader):
            '''Evals the derivative wrt. M of the loss'''

            classifier.eval()
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_tot = 0
            err_tot = 0
            #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

            with torch.no_grad():
                for idx, (x, y) in enumerate(dataloader):


                    x = x.to(device)
                    y = y.to(device)
                    out_class, M = classifier(x, prime=True)  # BxC,  # each output for each layer
                    loss = ce_loss_prime(out_class, y, M)  # LxTxB
                    #err = zero_one_loss_prime(M*out_class.cpu().numpy(), y)  # T
                    #err_tot = (idx * err_tot + err.detach().cpu().numpy()) / (idx+1)  # mean error
                    loss_tot = (idx * loss_tot + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                    # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                    #break

            return loss_tot

        def eval_epoch_hess(classifier, dataloader):
            '''Evals the derivative wrt. M of the loss'''

            classifier.eval()
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_tot = 0
            err_tot = 0
            #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

            with torch.no_grad():
                for idx, (x, y) in enumerate(dataloader):


                    x = x.to(device)
                    y = y.to(device)
                    out_class, M = classifier(x, prime=True)  # BxC,  # each output for each layer
                    loss = ce_loss_hess(out_class, y, M)  # LxTxB
                    #err = zero_one_loss_prime(M*out_class.cpu().numpy(), y)  # T
                    #err_tot = (idx * err_tot + err.detach().cpu().numpy()) / (idx+1)  # mean error
                    loss_tot = (idx * loss_tot + loss.mean(axis=-1)) / (idx+1)  # mean loss
                    # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                    #break

            return loss_tot



        loss_0, error_0 = eval_epoch(model, train_loader)  # original loss and error of the model
        loss_test, error_test = eval_epoch(model, test_loader)
        print(f'loss: {loss_0}, error: {error_0}', file=logs, flush=True)
        #stats_0 = pd.DataFrame(columns=['loss', 'error'])
        #stats_0['loss'] = loss_0
        #stats_0['error'] = error_0
        #stats_0.to_csv(os.path.join(path_output, 'original.csv'))
        Idx = pd.IndexSlice
        quant.loc[1, Idx[0, 'loss', 'train']] = loss_0
        quant.loc[1, Idx[0, 'error', 'train']] = error_0

        quant.loc[1, Idx[0, 'loss', 'test']] = loss_test
        quant.loc[1, Idx[0, 'error', 'test']] = error_test
        print(f'loss: {loss_test}, error: {error_test}', file=logs, flush=True)

        def get_output_class(classifier, loader):
            out = torch.empty((len(loader), loader.batch_size, classifier.n_classes))
            Y = torch.empty((len(loader), loader.batch_size), dtype=torch.long)
            for idx, (x,y) in enumerate(loader):
                x = x.to(device)
                #y = y.to(device)
                out[idx, :, : ] = classifier(x, no_mult=True).detach().cpu()
                Y[idx, :] = y.cpu().long()
            return out, Y




        #mult = 2**args.log_mult
        for t in range(1, args.steps+1):
            for l in range(1, N_L+1):

                def eval_mult(mult):
                    global out_class
                    loss  = eval_class_mult(out_class, mult, only_loss=True)#epoch(classifier, train_loader, with_error=False)
                    return loss

                # def eval_mult_prime(mult):  # the  derivative w.r.t M of the loss
                    # global classifier
                    # classifier.mult = mult
                    # loss  = eval_epoch_prime(classifier, train_loader)
                    # return loss

                # def eval_mult_hess(mult):  # the  derivative w.r.t M of the loss
                    # global classifier
                    # classifier.mult = mult
                    # loss  = eval_epoch_hess(classifier, train_loader)
                    # return loss

                mult = 1.
                classifier = models.classifiers.ClassifierCopyVGG(model,l,mult).to(device)
                out_class = get_output_class(classifier, train_loader)

                if args.optim_mult:
                    mult0 = 1
                    #res = minimize(eval_mult, mult0, method='BFGS')#l, options={'disp': True})
                    res = minimize_scalar(eval_mult, bounds=(0, 2**(N_L+2-l)), method='bounded')
                    print(res, file=logs)
                    mult = res.x
                else:
                    mult=2**(N_L+1-l) #res.multult0

                classifier.mult = torch.tensor(mult, device=device, dtype=dtype)
                loss, error = eval_epoch(classifier, train_loader)
                df_mult.loc[t, l] = mult


                quant.loc[pd.IndexSlice[t, (l, 'loss', 'train')]] =  loss
                quant.loc[pd.IndexSlice[t, (l, 'error', 'train')]] =  error

                loss_test, error_test = eval_epoch(classifier, test_loader)
                quant.loc[pd.IndexSlice[t, (l, 'loss', 'test')]] =  loss_test
                quant.loc[pd.IndexSlice[t, (l, 'error', 'test')]] =  error_test


                print('mult: {}, t: {}, l: {}, loss: {} (test {}) , error: {} (test {})'.format(mult, t, l, loss, loss_test, error, error_test), file=logs, flush=(l==N_L))


        quant = quant.sort_index(axis=1)
        df_mult = df_mult.sort_index(axis=1)
        quant.to_csv(os.path.join(path_output, 'quant.csv'))
        df_mult.to_csv(os.path.join(path_output, 'mult.csv'))
        df_mult.describe().to_csv(os.path.join(path_output, 'mult_describe.csv'))
        save_checkpoint()
        #process_df(quant, path_output)
            #end = err_train.max(axis=1).nonzero()[0].max() + 1  # requires _all_ the tries to be 0 to stop the computation, 1 indexed
            #if args.end_layer is not None:
            #    end = min(end, args.end_layer)

            #ones = ones[:end, :]
            #optimizer.param_groups,new_params_disc  = optimizer.param_groups[:end], optimizer.param_groups[end:]  # trim the parameters accordingly




            #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(



