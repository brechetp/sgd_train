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


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='linear', type=str, help='the name of the experiment')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('-R', '--remove', type=int, default=100, help='the number of neurons to remove at each layer')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--depth_max', type=int, help='the maximum depth to which operate')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:  # continuing previous computation
        try:
            nepochs = args.nepochs
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            args.nepochs = nepochs
            cont = True  # continue the computation
        except RuntimeError:
            print('Could not load the model')


    else:
        checkpoint = dict()

    try:
        checkpoint_model = torch.load(args.model, map_location=device)  # checkpoint is a dictionnary with different keys
    except RuntimeError as e:
        print('Error loading the model at {}'.format(e))



    #if 'seed' in checkpoint.keys():
    #    seed = checkpoint['seed']
    #    torch.manual_seed(seed)
    #else:
    #    seed = torch.random.seed()

    #if device.type == 'cuda':
    #    torch.cuda.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)


    args_model = checkpoint_model['args']  # restore the previous arguments

    if args.vary_name is not None:
        name = ''
        for field in args.vary_name:
            if field in args:
                arg = args.__dict__[field]
                if isinstance(arg, bool):
                    dirname = f'{field}' if arg else f'no-{field}'
                else:
                    val = str(arg)
                    key = ''.join(c[0] for c in field.split('_'))
                    dirname = f'{key}-{val}'
                name = os.path.join(name, dirname)
        args.name = name if name else args.name

    path_output = os.path.join(args_model.output_root, args_model.name, args.name)
    # Logs
    log_fname = os.path.join(args_model.output_root, args_model.name, 'logs.txt')


    os.makedirs(path_output, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(path_output, 'logs_lin.txt'), 'w')
    else:
        logs = sys.stdout
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    #imresize = (256, 256)
    #imresize=(64,64)
    imresize=None
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                          dataroot=args_model.dataroot,
                                                             imresize =imresize,
                                                             )
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None, pin_memory=False)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    #min_width = int(args.coefficient *math.sqrt(size_train)+0.5)
    #max_width = int(3*args.coefficient *math.sqrt(size_train)+0.5)
    #model = models.classifiers.FCN3(input_dim=input_dim, num_classes=num_classes, min_width=min_width, max_width=max_width)
    archi = utils.parse_archi(log_fname)
    model = utils.construct_FCN(archi)
    try:
        model.load_state_dict(checkpoint_model['model'])
        model.requires_grad_(False)
        model.eval()
    except RuntimeError as e:
        print("Can't load mode (error {})".format(e))

    #classifier = models.classifiers.Linear(model, args.ntry, args.keep_ratio).to(device)
    #classifier = models.classifiers.ClassifierFCN(model, num_tries=args.ntry, Rs=args.remove, depth_max=args.depth_max).to(device)
    classifier = models.classifiers.ClassifierFCNSimple(model, num_tries=args.ntry, R=args.remove).to(device)


    if 'classifier' in checkpoint.keys():
        classifier.load_state_dict(checkpoint['classifier'])

    num_parameters = utils.num_parameters(classifier)
    num_layers = 1
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(classifier.size_out), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)






    print('Linear classifier: {}'.format(str(classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    parameters = classifier.network.parameters()
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
        ''' x: TxBxC
        targets: Bx1

        returns: err of size T
        '''
        return  (x.argmax(dim=2)!=y).float().mean(dim=1)

    #mse_loss = nn.MSELoss()
    #ce_loss_check = nn.CrossEntropyLoss(reduction='none')

    def ce_loss(input, target):
        '''Batch cross entropy loss

        input: TxBxC output of the linear model
        target: Bx1: the target classes

        output: TxB the loss for each try
        '''


        T, B, C = input.size()
        cond = input.gather(2,target.view(1, -1, 1).expand(T, -1, -1)).squeeze(2)  # TxBx1
        output = - cond + input.logsumexp(dim=-1)
        return output

    #optimizer = torch.optim.SGD(param_list, momentum=0.95
    def find_learning_rate(model, train_loader, alpha=0.01, gamma=0.01, tol=0.01):

        def normalized(X): return X / X.norm()
        psi = normalized(torch.randn((utils.num_parameters(model.network),))).to(device)
        parameters_grad = [p for p in model.network.parameters() if p.requires_grad]
        #while not converged:
        for idx, (x, y)  in enumerate(train_loader):
            model.zero_grad()
            #x, y  =next(train_loader)
            x = x.to(device)
            y = y.to(device)
            out_class = model(x)  # TxBxC,  # each output for each layer
            #out = model(x).unsqueeze(0).unsqueeze(0) # 1x1xBxC
            # cross entropy loss on samples
            loss = ce_loss(out_class, y)  # LxTxB
            loss.mean().backward()
            # record the gradient and set it to zero
            g_1 = utils.get_grad_to_vector(parameters_grad, zero=True)

            # current weights of the model
            weights = torch.nn.utils.parameters_to_vector(parameters_grad)

            # perturbation on the weights
            perturbed = weights + alpha * normalized(psi)
            torch.nn.utils.vector_to_parameters(perturbed, parameters_grad)
            #weights_prev = utils.perturb_weights(model, alpha, psi)
            # new output with the perturbed weights
            out_class = model(x)  # TxBxC,  # each output for each layer
            loss = ce_loss(out_class, y)  # LxTxB
            loss.mean().backward()
            # record the second gradient
            g_2 = utils.get_grad_to_vector(parameters_grad, zero=True)

            # exponential average of the direction psi
            psi, psi_prev = (1-gamma) * psi + gamma / alpha * (g_2 - g_1), psi
            norm_psi, norm_prev = psi.norm(), psi_prev.norm()
            # set the weights to previous value
            torch.nn.utils.vector_to_parameters(weights, parameters_grad)

            variation = abs(norm_psi - norm_prev) / norm_prev
            converged = variation < tol
            #converged = False

            if converged:
                break

        return 1/norm_psi

    learning_rate = find_learning_rate(classifier, train_loader) / 2
    print('Learning rate: {}'.format(learning_rate), file=logs, flush=True)

    optimizer = torch.optim.SGD(parameters, momentum=0, lr=learning_rate
        #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        #parameters, lr=args.learning_rate, momentum=0.95
    )

    print('Optimizer: {}'.format(optimizer), file=logs, flush=True)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # divdes by 10 after the first epoch
    #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

    if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = 0
    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    sets = ['train', 'test']
    stats = ['loss', 'err']
    layers = np.arange(1, 1+1)#classifier.n_layers)  # the different layers, forward order
    tries = np.arange(1, 1+args.ntry)  # the different tries

    names=['set', 'stat', 'layer', 'try']
    columns=pd.MultiIndex.from_product([sets, stats, layers, tries], names=names)
    index = pd.Index(np.arange(1, start_epoch+args.nepochs+1), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    quant.sort_index(axis=1, inplace=True)  # sort for quicker access

    if 'quant' in checkpoint.keys():
        quant.update(checkpoint['quant'])

    stats = {
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
    }

    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])

    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes


    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, optimizer, lr_scheduler, epoch#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                'stats': stats,
            'quant': quant,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint

    def save_checkpoint(fname=None, checkpoint=None):
        '''Save checkpoint to disk'''

        global path_output

        if fname is None:
            fname = os.path.join(path_output, 'checkpoint_lin.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)






    DO_SANITY_CHECK = False
    stop = False
    separated = False
    epoch = (start_epoch - 1) if DO_SANITY_CHECK else start_epoch
    frozen = False
    ones = torch.ones(args.ntry, device=device, dtype=dtype)
    params_discarded = []  # for the discarded parameters




    while not stop:
    #for epoch in tqdm(range(start_epoch, start_epoch+args.nepochs)):


        if epoch == start_epoch-1:
            err = 0
        else:
            classifier.train()
            #loss_hidden_tot = np.zeros(classifier.L)  # for the
            loss_train = np.zeros(args.ntry)  # for the
            err_train = np.zeros(args.ntry)
            #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        for idx, (x, y) in enumerate(train_loader):


            x = x.to(device)
            y = y.to(device)
            if epoch == start_epoch -1:  # sanity check
                out = model(x).unsqueeze(0).unsqueeze(0) # 1x1xBxC
                #loss = ce_loss(out, y).mean()  # TxB
                err += zero_one_loss(out,y).mean().detach().cpu().numpy()  # just check if the number of error is 0
            else:
                optimizer.zero_grad()
                out_class = classifier(x)  # TxBxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB
                #loss_hidden = ce_loss(out_hidden, y)
                err = zero_one_loss(out_class, y)  # T
                err_train = (idx * err_train + err.detach().cpu().numpy()) / (idx+1)
                loss_train = (idx * loss_train + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)
            # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                if not frozen:  # if we have to update the weights
                    loss.mean(dim=-1).backward(ones)
                # loss_hidden.mean(dim=1).backward(ones_hidden)
                    optimizer.step()
                    #lr_scheduler.step()

        if epoch == start_epoch - 1:  # check if we have null training error (sanity check)
            print('Error: ', err, file=logs, flush=True)
            assert err == 0
            epoch += 1
            continue

        epoch += 1 if not frozen else 0

        err_min = err_train.min(axis=0)#max(axis=0)  # min over tries, max over layers (all layers have to have at least one try at 0)
        #ones = torch.tensor(1. - (err_train == 0), device=device, dtype=dtype)  # mask for the individual losses


        separated = frozen and err_min == 0
        frozen = err_min == 0 and not frozen # will test with frozen network next time, prevent from freezing twice in a row

        if frozen:
            print("Freezing the next iteration", file=logs)

        stop = (separated
                or epoch > start_epoch + args.nepochs
                )


        quant.loc[pd.IndexSlice[epoch, ('train', 'err', range(1, 2))]] =  err_train.reshape(-1)
        quant.loc[pd.IndexSlice[epoch, ('train', 'loss', range(1, 2))]] =  loss_train.reshape(-1)

        err_tot_test = np.zeros(args.ntry)
        err_test = np.zeros(args.ntry)
        loss_test = np.zeros(args.ntry)

        with torch.no_grad():

            testloader_iter = iter(test_loader)
            for idx, (x, y)  in enumerate(test_loader, 1):

                x = x.to(device)
                y = y.to(device)
                out_test = classifier(x)  # TxBxC, LxBxC  # each output for each layer
                loss = ce_loss(out_test, y)  # LxTxB
                loss_test = (idx * loss_test + loss.mean(dim=-1).detach().cpu().numpy())/(idx+1)
                err_test += zero_one_loss(out_test, y).detach().cpu().numpy()


        quant.loc[pd.IndexSlice[epoch, ('test', 'err')]] =  (err_test/idx).reshape(-1)
        quant.loc[pd.IndexSlice[epoch, ('test', 'loss')]] =  loss_test.reshape(-1)


        #end = err_train.max(axis=1).nonzero()[0].max() + 1  # requires _all_ the tries to be 0 to stop the computation, 1 indexed
        #if args.end_layer is not None:
        #    end = min(end, args.end_layer)

        #ones = ones[:end, :]
        #optimizer.param_groups,new_params_disc  = optimizer.param_groups[:end], optimizer.param_groups[end:]  # trim the parameters accordingly

        #params_discarded = new_params_disc+params_discarded



        #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
        print('ep {}, train loss (min/max): {:g} / {:g}, err (min/max): {:g}/{:g} {}'.format(
            epoch, quant.loc[epoch, ('train', 'loss')].min(), quant.loc[epoch, ('train', 'loss')].max(),
            err_min, quant.loc[epoch, ('train', 'err')].max(), ' (frozen)' if frozen else ''),
            file=logs, flush=True)

        #end_layer = 1
        #quant_limit = quant.loc[pd.IndexSlice[:, quant.columns.get_level_values('layer').isin(range(1, end_layer+1))]]
        #fig, ax = plt.sub
        #quant_reset = quant_limit.reset_index()
        quant_reset = quant.reset_index()
        quant_plot = pd.melt(quant_reset, id_vars='epoch')
        g = sns.relplot(
            data = quant_plot,
            col='layer',
            hue='set',
            row='stat',
            x='epoch',
            y='value',
            kind='line',
            ci=100,  # the whole spectrum of the data
            facet_kws={
            'sharey': 'row',
            'sharex': True
        }
        )

        g.set(yscale='log')
        #g.set(title='ds = {}, width = {}, removed = {}, Tries = {}'.format(args_model.dataset, args_model.width, args.remove, args.ntry))
        g.fig.subplots_adjust(top=0.9, left=1/g.axes.shape[1] * 0.1 )  # number of columns in the sublplot
        g.fig.suptitle('ds = {}, width = {}, removed = {}, Tries = {}, name = {}'.format(args_model.dataset, args_model.width, args.remove, args.ntry, args.name))
        #g.set_axis_labels

        plt.savefig(fname=os.path.join(path_output, 'losses.pdf'))

        plt.close('all')

        if args.save_model and (epoch) % 5 == 0:  # we save every 5 epochs
            save_checkpoint()

        if stop:
            save_checkpoint()
            if separated:
                print("Data is separated.", file=logs)
                sys.exit(0)  # success
            else:
                print("Data is NOT separated.", file=logs)
                sys.exit(1)  # failure



