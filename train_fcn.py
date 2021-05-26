import torch
import numpy as np
import pandas as pd
import os
import sys
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()
import math
from os.path import join as join_path
import datetime

import models
import random

import torch.optim
import torch
import argparse
import utils

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--dataset', '-dat', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dataroot', '-dr', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-oroot', type=str, help='output root for the results')
    parser.add_argument('--name', default = '', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help="the weight decay for SGD (L2 pernalization)")
    parser.add_argument('--momentum', type=float, default=0.95, help="the momentum for SGD")
    parser.add_argument('--lr_mode', '-lrm', default="manual", choices=["max", "hessian", "num_param", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--lr_step', '-lrs', type=int, default=0, help='if any, the step for the learning rate scheduler')
    parser.add_argument('--lr_gamma', '-lrg',  type=float, default=0.9, help='the gamma mult factor for the lr scheduler')
    parser.add_argument('--lr_update', '-lru', type=int, default=0, help='if any, the update of the learning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--depth', '-L', type=int, default=5, help='the number of layers for the network')
    parser_normalize = parser.add_mutually_exclusive_group()
    parser_normalize.add_argument('--normalize', action='store_true', dest='normalize',  help='normalize the input')
    parser_normalize.add_argument('--no-normalize', action='store_false', dest='normalize', help='normalize the input')
    parser.set_defaults(normalize=True)
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    net_size = parser.add_mutually_exclusive_group()
    net_size.add_argument('--coefficient', '-c', type=float, default=1, help='The coefficient for the minimum width layer')
    net_size.add_argument('--width', '-w', type=int, help='The width of the layers')
    #net_size.add_argument('--num_parameters', type=int, help='the total number of parameters')
    #parser.set_defaults(
    parser.add_argument('--last_layer', type=int, default=None, help='the number of neurons for the last layer')
    parser.add_argument('--first_layer', type=int, default=None, help='the width of the first layer')
    parser.add_argument('--net_shape', '-nt', default='square', choices=['square', 'linear'], help='how the network is constructed')
    parser.add_argument('--shuffle', default=0, type=float, help='shuffle a ratio of the target samples')
    #net_size = parser.add_mutually_exclusive_group(required=True)
    #net_size.add_argument('--num_parameters', type=int, help='the total number of parameters for the network')
    #net_size.add_argument('--width', '-w', type=int, default=3600, help='the width of the hidden layer')
    #net_size.add_argument('--width', '-w', type=int, default=3600, help='the width of the hidden layer')
    #parser.add_argument('--loss', '-l', choices=['mse', 'ce'], default='ce', help='the type of loss to use')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser.add_argument('--average', action='store_true', help='if true, saves the model to a new checkpoint and averages the different plots')
    # parser.add_argument('--id_run', type=int, default=1, help='the ID of the run (for averaging over several runs)')
    #parser.add_argument('--fract_val', type=float, default=0.10, help="the fraction of training samples to use for validation")
    #parser.add_argument('--tol', type=float, default=0, help="the tolerance in error rate for stoping")
    parser.add_argument('--early_stopping', type=int, default=0, help="the delay for the early stopping")
    # parser.add_argument('dir', nargs='?', help='if set the path of a previously trained model')



    args = parser.parse_args()

    # if args.dir:
        # if not os.path.isfile(join_path(args.dir, 'logs.txt')):
            # sys.exit(1)
        # if args.id_run is None:
            # args.id_run = 2


    dtype = torch.float
    num_gpus = torch.cuda.device_count()
    gpu_index = random.choice(range(num_gpus)) if num_gpus > 0  else 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', gpu_index)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    if args.checkpoint is not None:  # we have some networks weights to continue
        try:
            average = args.average
            nepoch=args.nepochs
            lr = args.learning_rate
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            args.nepochs=nepoch
            args.learning_rate = lr

        except RuntimeError as e:
            print('Error loading the checkpoint at {}'.format(e))

    else:
        checkpoint = dict()



    #if 'seed' in checkpoint.keys():
    #    seed = checkpoint['seed']
    #    torch.manual_seed(seed)
    #else:
    #    seed = torch.random.seed()

    #if device.type == 'cuda':
    #    torch.cuda.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)


    # Logs

    if args.output_root is None:
        # default output directory
        args.output_root = utils.get_output_root(args)

    if args.vary_name is not None:
        args.name = utils.get_name(args)




    path_output = os.path.join(args.output_root, args.name)
    #path_checkpoints = join_path(path_output, 'checkpoints')

    os.makedirs(path_output, exist_ok=True)
    #os.makedirs(path_checkpoints, exist_ok=True)









    if not args.debug:
        logs = open(os.path.join(path_output, 'logs.txt'), 'w')
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
    train_dataset, valid_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                             imresize =imresize,
                                                                            normalize=args.normalize,
                                                             shuffle=args.shuffle,
                                                             )
    print('Transform: {}'.format(train_dataset.transform), file=logs, flush=True)
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset,
                                                       valid_dataset,
                                                       test_dataset, batch_size =args.batch_size,
                                                       #ss_factor=(1-args.fract_val),
                                                       size_max=args.size_max,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       collate_fn=None)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    if args.width is None:
        min_width = int(args.coefficient * math.sqrt(size_train)+0.5)
    else:
        min_width = args.width

    if args.net_shape == 'square':
        max_width = min_width
    elif args.net_shape == 'linear':
        max_width = int(3*min_width)
    else:
        raise NotImplementedError('args.net_shape={}'.format(args.net_shape))

    model = models.classifiers.FCNHelper(num_layers=args.depth,
                                         input_dim=input_dim,
                                         num_classes=num_classes,
                                         min_width=min_width,
                                         max_width=max_width,
                                         shape=args.net_shape,
                                         first_layer=args.first_layer,
                                         last_layer=args.last_layer)

    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    print('Image size:'.format(imsize), file=logs)
    print('Model: {}'.format(str(model)), file=logs)
    model.to(device)
    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)

    if 'model' in checkpoint.keys():
        try:
            model.load_state_dict(checkpoint['model'])
            model.train()
        except RuntimeError as e:
            print("Can't load mode (error {})".format(e))

    def zero_one_loss(x, targets):
        return  (x.argmax(dim=1)!=targets).float().mean()

    #mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    def find_learning_rate(model, train_loader, alpha=1e-3, gamma=0.01, tol=0.01):
        '''Approximate the eigenvector with largest eigenvalue of the Hessian to set the learning rate as the inverse of its norm
        https://proceedings.neurips.cc/paper/1992/file/30bb3825e8f631cc6075c0f87bb4978c-Paper.pdf'''

        def normalized(X): return X / X.norm()
        #def normalized(X): return X
        psi = normalized(torch.randn((utils.num_parameters(model.main),))).to(device)
        parameters_grad = [p for p in model.main.parameters() if p.requires_grad]
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
            # record the gradient and set it to zero in the network
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
            #converged = variation < tol
            converged = False  # convergence criterion was not good enough, perform on the whole dataset

            if converged:
                break

        return 1/norm_psi

    def get_lr():
        """The learning rate depending on the lr_mode parameter"""

        global args, model, num_parameters

        rule_of_thumb = math.sqrt(1/num_parameters)
        hessian = find_learning_rate(model, train_loader)
        print("1 / norm(lambda_max) = {:2e}".format(hessian), file=logs)
        print("sqrt(1 / num_param) = {:2e}".format(rule_of_thumb), file=logs)
        #rule_of_thumb_train = math.sqrt(classifier.n_tries/num_parameters_trainable)
        #rule_hessian = find_learning_rate(classifier, train_loader)
        if args.lr_mode == "hessian":
            lr = hessian/2
        elif args.lr_mode == "num_param":  # rule of thumb
            lr = rule_of_thumb
        elif args.lr_mode == "manual":
            lr = args.learning_rate
        elif args.lr_mode == "max":
            lr = min(args.learning_rate, rule_of_thumb, hessian)

        return lr
    #learning_rate = min(args.max_learning_rate, rule_of_thumb, find_learning_rate(classifier, train_loader))
    #learning_rate = rule_of_thumb
    learning_rate = get_lr()

    #learning_rate = min(args.max_learning_rate, find_learning_rate(model, train_loader)/2)
    #rule_of_thumb = 1/math.sqrt(num_parameters)
    #learning_rate = min(rule_of_thumb, find_learning_rate(model, train_loader), args.max_learning_rate)
    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        # parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        parameters, lr=learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay,
    )

    print("Optimizer: {}".format(optimizer), file=logs, flush=True)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = None
    if args.lr_step>0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.lr_step==-1:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_gamma)
    last_lr = learning_rate

    if 'optimizer' in checkpoint.keys():

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    if 'lr_scheduler' in checkpoint.keys() and checkpoint['lr_scheduler'] is not None:

        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except RuntimeError as e:
            print("Can't load model (error {})".format(e))

    start_epoch = 0
    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    names=['set', 'stat']
    #tries = np.arange(args.ntry)
    sets = ['train', 'test']
    stats = ['loss', 'error']
    #layers = ['last', 'hidden']
    columns=pd.MultiIndex.from_product([sets, stats], names=names)
    index = pd.Index(np.arange(1, args.nepochs+start_epoch), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)


    if 'quant' in checkpoint.keys():
        quant.update(checkpoint['quant'])


    stats = {
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
        'lr': [],
    }

    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])


    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes



    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, optimizer, lr_scheduler, epoch

        checkpoint = {
            'model':model.state_dict(),
            'stats':stats,
            'quant': quant,
            'args' : args,
            'optimizer':optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epochs':epoch
                    }

        return checkpoint

    def save_checkpoint(checkpoint=None, name=None, fname=None):
        '''Save checkpoint to disk'''

        global path_output
        if name is None:
            name = "checkpoint"

        if fname is None:
            fname = os.path.join(path_output, name + '.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)


    def eval_epoch(model, dataloader):


        model.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        stats = np.zeros(2)
        stats_mean = np.zeros(2)
        #loss_mean = 0
        #err_mean = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)
                out_class = model(x)  # BxC,  # each output for each layer
                stats[0] = ce_loss(out_class, y).detach().cpu().numpy()  # LxTxB
                stats[1] = zero_one_loss(out_class, y).detach().cpu().numpy()  # T
                stats_mean = ((idx * stats_mean) + stats) / (idx+1)
                #err_mean = (idx * err_mean + err.detach().cpu().numpy()) / (idx+1)  # mean error
                #loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break


        return stats_mean


    stop = False
    epoch = start_epoch
    previous=False
    separated=False
    #tol = 1e-5
    #checkpoint_min_loss = checkpoint_min_err = None
    frozen = False
    #last_min = 0  # counter since last minimum observed
    cnt_loss = cnt_err = 0


    #for epoch in tqdm(range(start_epoch+1, start_epoch+args.nepochs+1)):
    # training loop
    while not stop:


        model.train()
        stats_train = np.zeros(2)
        # 0: cel loss
        # 1: 0-1 loss aka err
        # 2: mse loss
        stats = np.zeros(2)


        for idx, (x, y) in enumerate(train_loader):

            if args.gd_mode ==  "stochastic":
                optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            #age = age.to(device, dtype)
            out = model(x)#, is_man)
            #out_exp = out.exp()
            #S = out_exp.sum(dim=1, keepdim=True)
            #pred = out_exp / S


            loss = ce_loss(out, y)

            stats[0] = loss.item()
            stats[1] = zero_one_loss(out, y).item()
            #pred = model(x, is_man)
            stats_train = ((idx * stats_train) + stats) / (idx+1)
            if not frozen and args.gd_mode == "stochastic":
                loss.backward()
                optimizer.step()

        if args.gd_mode == "full":
            optimizer.step()
            optimizer.zero_grad()
        stats_test = eval_epoch(model, test_loader)

        if epoch == start_epoch:  # first epoch
            loss_min = stats_test[0]
            err_min = stats_test[1]

        epoch += 1 if not frozen else 0
        str_frozen= ' (frozen)' if frozen else ''



        separated = frozen and stats_train[1] == 0
        frozen = stats_train[1] == 0 and not frozen

        quant.loc[epoch, ('train', 'loss')] = stats_train[0]
        quant.loc[epoch, ('train', 'error')] = stats_train[1]


        #if stats_train[0] - loss_min < tol:  # new minimum found!

            #last_min = 0  # reset the last min
            #checkpoint_min = get_checkpoint()
            #loss_min = stats_train[0]

        #stop = (last_min >= wait
        #        or separated
        #        or epoch >= start_epoch+args.nepochs ) # no improvement over wait epochs or total of 400 epochs



        model.eval()

        quant.loc[epoch, ('test', 'loss')] = stats_test[0]
        quant.loc[epoch, ('test', 'error')] = stats_test[1]
        #stats_acc['stats_test']['zo'][id_run-1, epoch-1] = (stats_test[0])
        #stats_acc['stats_test']['ce'][id_run-1, epoch-1] = (stats_test[1])
        #lr_scheduler.step(loss)
        # if stats_test[0] < loss_min:
            # loss_min = stats_test[0]
            # chkpt_min = get_checkpoint()
            # cnt = 0
        # elif stats_test[0] > loss_min:
            # cnt += 1

        # stop = (stats_test[1] <args.tol
                # or cnt > wait
                # or separated
                # or epoch > start_epoch + args.nepochs) # no improvement over wait epochs or total of 400 epochs

        if stats_test[0] < loss_min:
            loss_min = stats_test[0]
            chkpt_min_loss = get_checkpoint()
            cnt_loss = 0
        elif stats_test[0] > loss_min:
            cnt_loss += 1

        if stats_test[1] < err_min:
            err_min = stats_test[1]
            chkpt_min_err = get_checkpoint()
            cnt_err = 0
        elif stats_test[1] > err_min:
            cnt_err += 1

        stop = (#stats_test[1] <=args.tol or
                (cnt_loss >= args.early_stopping  > 0)
                or (cnt_err >= args.early_stopping  > 0)
                or separated
                or epoch > start_epoch + args.nepochs) # no improvement over wait epochs or total of 400 epochs

        # update the learning rates
        if args.lr_update > 0 and epoch % args.lr_update == 0:
            lr = get_lr()
            print("Updating the learning rates ", lr, file=logs)
            for g in optimizer.param_groups:
                g['lr'] = lr


        print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g}) ({})'.format(
            epoch, quant.loc[epoch, ('train', 'loss')], quant.loc[epoch, ('train', 'error')],
            quant.loc[epoch, ('test', 'loss')], quant.loc[epoch, ('test', 'error')], str_frozen),
            #epoch, stats['stats_train']['ce'][-1], stats['stats_train']['zo'][-1],
            #stats['stats_test']['ce'][-1], stats['stats_test']['zo'][-1], lr_str),
            file=logs, flush=True)


        if args.lr_step>0:
            lr_scheduler.step()
        #means =
        #plt.savefig(fname=os.path.join(path_output, 'lr.pdf'))


        if stop or (epoch) % 5 == 0:  # we save every 5 epochs

            #err_train_hidden  = np.zeros(args.ntry)
            #err_hidden_val  = np.zeros(args.ntry)

            quant_reset = quant.reset_index()
            quant_plot = pd.melt(quant_reset, id_vars='epoch')
            g = sns.relplot(
                data = quant_plot,
                #col='layer',
                hue='set',
                row='stat',
                x='epoch',
                y='value',
                kind='line',
                #ci=100,  # the whole spectrum of the data
                facet_kws={
                'sharey': False,
                'sharex': True
            }
            )

            g.set(yscale='log')
            plt.savefig(fname=os.path.join(path_output, 'stats.pdf'))
            plt.close('all')

            if args.save_model:  # we save every 5 epochs
                save_checkpoint()

        #    print('Data has been separated', file=logs)
        #    break

    # at the end of the while loop
    # if separated:  #
        # save_checkpoint()
        # print("Data is separated.", file=logs)
        # sys.exit(0)  # success
    # else:
        # # save the 'best' model
        # save_checkpoint(checkpoint=checkpoint_min)
        # sys.exit(1)  # failure
    logs.close()
    #save_checkpoint(chkpt_min_loss, 'checkpoint_min_loss')
    #save_checkpoint(chkpt_min_err, 'checkpoint_min_err')
    #save_checkpoint()




