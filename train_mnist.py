import torch
import numpy as np
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import re
import shutil
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
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-d', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-o', type=str, help='output root for the results')
    parser.add_argument('--name', default='debug', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--lr_step', '-lrs', type=int, help='if any, the step for the learning rate scheduler')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=400, help='the number of epochs to train for')
    parser.add_argument('--nlayers', type=int, default=3, help='the number of layers for the network')

    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--coefficient', '-c', type=float, default=1, help='The coefficient for the minimum width layer')
    parser.add_argument('--last_layer', type=int, default=None, help='the number of neurons for the last layer')
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
    parser.add_argument('--id_run', type=int, default=1, help='the ID of the run (for averaging over several runs)')
    parser.add_argument('dir', nargs='?', help='if set the path of a previously trained model')



    args = parser.parse_args()

    if args.dir:
        if not os.path.isfile(join_path(args.dir, 'logs.txt')):
            sys.exit(1)
        if args.id_run is None:
            args.id_run = 2


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    num_gpus = torch.cuda.device_count()



    if args.checkpoint is not None:  # we have some networks weights to continue
        try:
            average = args.average
            nepoch=args.nepochs
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
            args.nepochs=nepoch

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

        date = datetime.date.today().strftime('%y%m%d')
        args.output_root = f'results/{args.dataset}/{date}'

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



    path_output = os.path.join(args.output_root, args.name)
    #path_checkpoints = join_path(path_output, 'checkpoints')
    path_checkpoints=path_output

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
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                             imresize =imresize,
                                                             shuffle=args.shuffle,
                                                             )
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    min_width = int(args.coefficient *math.sqrt(size_train)+0.5)
    if args.net_shape == 'square':
        max_width = min_width
    elif args.net_shape == 'linear':
        max_width = int(3*min_width)
    else:
        raise NotImplementedError('args.net_shape={}'.format(args.net_shape))

    model = models.classifiers.FCNHelper(num_layers=args.nlayers, input_dim=input_dim, num_classes=num_classes, min_width=min_width, max_width=max_width, shape=args.net_shape, last_layer=args.last_layer)

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



    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    parameters = list(model.parameters())

    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    optimizer = torch.optim.SGD(
        #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        parameters, lr=args.learning_rate, momentum=0.95
    )
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = None
    if args.lr_step is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.9)
    last_lr = args.learning_rate

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

    stats = {
        'loss_test': defaultdict(list),
        'loss_train': defaultdict(list),
        'epochs': [],
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
        'lr': [],
    }

    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])

    start_epoch = 0
    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes

    def zero_one_loss(x, targets):
        return  (x.argmax(dim=1)!=y).float().mean()

    #mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()


    def save_checkpoint(fname, checkpoint=None):
        '''Saves a checkpoint at fname'''

        if checkpoint is None:

            checkpoint = {'model':model.state_dict(),
        'stats':stats,
        'args' : args,
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'epochs':epoch
                    }

        torch.save(checkpoint, fname)


    stop = False
    epoch = start_epoch
    previous=False
    exit_now=False
    tol = 1e-5
    checkpoint_min=None

    #for epoch in tqdm(range(start_epoch+1, start_epoch+args.nepochs+1)):
    while not stop:

        if  epoch > start_epoch+args.nepochs:  # can't separate the data
            sys.exit(1)  # failure

        model.train()
        loss_train = np.zeros(2)
        # 0: cel loss
        # 1: 0-1 loss
        # 2: mse loss
        losses = np.zeros(2)


        for idx, (x, y) in enumerate(train_loader):

            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            #age = age.to(device, dtype)
            out = model(x)#, is_man)
            #out_exp = out.exp()
            #S = out_exp.sum(dim=1, keepdim=True)
            #pred = out_exp / S


            loss = ce_loss(out, y)

            losses[0] = loss.item()
            losses[1] = zero_one_loss(out, y).item()
            #pred = model(x, is_man)
            loss.backward()
            loss_train = ((idx * loss_train) + losses) / (idx+1)
            optimizer.step()

        if epoch == start_epoch:  # first epoch
            loss_min = loss_train[0]

        epoch += 1



        exit_now = previous and loss_train[1] == 0
        previous = loss_train[1] == 0


        if loss_train[0] - loss_min < tol:  # new minimum found!

            last_min = 0  # reset the last min
            checkpoint_min = {'model':model.state_dict(),
                              'stats':stats,
                              'args' : args,
                              'optimizer':optimizer.state_dict(),
                              'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
                              'epochs':epoch
                              }
            loss_min = loss_train[0]

        stop = (last_min > 20
                or exit_now) # no improvement over 20 epochs or total of 400 epochs

        last_min += 1

        stats['loss_train']['ce'].append(loss_train[0])
        stats['loss_train']['zo'].append(loss_train[1])

        model.eval()
        loss_test = np.zeros(2)

        with torch.no_grad():

            for idx, (x, y) in enumerate(test_loader):

                x = x.to(device, dtype)
                y = y.to(device, dtype=torch.long)
                out = model(x)#, is_man)
               # out_exp = out.exp()
               # S = out_exp.sum(dim=1, keepdim=True)
               # pred = out_exp / S

                #pred = model(x, is_man)
                #loss = loss_fn(pred,y)
                losses[0] = zero_one_loss(out, y).item()
                losses[1] = ce_loss(out, y).item()

                loss_test = (idx * loss_test + losses) / (idx + 1)  # mean over all test data

        stats['loss_test']['zo'].append(loss_test[0])
        stats['loss_test']['ce'].append(loss_test[1])
        #stats_acc['loss_test']['zo'][id_run-1, epoch-1] = (loss_test[0])
        #stats_acc['loss_test']['ce'][id_run-1, epoch-1] = (loss_test[1])
        stats['epochs'].append(epoch)
        #lr_scheduler.step(loss)
        lr_str=''
        if lr_scheduler is not None:
            lr_scheduler.step()
            stats['lr'].append(lr_scheduler.get_last_lr())
            if stats['lr'][-1] != last_lr:
                lr_str = ', new lr: {}'.format(stats['lr'][-1])
            last_lr = stats['lr'][-1]



        print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g}){}'.format(
            epoch, stats['loss_train']['ce'][-1], stats['loss_train']['zo'][-1],
            stats['loss_test']['ce'][-1], stats['loss_test']['zo'][-1], lr_str),
            file=logs, flush=True)

        #means =
        #err =
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(stats['epochs'], stats['loss_train']['mse'], label='Train')
        #ax.plot(stats['epochs'], stats['loss_test']['mse'], label='Test')
        #ax.legend()
        #ax.set_title('MSE')
        #ax.set_yscale('log')
        #plt.savefig(fname=os.path.join(path_output, 'mse_loss.pdf'))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(stats['epochs'], stats['loss_train']['zo'], label='Train')
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
        plt.savefig(fname=os.path.join(path_output, 'zero_one_loss.pdf'))

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
        ax.set_yscale('log')
        plt.savefig(fname=os.path.join(path_output, 'cross_entropy_loss.pdf'))

        #fig=plt.figure()
        #plt.plot(stats['epochs'], stats['lr'], label='lr')
        #plt.legend()
        #plt.savefig(fname=os.path.join(path_output, 'lr.pdf'))

        plt.close('all')

        if args.save_model and (epoch) % 5 == 0 or (epoch==start_epoch+args.nepochs):  # we save every 5 epochs

            save_checkpoint(os.path.join(path_checkpoints, 'checkpoint-r{}.pth').format(args.id_run), checkpoint_min)

        #if stats['loss_train']['zo'][-1] == 0.:  # separation of

        #    print('Data has been separated', file=logs)
        #    save_checkpoint(os.path.join(path_checkpoints, 'checkpoint-r{}.pth'.format(args.id_run)))

        #    break



