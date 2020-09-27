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
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-1, help='leraning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=100, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='The ratio of neurons to keep')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')


    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:  # continuing previous computation
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            args.__dict__.update(checkpoint['args'].__dict__)
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
    output_path = os.path.join(args_model.output_root, args_model.name, args.name)
    # Logs
    log_fname = os.path.join(args_model.output_root, args_model.name, 'logs.txt')


    os.makedirs(output_path, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(output_path, 'logs_lin.txt'), 'w')
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

    #linear_classifier = models.classifiers.Linear(model, args.ntry, args.keep_ratio).to(device)
    linear_classifier = models.classifiers.Linear(model, args.ntry, args.keep_ratio).to(device)


    if 'linear_classifier' in checkpoint.keys():
        linear_classifier.load_state_dict(checkpoint['linear_classifier'])

    num_parameters = utils.num_parameters(linear_classifier)
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    print('Layer dimensions'.format(linear_classifier.neurons), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)






    print('Linear classifier: {}'.format(str(linear_classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    parameters = list(linear_classifier.parameters())

    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    optimizer = torch.optim.SGD(
        #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        parameters, lr=args.learning_rate, momentum=0.95
    )
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    names=['set', 'stat', 'layer']
    tries = np.arange(args.ntry)
    sets = ['train', 'test']
    stats = ['loss', 'err']
    layers = ['last', 'hidden']
    columns=pd.MultiIndex.from_product([sets, stats, layers, tries], names=names)
    index = pd.Index(np.arange(1, args.nepochs+1), name='epoch')
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
        cond = input.gather(2,target.view(1, -1, 1).expand(T, -1, -1)).squeeze()
        output = - cond + input.logsumexp(dim=2)
        return output

    def get_checkpoint():
        '''Get current checkpoint'''
        global model, stats, quant, args, optimizer, lr_scheduler, epoch

        checkpoint = {
                'linear_classifier': linear_classifier.state_dict(),
                'stats': stats,
            'quant': quant,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
                    }

        return checkpoint

    def save_checkpoint(fname=None, checkpoint=None):
        '''Save checkpoint to disk'''

        global path_output

        if fname is None:
            fname = os.path.join(path_output, 'checkpoint_lin.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)


    start_epoch = 0

    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']

    for epoch in tqdm(range(start_epoch, start_epoch+args.nepochs)):


        linear_classifier.train()
        loss_tot = np.zeros(args.ntry)  # for the
        loss_hidden_tot = np.zeros(linear_classifier.L)  # for the
        ones = torch.ones(args.ntry, device=device, dtype=dtype)
        ones_hidden = torch.ones(linear_classifier.L, device=device, dtype=dtype)
        for idx, (x, y) in enumerate(train_loader):


            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out, out_hidden = linear_classifier(x)  # TxBxC, LxBxC  # each output for each layer
            loss = ce_loss(out, y)  # TxB
            loss_hidden = ce_loss(out_hidden, y)
            #err = zero_one_loss(out, y)  #
            #err_tot = (idx * err_tot + err.detach().cpu().numpy()) / (idx+1)
            loss_tot = (idx * loss_tot + loss.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            loss.mean(dim=1).backward(ones)
            loss_hidden.mean(dim=1).backward(ones_hidden)
            optimizer.step()

        err_tot_test = np.zeros(args.ntry)
        err_tot = np.zeros(args.ntry)
        err_train_hidden  = np.zeros(linear_classifier.L)
        err_hidden_test  = np.zeros(linear_classifier.L)
        loss_test = np.zeros(args.ntry)
        loss_hidden_test = np.zeros(linear_classifier.L)

        with torch.no_grad():

            testloader_iter = iter(test_loader)
            for idx, (x, y)  in enumerate(train_loader, 1):

                x = x.to(device)
                y = y.to(device)
                out_train, out_train_hidden = linear_classifier(x)  # TxBxC, LxBxC  # each output for each layer
                err_tot += zero_one_loss(out_train, y).detach().cpu().numpy()
                err_train_hidden += zero_one_loss(out_train_hidden, y).detach().cpu().numpy()
                if idx-1 < len(test_loader):
                    t, w = next(testloader_iter)
                    t = t.to(device)
                    w = w.to(device)
                    out_test, out_hidden_test = linear_classifier(t)
                    loss_test = (idx * loss_test + ce_loss(out_test, w).mean(dim=1).detach().cpu().numpy())/(idx+1)
                    loss_hidden_test = (idx * loss_hidden_test + ce_loss(out_hidden_test, w).mean(dim=1).detach().cpu().numpy())/(idx+1)
                    err_tot_test += zero_one_loss(out_test, w).detach().cpu().numpy()
                    err_hidden_test += zero_one_loss(out_hidden_test, w).detach().cpu().numpy()
                else:
                    if err_tot.max() >0:
                        break


        stats['err_train'].append(err_tot/idx)
        stats['err_hidden'].append(err_train_hidden/idx)
        stats['err_hidden_test'].append(err_hidden_test/idx)
        stats['err_test'].append(err_tot_test/idx)
        stats['loss_train'].append(loss_tot)
        stats['loss_test'].append(loss_tot)
        stats['loss_hidden_train'].append(loss_hidden_tot)
        stats['loss_hidden_test'].append(loss_hidden_test)

        stats['epochs'].append(epoch)



        #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
        print('ep {}, train loss (min/max): {:g} / {:g}, err (min/max): {:g}/{:g}, progress: {}/{}'.format(
            epoch, stats['loss_train'][-1].min(), stats['loss_train'][-1].max(),
            stats['err_train'][-1].min(), stats['err_train'][-1].max(), idx, len(train_loader)),
            #stats['loss_test']['ce'][-1], stats['loss_test']['zo'][-1]),
            file=logs, flush=True)


        fig, axes = plt.subplots(2, 1, squeeze=True, sharex=True)
        axes[0].plot(stats['epochs'], stats['err_train'],  marker='o')
        axes[1].plot(stats['epochs'], stats['err_test'], marker='o')
        axes[0].set_title('Train')
        axes[1].set_title('Test')
        fig.suptitle(f'ntry={args.ntry}, layers={linear_classifier.neurons}, keep_ratio={linear_classifier.keep_ratio}')
        axes[0].set_yscale('linear')
        axes[1].set_yscale('linear')

        plt.savefig(fname=os.path.join(output_path, 'zero_one_loss.pdf'))

        fig, axes= plt.subplots(2, 1, squeeze=True, sharey=True, sharex=True)

        axes[0].plot(stats['epochs'], stats['err_hidden'], marker='o')
        axes[0].legend([f'Layer {i}' for i in range(1, 1+linear_classifier.L)])
        axes[0].set_title('Train')

        axes[0].set_ylabel('Error')
        axes[0].set_yscale('linear')

        axes[1].plot(stats['epochs'], stats['err_hidden_test'], label=[f'Layer {i}' for i in range(1, 1+linear_classifier.L)], marker='x')
        #ax.plot(stats['epochs'], stats['err_test'], label='Test')
        axes[1].legend([f'Layer {i}' for i in range(1, 1+linear_classifier.L)])
        axes[1].set_title('Test')
        axes[1].set_yscale('linear')
        axes[1].set_ylabel('Error')

        fig.suptitle(f'layers={linear_classifier.neurons}')
        for ax in axes:
            ax.label_outer()
        plt.savefig(fname=os.path.join(output_path, 'zo-layers.pdf'))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(stats['epochs'], stats['loss_train'])
        #ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        #ax.legend()
        ax.set_title('Cross-entropy loss for the tries')
        ax.set_yscale('linear')
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss.pdf'))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(stats['epochs'], stats['loss_hidden_train'])
        #ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        ax.legend([f'Layer {i}' for i in range(1, 1+linear_classifier.L)])
        ax.set_title('Cross-entropy loss for the layers')
        ax.set_yscale('linear')
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss_hidden.pdf'))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(stats['epochs'], stats['loss_hidden_test'])
        #ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        ax.legend([f'Layer {i}' for i in range(1, 1+linear_classifier.L)])
        ax.set_title('Cross-entropy loss for the layers')
        ax.set_yscale('linear')
        plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss_hidden_test.pdf'))
        #fig=plt.figure()
        #plt.plot(stats['epochs'], stats['lr'], label='lr')
        #plt.legend()
        #plt.savefig(fname=os.path.join(output_path, 'lr.pdf'))

        plt.close('all')

        if args.save_model and (epoch) % 5 == 0 or (epoch==start_epoch+args.nepochs):  # we save every 5 epochs
            checkpoint = {
                'linear_classifier': linear_classifier.state_dict(),
                'stats': stats,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
                #'seed': seed,
            }
            torch.save(checkpoint, os.path.join(output_path, 'checkpoint_lin.pth'))

        if err_tot.min() == 0:  # the data has been separated

            checkpoint = {
                'linear_classifier': linear_classifier.state_dict(),
                'stats': stats,
                'args': args,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
                #'seed': seed,
            }
            torch.save(checkpoint, os.path.join(output_path, 'checkpoint_lin.pth'))
            print('the data is separable!', file=logs)
            sys.exit(1)

