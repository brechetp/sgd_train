import torch
import numpy as np
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
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

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--output_root', '-oroot', type=str, default='./results/mnist/200811/', help='output root for the results')
    parser.add_argument('--name', default='baseline', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='leraning rate')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=200, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--coefficient', type=float, default=2, help='The coefficient for the minimum width layer')
    parser.add_argument('--model', required=True, help='path of the previous computation checkpoint')
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
    # Logs
    log_fname = os.path.join(args_model.output_root, args_model.name, 'logs.txt')

    output_path = os.path.join(args_model.output_root, args_model.name, 'linear')

    os.makedirs(output_path, exist_ok=True)


    if not args.debug:
        logs = open(os.path.join(output_path, 'logs.txt'), 'w')
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
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset, batch_size =args.batch_size, ss_factor=1, size_max=args.size_max, collate_fn=None, pin_memory=(device.type=='cuda'))

    #model = models.cnn.CNN(1)

    clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)

    num_classes = 10
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

    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    print('Input dimension: '.format(input_dim), file=logs)
    print('Model: {}'.format(str(model)), file=logs)

    N = model.main[-1].in_features
    P  = N//2
    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)




    linear_classifier = models.classifiers.Linear(model)

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

    try_total = 10
    nbatchs = len(train_loader)  # the number of batches

    stats = {
        'err_train': np.array((try_total,nbatches)),
        'loss_train': np.array((try_total,nbatches)),
        'epochs': [],
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
        'lr': [],
    }


    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes

    def zero_one_loss(x, targets):
        return  (x.argmax(dim=1)!=y).float().mean()

    #mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()


    for try_idx in tqdm(range(0, try_total)):  # we try different times to find a sub sample

        linear_classifier.reset()  # resets the quantities
        random_perm =torch.randperm(N)[:P]  # the random choice
        optimizer = torch.optim.SGD(  # for now no other means to  do
            #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
            parameters, lr=args.learning_rate, momentum=0.95
        )
        start_epoch = 0

        for epoch in tqdm(range(0, 20)):


            linear_classifier.train()
            loss_train = np.zeros(2)
            # 0: cel loss
            # 1: 0-1 loss
            # 2: mse loss
            losses = np.zeros(2)





            optimizer.zero_grad()
            err_tot = loss_tot = 0
            for idx, (x, y) in enumerate(train_loader):

                #x = x.to(device)
                #y = y.to(device)
                X = model.main[:-2](x.view(x.size(0), -1))[:, random_perm]
                clf.partial_fit(X, y)

                #loss = ce_loss(out, y)
                err = 1-clf.score(X, y)

                err_tot = (idx * err_tot + err.item()) / (idx+1)
                #loss_tot = (idx * loss_tot + loss.item()) / (idx+1)
                #loss.backward()

            optimizer.step()
            stats['err_train'][try_idx, idx] = err_tot
            stats['loss_train'][try_idx, idx] = loss_tot
            stats['epochs'].append(epoch)



            #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
            print('ep {}, train loss (err) {:g} ({:g})'.format(
                epoch, stats['loss_train'][try_idx, idx], stats['err_train'][try_idx, idx]),
                #stats['loss_test']['ce'][-1], stats['loss_test']['zo'][-1]),
                file=logs, flush=True)


            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(stats['epochs'], stats['err_train'], label='Err')
            ax.legend()
            ax.set_title('Classification error')
            ax.set_yscale('log')
            plt.savefig(fname=os.path.join(output_path, 'zero_one_loss.pdf'))

            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.plot(stats['epochs'], stats['loss_train']['ce'], label='Train')
            #ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
            #ax.legend()
            #ax.set_title('Cross-entropy loss')
            #ax.set_yscale('log')
            #plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss.pdf'))

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

            if err_tot == 0:
                break  # success

        if err_tot > 0:
            # could not linearly separate the clases
            # train the model again and increasing the constant
            print('({}/{}) The data could not be separable err={:g}, trying a other random sample'.format(try_idx, try_total, err_tot))
            linear_classifier.reset()

        else:

            print('The data is separable!')
            break

    if err_tot > 0:
        # at the end of the for loop

        pass #retrain the model with larger c

