
import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_theme()

import math
import models

import torch.optim
import torch
import argparse
import utils
import datetime


#from sklearn.linear_model import LogisticRegression

#from torchvision import models, datasets, transforms

try:
    from tqdm import tqdm
except:
    def tqdm(x): return x


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser('Training a classifier to inspect the layers')
    parser.add_argument('--dataset', '-dat', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='vgg', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='leraning rate')
    parser.add_argument('--lr_mode', '-lrm', default="max", choices=["max", "hessian", "num_param", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    #parser.add_argument('--coefficient', type=float, default=2, help='The coefficient for the minimum width layer')
    #parser.add_argument('--ntry', type=int, default=10, help='The number of permutations to test')
    parser.add_argument('--output_root', '-o', help='the root path for the outputs')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    #parser.add_argument('--keep_ratio', type=float, default=0.5, help='The ratio of neurons to keep')
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', choices=['vgg-16', 'vgg-11'], help='the type of the model to train')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.set_defaults(cpu=False)
    parser_feat = parser.add_mutually_exclusive_group()
    parser_feat.add_argument('--feature_extract', action='store_true', dest='feature_extract', help='use the pretrained model as a feature extractor')
    parser_feat.add_argument('--no-feature_extract', action='store_false', dest='feature_extract')
    parser.set_defaults(feature_extract=False)
    parser_proceed = parser.add_mutually_exclusive_group()
    parser_proceed.add_argument('--proceed', action='store_true', dest='proceed', help='proceed with the same parameters')
    parser_proceed.add_argument('--discontinue', action='store_false', dest='proceed')
    parser.set_defaults(process=True)
    parser.add_argument('--tol', type=float, default=0.15, help="the tolerance in error rate for stoping")



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')

    if args.output_root is None:
        args.output_root = utils.get_output_root(args)
        # default output directory

    if args.vary_name is not None:
        args.name = utils.get_name(args)

    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            #args = checkpoint['args']
            if args.proceed:  # if we proceed with exactly the same parameters
                args.__dict__.update(checkpoint['args'].__dict__)
            else:
                feat_ext = args.feature_extract
                nepochs = args.nepochs
                name = args.name
                root = args.output_root
                args.__dict__.update(checkpoint['args'].__dict__)
                args.feature_extract = feat_ext
                args.nepochs = nepochs
                args.name = name
                args.output_root = root

            #cont = True  # proceed the computation
        except RuntimeError:
            print('Could not load the model')


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
    NUM_CLASSES = utils.get_num_classes(args.dataset)
    log_fname = os.path.join(args.output_root, args.name, 'logs.txt')

    path_output = os.path.join(args.output_root, args.name)

    os.makedirs(path_output, exist_ok=True)

    feature_extract=args.feature_extract
    model, input_size = models.pretrained.initialize_model(args.model, pretrained=True, feature_extract=feature_extract, num_classes=NUM_CLASSES)

    model.to(device)

    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])




    if not args.debug:
        logs = open(os.path.join(path_output, 'logs.txt'), 'w')
    else:
        logs = sys.stdout

    logs_debug = open(os.path.join(path_output, 'debug.log'), 'w')
#     logs = None

    print(os.sep.join((os.path.abspath(__file__).split(os.sep)[-2:])), file=logs)  # folder + name of the script
    print('device= {}, num of gpus= {}'.format(device, num_gpus), file=logs)
    print('dtype= {}'.format(dtype), file=logs)

    for k, v in vars(args).items():
        print("%s= %s" % (k, v), file=logs, flush=True)


    #imresize = (256, 256)
    #imresize=(64,64)
    imresize=input_size
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
                                                             imresize =imresize,
                                                             )
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, test_dataset,
                                                       batch_size =args.batch_size, ss_factor=0.8,
                                                       size_max=args.size_max, collate_fn=None, pin_memory=False)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]



    #min_width = int(args.coefficient *math.sqrt(size_train)+0.5)
    #max_width = int(3*args.coefficient *math.sqrt(size_train)+0.5)
    #model = models.classifiers.FCN3(input_dim=input_dim, num_classes=num_classes, min_width=min_width, max_width=max_width)
    #archi = utils.parse_archi(log_fname)


    #Rs = [0, 0]  # the neurons to remove from L-1, L-2 ... layers of the classifier
    #linear_classifier = models.classifiers.ClassifierVGG(model,args.ntry, Rs).to(device)

    #if 'linear_classifier' in checkpoint.keys():
    #    linear_classifier.load_state_dict(checkpoint['linear_classifier'])

    num_parameters = utils.num_parameters(model)
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(linear_classifier.neurons), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)



    ce_loss = nn.CrossEntropyLoss()

    def find_learning_rate(model, train_loader, alpha=1e-3, gamma=0.01, tol=0.01):
        '''Approximate the eigenvector with largest eigenvalue of the Hessian to set the learning rate as the inverse of its norm
        https://proceedings.neurips.cc/paper/1992/file/30bb3825e8f631cc6075c0f87bb4978c-Paper.pdf'''

        def normalized(X): return X / X.norm()
        #def normalized(X): return X
        psi = normalized(torch.randn((utils.num_parameters(model),))).to(device)
        parameters_grad = [p for p in model.parameters() if p.requires_grad]
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
            lr = hessian
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

    #print('Linear classifier: {}'.format(str(linear_classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
    #parameters = list(linear_classifier.parameters())

    #optimizer = torch.optim.AdamW(
    #        parameters, lr=args.learning_rate, betas=(0.95, 0.999), weight_decay=0,
    #        )
    #optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate)

    if not feature_extract:
        optimizer = torch.optim.SGD([
            #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
                {'params': model.classifier.parameters()},
                {'params': model.features.parameters(), 'lr': 1e-5}],
                lr=learning_rate, momentum=0.95, nesterov=True)
    else:
        optimizer = torch.optim.SGD([
            #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
                {'params': model.classifier.parameters()}],
                #{'params': model.features.parameters(), 'lr': 1e-5}],
                lr=args.learning_rate, momentum=0.95, nesterov=True)

    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    print("Optimizer: {}".format(optimizer), file=logs, flush=True)
    print("LR Scheduler: {}".format(lr_scheduler), file=logs, flush=True)

    if 'optimizer' in checkpoint.keys():
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(e)


    if 'lr_scheduler' in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = 0

    names=['set', 'stat']
    #tries = np.arange(args.ntry)
    sets = ['train', 'test']
    stats = ['loss', 'err']
    #layers = ['last', 'hidden']
    columns=pd.MultiIndex.from_product([sets, stats], names=names)
    index = pd.Index(np.arange(start_epoch, args.nepochs+start_epoch), name='epoch')
    quant = pd.DataFrame(columns=columns, index=index, dtype=float)

    stats = {
        'num_parameters': num_parameters,
        'num_samples_train': num_samples_train,
    }

    if 'quant' in checkpoint.keys():
        stats.update(checkpoint['quant'])

    if 'stats' in checkpoint.keys():
        stats.update(checkpoint['stats'])

    classes = torch.arange(num_classes).view(1, -1).to(device)  # the different possible classes

    def zero_one_loss(x, targets):
        ''' x: BxC
        targets: Bx1

        returns: err of dim 0
        '''
        return  (x.argmax(dim=1)!=targets).float().mean(dim=0)

    #mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    if 'epochs' in checkpoint.keys():
        start_epoch = checkpoint['epochs']


    def get_checkpoint():

        global epoch
        global model
        global args
        global optimizer

        checkpoint = {'model':model.state_dict(),
                                #'stats':stats,
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


    stop = False
    epoch = start_epoch
    previous=False
    separated=False
    tol = 1e-5
    checkpoint_min=None
    wait=5  # in epochs, tolerance for the minimum
    err_min = 1
    cnt = 0
    frozen = False  # will freeze the update to check if data is separated

    def eval_epoch(model, dataloader):


        model.eval()
        #loss_hidden_tot = np.zeros(classifier.L)  # for the
        loss_mean = 0
        err_mean = 0
        #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):

                x = x.to(device)
                y = y.to(device)
                out_class = model(x)  # BxC,  # each output for each layer
                loss = ce_loss(out_class, y)  # LxTxB
                err = zero_one_loss(out_class, y)  # T
                err_mean = (idx * err_mean + err.detach().cpu().numpy()) / (idx+1)  # mean error
                loss_mean = (idx * loss_mean + loss.mean(dim=-1).detach().cpu().numpy()) / (idx+1)  # mean loss
                # loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
                #break


        return loss_mean, err_mean



    while not stop:


        loss_train =  0
        err_train = 0
        #loss_hidden_tot = np.zeros(args.ntry)  # for the
        #ones = torch.ones(args.ntry, device=device, dtype=dtype)

        for idx, (x, y) in enumerate(train_loader):


            x = x.to(device)
            y = y.to(device)

            for p in model.parameters():
                p.grad=None

            out = model(x)  # TxBxC, LxBxC  # each output for each layer
            loss = ce_loss(out, y).mean()  # TxB
            #loss_hidden = ce_loss(out_hidden, y)
            #err = zero_one_loss(out, y)  #
            #err_train = (idx * err_train + err.detach().cpu().numpy()) / (idx+1)

            loss_train = (idx * loss_train + loss.detach().cpu().numpy()) / (idx+1)
            err = zero_one_loss(out,y)
            err_train = (idx * err_train + err.detach().cpu().numpy() )/ (idx+1)
            #loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            #loss_hidden.mean(dim=1).backward(ones)
            if not frozen:
                loss.backward()
                optimizer.step()

        if epoch == start_epoch:  # first epoch
            loss_min = loss_train

        epoch += 1 if not frozen else 0

        quant.loc[epoch, ('train', 'err')] = err_train
        quant.loc[epoch, ('train', 'loss')] = loss_train

        separated =  frozen and err_train == 0
        frozen = err_train == 0  and not frozen # will test with frozen network next time, prevent from freezing twice in a row

        if frozen:
            print("Freezing the next iteration", file=logs, flush=True)



        #if loss_train - loss_min < tol:  # new minimum found!

        #    last_min = 0  # reset the last min
        #    checkpoint_min = get_checkpoint()
        #    loss_min = loss_train

        #stop = (#False #last_min > wait

        #last_min += 1
        #err_train_val = np.zeros(args.ntry)
        loss_val, err_val = eval_epoch(model, val_loader)
        #err_train_hidden  = np.zeros(args.ntry)
        #err_hidden_val  = np.zeros(args.ntry)
        quant.loc[epoch, ('val', 'err')] = err_val
        quant.loc[epoch, ('val', 'loss')] = loss_val

        if err_val < err_min:
            err_min = err_val
            chkpt_min = get_checkpoint()
            cnt = 0
        elif err_val > err_min:
            cnt += 1

        stop = (err_val <=args.tol
                or cnt > wait
                or separated
                or epoch > start_epoch + args.nepochs) # no improvement over wait epochs or total of 400 epochs
        #stats['err_val'].append(err_val/idx)
        #stats['loss_val'].append(loss_val)
        #stats['loss_hidden_train'].append(loss_hidden_tot)
        #stats['loss_hidden_val'].append(loss_hidden_val)

        #stats['epochs'].append(epoch)



        print('ep {}, train loss (err) {:g} ({:g}), val loss (err) {:g} ({:g})'.format(
        epoch, quant.loc[epoch, ('train', 'loss')], quant.loc[epoch, ('train', 'err')], quant.loc[epoch, ('val', 'loss')], quant.loc[epoch, ('val', 'err')]),
        #stats['loss_val']['ce'][-1], stats['loss_val']['zo'][-1]),
        file=logs, flush=True)


        utils.print_cuda_memory_usage(device, logs_debug)

        quant_reset = quant.reset_index()
        quant_plot = pd.melt(quant_reset, id_vars='epoch')
        g = sns.relplot(
            data = quant_plot,
            #col='layer',
            hue='set',
            col='stat',
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

        plt.savefig(fname=os.path.join(path_output, 'losses.pdf'))


        plt.close('all')

        if args.save_model and (epoch) % 5 == 0:  # we save every 5 epochs
            save_checkpoint()


    logs.close()
    logs_debug.close()

    save_checkpoint(chkpt_min, 'checkpoint_min')
    save_checkpoint()
    sys.exit(0)  # success

