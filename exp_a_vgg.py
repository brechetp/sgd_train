import torch
import numpy as np
import pandas as pd
import os
import sys
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict, OrderedDict
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

    parser = argparse.ArgumentParser('Training an annex classifier based on original VGG model')
    parser.add_argument('--dataset', '-dat', default='mnist', type=str, help='dataset')
    parser.add_argument('--dataroot', '-droot', default='./data/', help='the root for the input data')
    parser.add_argument('--name', default='annex', type=str, help='the name of the experiment')
    parser.add_argument('--vary_name', nargs='*', default=None, help='the name of the parameter to vary in the name (appended)')
    parser.add_argument('--learning_rate', '-lr', nargs='*', type=float, default=[5e-3], help='manual learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="the weight decay for SGD (L2 pernalization)")
    parser.add_argument('--momentum', type=float, default=0.9, help="the momentum for SGD")
    parser.add_argument('--lr_update', '-lru', type=int, default=0, help='if any, the update of the learning rate')
    parser.add_argument('--lr_mode', '-lrm', default="num_param_tot", choices=["max", "hessian", "num_param_tot", "num_param_train", "manual"], help="the mode of learning rate attribution")
    parser.add_argument('--lr_step', '-lrs', type=int, default=30, help='the number of epochs for the lr scheduler')
    parser.add_argument('--lrs_gamma',  type=float, default=0.5, help='the gamma mult factor for the lr scheduler')
    parser.add_argument('--save_model', action='store_true', default=True, help='stores the model after some epochs')
    parser.add_argument('--nepochs', type=int, default=1000, help='the number of epochs to train for')
    parser.add_argument('--batch_size', '-bs', type=int, default=100, help='the dimension of the batch')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--size_max', type=int, default=None, help='maximum number of traning samples')
    parser.add_argument('--draws', type=int, default=20, help='The number of permutations to test')
    parser_remove = parser.add_mutually_exclusive_group(required=True)
    #parser_remove.add_argument('-R', '--remove', type=int, help='the number of neurons to remove at each layer')
    parser_remove.add_argument('-F', '--fraction', type=int, help='the denominator of the removed fraction of the width')
    parser.set_defaults(F=2)  # default: remove half the width
    parser_model = parser.add_mutually_exclusive_group(required=True)
    parser_model.add_argument('--model', help='path of the model to separate')
    parser_model.add_argument('--checkpoint', help='path of the previous computation checkpoint')
    parser.add_argument('--gd_mode', '-gdm', default='stochastic', choices=['full', 'stochastic'], help='whether the gradient is computed full batch or stochastically')
    parser_device = parser.add_mutually_exclusive_group()
    parser_device.add_argument('--cpu', action='store_true', dest='cpu', help='force the cpu model')
    parser_device.add_argument('--cuda', action='store_false', dest='cpu')
    parser.add_argument('--entry_layer', type=int, default=1, help='the layer ID for the tunnel entry')
    #parser.add_argument('--fract_val', type=float, default=0.10, help="the fraction of training samples to use for validation")
    parser.add_argument('--early_stoping', type=int, default=0, help='the number of epochs to early stop')
    #parser.add_argument('--end_layer', type=int, help='if set the maximum layer for which to compute the separation (forward indexing)')
    parser.set_defaults(cpu=False)



    args = parser.parse_args()

    if  len(args.learning_rate) != 2:
        args.learning_rate = 2*[args.learning_rate[0]]

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

    start_id_draw = checkpoint.get('draws', 0) + 1
    start_epoch = checkpoint.get('epochs', 0)

    if args.entry_layer == 0:
        args.draws = 1

    args_model = checkpoint_model['args']  # restore the previous arguments


    if args.vary_name is not None:
        args.name = utils.get_name(args)






    str_entry = 'entry_{}'.format(args.entry_layer)
    path_output = os.path.join(args_model.output_root, args_model.name, args.name, str_entry)
    # Logs
    #log_fname = os.path.join(args_model.output_root, args_model.name, 'logs.txt')
    #draw_idx = utils.find_draw_idx(path_output)


    os.makedirs(path_output, exist_ok=True)

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


    #model = utils.construct_FCN(archi)
    NUM_CLASSES = utils.get_num_classes(args_model.dataset)
    model,input_size  = models.pretrained.initialize_model(args_model.model, pretrained=False, freeze=True, num_classes=NUM_CLASSES)
    model.n_layers = utils.count_hidden_layers(model)
    #imresize = (256, 256)
    #imresize=(64,64)
    imresize=input_size
    train_dataset, valid_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args_model.dataset,
                                                          dataroot=args_model.dataroot,
                                                                            augment=False,
                                                                            normalize=True,
                                                             #imresize =imresize,
                                                             )
    train_loader, size_train,\
        val_loader, size_val,\
        test_loader, size_test  = utils.get_dataloader( train_dataset, valid_dataset, test_dataset, batch_size =args.batch_size, size_max=args.size_max, collate_fn=None, pin_memory=False)

    #model = models.cnn.CNN(1)

    num_classes = len(train_dataset.classes) if args.dataset != 'svhn' else 10
    imsize = next(iter(train_loader))[0].size()[1:]
    input_dim = imsize[0]*imsize[1]*imsize[2]




    #min_width = int(args.coefficient *math.sqrt(size_train)+0.5)
    #max_width = int(3*args.coefficient *math.sqrt(size_train)+0.5)
    #model = models.classifiers.FCN3(input_dim=input_dim, num_classes=num_classes, min_width=min_width, max_width=max_width)
    #archi = utils.parse_archi(log_fname)
    #model = utils.construct_FCN(archi)
    try:
        new_keys = map(lambda x:x.replace('module.', ''), checkpoint_model['model'].keys())
        checkpoint_model['model'] = OrderedDict(zip(list(new_keys), checkpoint_model['model'].values()))
        #for key in state_dict.keys():
            #new_key = key.replace('module.', '')

        model.load_state_dict(checkpoint_model['model'])
        model.requires_grad_(False)
        model.eval()
    except RuntimeError as e:
        print("Can't load mode (error {})".format(e))

    #classifier = models.classifiers.Linear(model, args.ntry, args.keep_ratio).to(device)
    #classifier = models.classifiers.ClassifierFCN(model, num_tries=args.ntry, Rs=args.remove, =args.depth_max).to(device)
    #if args.remove is not None:
    #    remove = args.remove # the number of neurons
    #else:  # fraction is not None
    fraction = 1/args.fraction

    classifier = models.classifiers.ClassifierVGGAnnex(model, F=fraction, idx_entry=args.entry_layer).to(device)




    if 'classifier' in checkpoint.keys():
        classifier.load_state_dict(checkpoint['classifier'])


    classifier.to(device)

    num_parameters = utils.num_parameters(classifier, only_require_grad=False)
    num_parameters_trainable = utils.num_parameters(classifier, only_require_grad=True)
    num_layers = 1
    num_samples_train = size_train
    num_samples_val = size_val
    num_samples_test = size_test
    print('Number of parameters: {}'.format(num_parameters), file=logs)
    print('Number of trainable parameters: {}'.format(num_parameters_trainable), file=logs)
    print('Number of training samples: {}'.format(num_samples_train), file=logs)
    print('Number of testing samples: {}'.format(size_test), file=logs)
    #print('Layer dimensions'.format(classifier.size_out), file=logs)
    print('Image dimension: {}'.format(imsize), file=logs)

    #summary(model,  [imsize, (1,)])
    #model.apply(models.cnn.init_weights)






    print('Linear classifier: {}'.format(str(classifier)), file=logs)
    #parameters = [ p for p in model.parameters() if not feature_extraction or p.requires_grad ]
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
        return  (x.argmax(dim=-1)!=targets).float().mean(dim=-1)

    #mse_loss = nn.MSELoss()
    #if args.ntry == 1 or args.entry_layer == 0:  # only one try
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    # else:
        # def ce_loss(input, target):
            # '''Batch cross entropy loss

            # input: TxBxC output of the linear model
            # target: Bx1: the target classes

            # output: TxB the loss for each try
            # '''


            # if input.ndim == 3:
                # T, B, C = input.size()
                # cond = input.gather(2,target.view(1, -1, 1).expand(T, -1, -1)).squeeze(2)  # TxBx1
            # #else:
            # #    B, C = input.size()
            # #    cond = input.gather(1, target.view(-1, 1)).squeeze()
            # output = - cond + input.logsumexp(dim=-1)
            # return output

    #optimizer = torch.optim.SGD(param_list, momentum=0.95
    def find_learning_rate(model, train_loader, alpha=1e-3, gamma=0.01, tol=0.01):
        '''Approximate the eigenvector with largest eigenvalue of the Hessian to set the learning rate as the inverse of its norm
        https://proceedings.neurips.cc/paper/1992/file/30bb3825e8f631cc6075c0f87bb4978c-Paper.pdf'''

        def normalized(X): return X / X.norm()
        #def normalized(X): return X
        psi_feat = normalized(torch.randn((utils.num_parameters(model.features),))).to(device)
        psi_class = normalized(torch.randn((utils.num_parameters(model.classifier),))).to(device)
        norm_psi_feat = psi_feat.norm()
        norm_psi_class = psi_class.norm()
        parameters_grad_feat = [p for p in model.features.parameters() if p.requires_grad]
        parameters_grad_class = [p for p in model.classifier.parameters() if p.requires_grad]

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
            g_1_feat = utils.get_grad_to_vector(parameters_grad_feat, zero=True)
            g_1_class = utils.get_grad_to_vector(parameters_grad_class, zero=True)

            # current weights of the model
            weights_feat = torch.nn.utils.parameters_to_vector(parameters_grad_feat)
            weights_class = torch.nn.utils.parameters_to_vector(parameters_grad_class)

            # perturbation on the weights
            perturbed_feat = weights_feat + alpha * normalized(psi_feat)
            perturbed_class = weights_class + alpha * normalized(psi_class)
            torch.nn.utils.vector_to_parameters(perturbed_feat, parameters_grad_feat)
            torch.nn.utils.vector_to_parameters(perturbed_class, parameters_grad_class)
            #weights_prev = utils.perturb_weights(model, alpha, psi)
            # new output with the perturbed weights
            out_class = model(x)  # TxBxC,  # each output for each layer
            loss = ce_loss(out_class, y)  # LxTxB
            loss.mean().backward()
            # record the second gradient
            g_2_feat = utils.get_grad_to_vector(parameters_grad_feat, zero=True)
            g_2_class = utils.get_grad_to_vector(parameters_grad_class, zero=True)

            # exponential average of the direction psi
            psi_feat, psi_feat_prev = (1-gamma) * psi_feat + gamma / alpha * (g_2_feat - g_1_feat), psi_feat
            norm_psi_feat, norm_prev_feat = psi_feat.norm(), norm_psi_feat
            # set the weights to previous value
            psi_class, psi_class_prev = (1-gamma) * psi_class + gamma / alpha * (g_2_class - g_1_class), psi_class
            norm_psi_class, norm_prev_class = psi_class.norm(), norm_psi_class

            torch.nn.utils.vector_to_parameters(weights_feat, parameters_grad_feat)
            torch.nn.utils.vector_to_parameters(weights_class, parameters_grad_class)

            #variation = abs(norm_psi - norm_prev) / norm_prev
            #converged = variation < tol
            converged = False  # convergence criterion was not good enough, perform on the whole dataset

            if converged:
                break

        #return 1/norm_psi_feat, 1/norm_psi_class
        return  [1/norm_psi_class]

    def get_lr(model):
        """The learning rate depending on the lr_mode parameter"""

        global args

        num_parameters = [utils.num_parameters(model.features), utils.num_parameters(model.classifier)]
        rule_of_thumb = [math.sqrt(1/n) for n in num_parameters if n>0]
        print("sqrt(1 / num_param) = {:2e}".format(*rule_of_thumb), file=logs)
        #rule_of_thumb_train = math.sqrt(classifier.n_tries/num_parameters_trainable)
        #rule_hessian = find_learning_rate(classifier, train_loader)
        if args.lr_mode == "hessian" or args.lr_mode == "max":
            hessian = find_learning_rate(model, train_loader)
            print("1 / norm(lambda_max) = {:2e}".format(*hessian), file=logs)
            if args.lr_mode == "hessian":
                lr = hessian
            elif args.lr_mode == "max":
                lr = [ min(l1, l2, l3) for l1, l2, l3 in zip(args.learning_rate, rule_of_thumb, hessian)]
        elif args.lr_mode == "num_param_tot":  # rule of thumb
            lr = rule_of_thumb
        elif args.lr_mode == "manual":
            lr = args.learning_rate
        else:
            raise NotImplementedError


        return lr

    #optimizer = torch.optim.SGD(parameters, momentum=0.95, lr=learning_rate, nesterov=True,
        #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
        #parameters, lr=args.learning_rate, momentum=0.95


    sets = ['train', 'val', 'test']
    stats = ['loss', 'err']
    #layers = np.arange(1, 1+1)#classifier.n_layers)  # the different layers, forward order
    draws = np.arange(1, start_id_draw+args.draws)  # the different tries

    names=['set', 'stat', 'draw']
    columns=pd.MultiIndex.from_product([sets, stats, draws], names=names)
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
        global model, stats, quant, args, args_model, optimizer, lr_scheduler, id_draw, epoch, stop#, params_discarded, end

        #optimizer.param_groups = optimizer.param_groups + params_discarded

        checkpoint = {
                'classifier': classifier.state_dict(),
                'stats': stats,
            'quant': quant,
                'args': args,
            'args_model': args_model,
                'optimizer': optimizer.state_dict(),
                'epochs': epoch,
            'draws': id_draw if stop else id_draw-1,
                    }

        #optimizer.param_groups = optimizer.param_groups[:end]

        return checkpoint


    def save_checkpoint(checkpoint=None, name=None, fname=None):
        '''Save checkpoint to disk'''

        global path_output, id_draw
        if name is None:
            name = "checkpoint_draw_{}".format(id_draw)

        if fname is None:
            fname = os.path.join(path_output, name + '.pth')


        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)



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


    if not args.debug:
        logs.close()

    for id_draw in range(start_id_draw, start_id_draw+args.draws):

        if not args.debug:
            logs = open(os.path.join(path_output, 'logs_draw_{}.txt'.format(id_draw)), 'w')
        else:
            logs = sys.stdout


        classifier = models.classifiers.ClassifierVGGAnnex(model, F=fraction, idx_entry=args.entry_layer).to(device)
        classifier.features.requires_grad_(False)
        #learning_rate = min(args.max_learning_rate, rule_of_thumb, find_learning_rate(classifier, train_loader))
        #learning_rate = rule_of_thumb
        #learning_rate = get_lr(classifier)
        #print('Learning rate: {}'.format(learning_rate), file=logs, flush=True)
        params_tot = []
        learning_rates = []
        params_feat = [p for p in classifier.features.parameters() if p.requires_grad]
        if params_feat:

            #learning_rates.append(math.sqrt(1/utils.num_parameters(classifier.features)))
            learning_rates.append(args.learning_rate[0])
            params_tot.append({'params': params_feat, 'lr': learning_rates[-1]})
        params_class = [p for p in classifier.classifier.parameters() if p.requires_grad]
        if params_class:
            #learning_rates.append(math.sqrt(1/utils.num_parameters(classifier.classifier)))
            learning_rates.append(args.learning_rate[1])
            params_tot.append({ 'params': params_class, 'lr': learning_rates[-1]})




        #parameters = [p for p in classifier.network.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params_tot, #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
                momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", file=logs, flush=True)
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            classifier.features = nn.DataParallel(classifier.features)
            #classifier.new_sample()
            #model.classifier = nn.DataParallel(model.classifier
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lrs_gamma)  # reduces the learning rate by half every 20 epochs

        if id_draw ==  start_id_draw:

            str_lr = "Learning rates: "
            for lr in learning_rates:
                str_lr += f'{lr:2e} '
            print(str_lr, file=logs, flush=True)
            print('Optimizer: {}'.format(optimizer), file=logs, flush=True)
            #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            # divdes by 10 after the first epoch
            #lr_lambdas = [lambda epoch: (epoch == 1) * 1  + (epoch > 1)*1 for _ in param_list]
            #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, lr_min=1e-3)

            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint.keys():
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


        DO_SANITY_CHECK = False
        stop = False
        separated = False
        epoch = (start_epoch - 1) if DO_SANITY_CHECK else start_epoch
        frozen = False
        #ones = torch.ones(args.ntry, device=device, dtype=dtype) if args.ntry > 1 else torch.ones([])
        params_discarded = []  # for the discarded parameters
        cnt = 0




        while not stop:
        #for epoch in tqdm(range(start_epoch, start_epoch+args.nepochs)):


            if epoch == start_epoch-1:
                err = 0
            else:
                classifier.train()
                #loss_hidden_tot = np.zeros(classifier.L)  # for the
                loss_train = 0 # np.zeros(args.ntry)  # for the
                err_train = 0 #np.zeros(args.ntry)
                #ones_hidden = torch.ones(classifier.L, device=device, dtype=dtype)

            for idx, (x, y) in enumerate(train_loader):


                x = x.to(device)
                y = y.to(device)
                if epoch == start_epoch -1:  # sanity check
                    out = model(x)#*.unsqueeze(0).unsqueeze(0) # 1x1xBxC
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
                    #if not frozen:  # if we have to update the weights
                    loss.mean(dim=-1).backward()
                    # loss_hidden.mean(dim=1).backward(ones_hidden)
                    optimizer.step()
                        #lr_scheduler.step()

            # if epoch == start_epoch - 1:  # check if we have null training error (sanity check)
                # print('Error: ', err, file=logs, flush=True)
                # assert err == 0
                # epoch += 1
                # continue
            loss_val, err_val = eval_epoch(classifier, val_loader)

            if epoch == start_epoch:
                loss_min = loss_val
                chkpt_min = get_checkpoint()

            epoch += 1 if not frozen else 0

            #err_min = err_train=0)#max(axis=0)  # min over tries, max over layers (all layers have to have at least one try at 0)
            #ones = torch.tensor(1. - (err_train == 0), device=device, dtype=dtype)  # mask for the individual losses


            separated =  err_train == 0
            #frozen = err_train == 0 and not frozen # will test with frozen network next time, prevent from freezing twice in a row

            #if frozen:
            #    print("Freezing the next iteration", file=logs)



            quant.loc[pd.IndexSlice[epoch, ('train', 'err', id_draw)]] =  err_train
            quant.loc[pd.IndexSlice[epoch, ('train', 'loss', id_draw)]] =  loss_train

            quant.loc[pd.IndexSlice[epoch, ('val', 'err', id_draw)]] =  err_val
            quant.loc[pd.IndexSlice[epoch, ('val', 'loss', id_draw)]] =  loss_val


            if loss_val < loss_min:
                loss_min = loss_val
                chkpt_min = get_checkpoint()
                cnt = 0
            elif loss_val > loss_min:
                cnt += 1

            stop = ((cnt > args.early_stoping > 0)
                    or epoch > start_epoch + args.nepochs
                    )
                #err_tot_test = np.zeros(args.ntry)
            #err_test = np.zeros(args.ntry)
            #loss_test = np.zeros(args.ntry)



            #end = err_train.max(axis=1).nonzero()[0].max() + 1  # requires _all_ the tries to be 0 to stop the computation, 1 indexed
            #if args.end_layer is not None:
            #    end = min(end, args.end_layer)

            #ones = ones[:end, :]
            #optimizer.param_groups,new_params_disc  = optimizer.param_groups[:end], optimizer.param_groups[end:]  # trim the parameters accordingly

            #params_discarded = new_params_disc+params_discarded

            if args.lr_update > 0 and epoch % args.lr_update == 0:
                #lr = min(args.max_learning_rate, rule_of_thumb, find_learning_rate(classifier, train_loader))
                #lr = rule_of_thumb
                lr = get_lr()
                print("Updating the learning rates ", lr, file=logs)
                for g in optimizer.param_groups:
                    g['lr'] = lr

            if args.lr_step >0:
                lr_scheduler.step()



            #print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
            print('try: {}, ep {}, loss (val): {:g} ({:g}), err (val): {:g} ({:g}) {}'.format(
                id_draw, epoch, quant.loc[epoch, ('train', 'loss', id_draw)], quant.loc[epoch, ('val', 'loss', id_draw)],
                err_train,  quant.loc[epoch, ('val', 'err', id_draw)], ' (separated)' if separated else ''),
                file=logs, flush=True)

            #end_layer = 1
            #quant_limit = quant.loc[pd.IndexSlice[:, quant.columns.get_level_values('layer').isin(range(1, end_layer+1))]]
            #fig, ax = plt.sub
            #quant_reset = quant_limit.reset_index()

            if epoch % 5 ==0 or stop:  # we save every 5 epochs

                loss_test, err_test = eval_epoch(classifier, test_loader)

                quant.loc[pd.IndexSlice[epoch, ('test', 'err', id_draw)]] =  err_test
                quant.loc[pd.IndexSlice[epoch, ('test', 'loss', id_draw)]] =  loss_test

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
                    ci=None,  # the whole spectrum of the data
                    facet_kws={
                    'sharey': 'row',
                    'sharex': True
                }
                )

                g.set(yscale='log')
                #g.set(title='ds = {}, width = {}, removed = {}, Tries = {}'.format(args_model.dataset, args_model.width, args.remove, args.ntry))
                g.fig.subplots_adjust(top=0.9, left= 0.1 )  # number of columns in the sublplot
                g.fig.suptitle('ds = {}, removed = width / {}, draw = {}, name = {}'.format(args_model.dataset, args.fraction, id_draw,  args.name))
                #g.set_axis_labels

                g.fig.tight_layout()
                plt.margins()
                plt.savefig(fname=os.path.join(path_output, 'quant_draw_{}.pdf'.format(id_draw)), bbox_inches="tight")

                plt.close('all')
                save_checkpoint()



        #logs_debug.close()

        logs.close()
        #save_checkpoint(chkpt_min, name='checkpoint_min_draw_{}'.format(id_draw))

    if separated:
        sys.exit(0)  # success
    else:
        sys.exit(1)


