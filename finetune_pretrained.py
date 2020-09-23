
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
    parser.add_argument('--name', default='vgg-16', type=str, help='the name of the experiment')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='leraning rate')
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
    parser.set_defaults(feature_extract=True)



    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    #device = torch.device('cpu')

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

    dtype = torch.float
    num_gpus = torch.cuda.device_count()

    if args.checkpoint is not None:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            #args = checkpoint['args']
            args.__dict__.update(checkpoint['args'].__dict__)
            cont = True  # continue the computation
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

    output_path = os.path.join(args.output_root, args.name)

    os.makedirs(output_path, exist_ok=True)

    feature_extract=args.feature_extract
    model, input_size = models.pretrained.initialize_model(args.model, pretrained=True, feature_extract=feature_extract, num_classes=NUM_CLASSES)

    model.to(device)

    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])




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
    imresize=input_size
    train_dataset, test_dataset, num_chs = utils.get_dataset(dataset=args.dataset,
                                                          dataroot=args.dataroot,
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
                lr=args.learning_rate, momentum=0.95)
    else:
        optimizer = torch.optim.SGD([
            #parameters, lr=args.learning_rate, momentum=(args.gd_mode=='full') * 0 + (args.gd_mode =='stochastic')*0.95
                {'params': model.classifier.parameters()}],
                #{'params': model.features.parameters(), 'lr': 1e-5}],
                lr=args.learning_rate, momentum=0.95)

    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])

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
        return  (x.argmax(dim=1)!=y).float().mean(dim=0)

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
                                'stats':stats,
                                'args' : args,
                                'optimizer':optimizer.state_dict(),
                                'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                'epochs':epoch
                                }
        return checkpoint



    stop = False
    epoch = start_epoch
    previous=False
    separated=False
    tol = 1e-5
    checkpoint_min=None
    wait=100  # in epochs, tolerance for the minimum
    frozen = False  # will freeze the update to check if data is separated


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
            err_train += zero_one_loss(out,y).detach().cpu().numpy()
            #loss_hidden_tot = (idx * loss_hidden_tot + loss_hidden.mean(dim=1).detach().cpu().numpy()) / (idx+1)
            loss.backward()
            #loss_hidden.mean(dim=1).backward(ones)
            if not frozen:
                loss.backward()
                optimizer.step()

        if epoch == start_epoch:  # first epoch
            loss_min = loss_train

        epoch += 1 if not frozen else 0

        quant.loc[epoch, ('train', 'err')] = err_train/idx
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
        stop = (separated
                or epoch > start_epoch + args.nepochs) # no improvement over wait epochs or total of 400 epochs

        #last_min += 1
        #err_train_test = np.zeros(args.ntry)
        err_test = 0
        #err_train_hidden  = np.zeros(args.ntry)
        #err_hidden_test  = np.zeros(args.ntry)
        loss_test = 0
        #loss_hidden_test = np.zeros(args.ntry)

        model.eval()
        with torch.no_grad():

            #testloader_iter = iter(test_loader)
            for idx, (x, y)  in enumerate(test_loader, 1):

                x = x.to(device)
                y = y.to(device)
                #out_train, out_train_hidden = linear_classifier(x)  # TxBxC, LxBxC  # each output for each layer
                out_test = model(x)
                err_test += zero_one_loss(out_test, y).detach().cpu().numpy()
                loss_test = (idx * loss_test + ce_loss(out_test, y).detach().cpu().numpy())/(idx+1)
                #err_train_hidden += zero_one_loss(out_train_hidden, y).detach().cpu().numpy()
                #if idx-1 < len(test_loader):
                #    t, w = next(testloader_iter)
                #    t = t.to(device)
                #    w = w.to(device)
                #    out_test, out_hidden_test = linear_classifier(t)
                #    loss_test = (idx * loss_test + ce_loss(out_test, w).mean(dim=1).detach().cpu().numpy())/(idx+1)
                #    loss_hidden_test = (idx * loss_hidden_test + ce_loss(out_hidden_test, w).mean(dim=1).detach().cpu().numpy())/(idx+1)
                #    err_train_test += zero_one_loss(out_test, w).detach().cpu().numpy()
                #    err_hidden_test += zero_one_loss(out_hidden_test, w).detach().cpu().numpy()


        #stats['err_hidden'].append(err_train_hidden/idx)
        #stats['err_hidden_test'].append(err_hidden_test/idx)
        quant.loc[epoch, ('test', 'err')] = err_test / idx
        quant.loc[epoch, ('test', 'loss')] = loss_test
        #stats['err_test'].append(err_test/idx)
        #stats['loss_test'].append(loss_test)
        #stats['loss_hidden_train'].append(loss_hidden_tot)
        #stats['loss_hidden_test'].append(loss_hidden_test)

        #stats['epochs'].append(epoch)



        print('ep {}, train loss (err) {:g} ({:g}), test loss (err) {:g} ({:g})'.format(
        epoch, quant.loc[epoch, ('train', 'loss')], quant.loc[epoch, ('train', 'err')], quant.loc[epoch, ('test', 'loss')], quant.loc[epoch, ('test', 'err')]),
        #stats['loss_test']['ce'][-1], stats['loss_test']['zo'][-1]),
        file=logs, flush=True)

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

        plt.savefig(fname=os.path.join(output_path, 'losses.pdf'))

        plt.close('all')

        #fig, axes = plt.subplots(2, 1, squeeze=True, sharex=True)
        #axes[0].plot(stats['epochs'], stats['err_train'],  marker='o')
        #axes[1].plot(stats['epochs'], stats['err_test'], marker='o')
        #axes[0].set_title('Train')
        #axes[1].set_title('Test')
        ##fig.suptitle(f'ntry={args.ntry}, remove={linear_classifier.Rs}')
        #fig.suptitle(f'ds={args.dataset}')
        #axes[0].set_yscale('log')
        #axes[1].set_yscale('log')

        #plt.savefig(fname=os.path.join(output_path, 'zero_one_loss.pdf'))

       # fig, axes= plt.subplots(2, 1, squeeze=True, sharey=True, sharex=True)

        #axes[0].plot(stats['epochs'], stats['err_hidden'], marker='o')
        #axes[0].legend([f'Layer {i}' for i in range(1, 1+args.ntry)])
    #axes[0].set_title('Train')

       # axes[0].set_ylabel('Error')
       # axes[0].set_yscale('linear')

       # axes[1].plot(stats['epochs'], stats['err_hidden_test'])
        #ax.plot(stats['epochs'], stats['err_test'], label='Test')
       # axes[1].legend('test')
       # axes[1].set_title('Test')
       # axes[1].set_yscale('linear')
       # axes[1].set_ylabel('Error')

        #fig.suptitle(f'Rs={linear_classifier.Rs}')
        #for ax in axes:
        #    ax.label_outer()
        #plt.savefig(fname=os.path.join(output_path, 'zo-layers.pdf'))

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(stats['epochs'], stats['loss_train'])
        ##ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        ##ax.legend()
        #ax.set_title('Cross-entropy loss for the tries')
        #ax.set_yscale('linear')
        #plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss.pdf'))

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(stats['epochs'], stats['loss_hidden_train'])
        ##ax.plot(stats['epochs'], stats['loss_test']['ce'], label='Test')
        ##ax.legend([f'Layer {i}' for i in range(1, 1+args.ntry)])
        #ax.set_title('Cross-entropy loss for the layers')
        #ax.set_yscale('linear')
        #plt.savefig(fname=os.path.join(output_path, 'cross_entropy_loss_hidden.pdf'))

        #fig=plt.figure()
        #plt.plot(stats['epochs'], stats['lr'], label='lr')
        #plt.legend()
        #plt.savefig(fname=os.path.join(output_path, 'lr.pdf'))

        plt.close('all')

        if args.save_model and (epoch) % 5 == 0:  # we save every 5 epochs
            checkpoint = get_checkpoint()
            torch.save(checkpoint, os.path.join(output_path, 'checkpoint.pth'))


        if stop:
            if separated:
                checkpoint = get_checkpoint()
                torch.save(checkpoint, os.path.join(output_path, 'checkpoint.pth'))
                print("Data is separated.", file=logs)
                sys.exit(0)  # success
            else:
                #torch.save(checkpoint_min, os.path.join(output_path, 'checkpoint.pth'))
                sys.exit(1)  # failure

