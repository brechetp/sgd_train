import torchvision
import torch
import pandas as pd
#from datasets.rsna import RSNADataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, ImageNet
import torchvision.datasets
import torch.nn as nn
from collections import OrderedDict
from subprocess import Popen, PIPE
import re
import PIL
import math
import os
import numpy as np
import sys
import argparse
from os.path import join as join_path
import models
import datetime

def pad_collate_fn(batch):
    '''collate function for images of different sizes'''
    data, man, age = zip(*batch)
    sizes = torch.tensor([item[0].size()[1:] for item in batch])
    bounds = torch.max(sizes, dim=0)[0]
    margins = bounds.view(1, -1) - sizes
    pad_sizes = [(margin_w//2, margin_w - margin_w//2, margin_h//2, margin_h - margin_h//2) for (margin_h, margin_w) in margins]
    data = torch.cat([nn.functional.pad(item[0].unsqueeze(0), pad_s) for item, pad_s in zip(batch, pad_sizes)])

    return (data, torch.tensor(man, dtype=int), torch.tensor(age, dtype=torch.float))

def get_output_root(args):

    date = datetime.date.today().strftime('%y%m%d')
    output_root = f'results/{args.dataset}/{date}'
    return output_root

def get_name(args):
    name = ''
    for field in args.vary_name:
        if field in args:
            arg = args.__dict__[field]
            if isinstance(arg, bool):
                dirname = f'{field}' if arg else f'no-{field}'
            else:
                val = str(arg)
                if field == 'depth':
                    key = 'L'
                else:
                    key = ''.join(c[0] for c in field.split('_'))
                dirname = f'{key}-{val}'
            name = os.path.join(name, dirname)
    name = os.path.join(name, args.name)
    if name  == "":
        name = "debug"
    return name

def get_num_classes(dataset):

    NCLASSES = {'mnist': 10,
                'cifar10': 10,
                'svhn': 10,
                'cifar100': 100,
                'imagenet': 1000,
                }
    return NCLASSES[dataset.lower()]





def get_dataset(dataset='mnist', dataroot='data/', imresize=None, augment=False,
                normalize=False, tfm=None, shuffle=0):


    num_chs = 1 if dataset.lower() in [ 'rsna', 'mnist' ] else 3
    valid_dataset = None

    if tfm is None:
        transform_lst = []


        if imresize is not None:
            transform_lst.append(transforms.Resize(imresize))

        transform_lst.append(transforms.ToTensor())

        if normalize:
            transform_lst.append(transforms.Normalize(num_chs*(0.5,), num_chs*(0.5,)))

        transform = transforms.Compose(transform_lst)
    else:
        transform=tfm

    if dataset.lower() == 'rsna':

        raise NotImplementedError


    elif dataset.lower() == 'imagenet':

        train_dataset = ImageNet(dataroot, train=True, transform=transform, download=True)
        test_dataset = ImageNet(dataroot, train=False, transform=transform, download=True)

    elif dataset.lower() == 'cifar10':

        if tfm is None:
            normalize_cifar10 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform = transform
            if normalize:  # overwrites the normalize
                transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar10,
            ])
        # if augment:
            # transform_train=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                # transforms.ToTensor(),
                # normalize,
            # ])
        # else:
            # transform_train = transform
        train_dataset = CIFAR10(dataroot, train=True, transform=transform, download=True)


        valid_dataset = CIFAR10(dataroot, train=True, transform=transform)
        test_dataset = CIFAR10(dataroot, train=False, transform=transform, download=True)
        #train_dataset  = RSNADataset(dataroot, transform)
        #test_dataset = RSNADataset(dataroot, transform, train=False)

    elif dataset.lower() == 'cifar100':

        train_dataset = CIFAR100(dataroot, train=True, transform=transform, download=True)
        test_dataset = CIFAR100(dataroot, train=False, transform=transform, download=True)

    elif dataset.lower() == 'mnist':

        train_dataset = MNIST(dataroot, train=True, transform=transform, download=True)
        valid_dataset = train_dataset
        test_dataset = MNIST(dataroot, train=False, transform=transform, download=True)


    elif dataset.lower() == 'svhn':

        dataroot = os.path.join(dataroot, 'svhn')
        train_dataset = SVHN(dataroot, split='train', transform=transform, download=True)
        test_dataset = SVHN(dataroot, split='test',  transform=transform, download=True)

    return train_dataset, valid_dataset, test_dataset, num_chs



def get_dataloader(train_dataset, valid_dataset, test_dataset, batch_size,
                  size_max=None, collate_fn=None, num_workers=1,
                   pin_memory=True):
    '''ss_factor: for the valid set'''

    #indices = torch.randperm(len(train_dataset))
    indices = torch.arange(len(train_dataset))
    train_size = round(len(train_dataset))
    val_size = 0 #len(train_dataset) - train_size


    if size_max is not None:
        train_size = min(size_max, train_size)

    train_idx = indices[:train_size]
    # val_idx = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    val_sampler = None


    if test_dataset is None:
        test_size = None
    else:
        test_size = len(test_dataset)

    #  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)  # the first

    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler,
                            collate_fn=collate_fn,
                              num_workers=num_workers,
                            pin_memory=pin_memory) if val_size >0 else None # the rest of the indices

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                              num_workers=num_workers,
                             pin_memory=pin_memory) if not(test_dataset is None) else None

    return train_loader,train_size, val_loader, val_size, test_loader, test_size


def construct_mlp_layers(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], kwargs_linear={}, tanh=False, out_layer=True, batchnorm=False, batchnorm_in=False):
    '''Constructs layers  with each layer being of sizes[i]
    with specified init and end size in sizes[0] sizes[-1]

    Args:
        sizes (list): the list [nin, *layers, nout] of dimensions for the layers
        layer (nn.): the linear transformation for the layers (default: nn.Linear)
        fct_act: the non linear activation function (default: nn.ReLU)
        tanh (bool): if true will append a tanh activation layer as the last layer (to map output to -1, 1)
        batchnorm (bool): if true, use a batch norm

    return: ordered dict of layers
'''

    idx = 0
    size = sizes[0]
    layers = []

    if isinstance(size, tuple):
        N, R = size
        size =N-R  # the kept units
        layers.append((
            '{}{}-{}/{}'.format(models.classifiers.RandomSampler.__name__, idx, size, N),
                models.classifiers.RandomSampler(N, R)
                ))

    norm_layer = nn.BatchNorm1d

    for idx, new_size in enumerate(sizes[1:-1] if out_layer else sizes[1:], 1):  # for all layers specified in sizes
        # switch the sizes
        prev_size, size = size, new_size
        # adds the layer
        layers.append((
                '{}{}-{}-{}'.format(nn.Linear.__name__, idx, prev_size, size),
                nn.Linear(prev_size, size, *args_linear, **kwargs_linear)
                ))
        if batchnorm:
            if idx == 1 and not out_layer and not batchnorm_in:  # for discriminator network, no normalization at the beginning
                pass
            #  elif out_layer and not tanh and idx == len(sizes) - 2:  # for generator, no batchnorm at the last layer (following DCGAN)
                #  pass
            else:
                layers.append((
                    '{}{}-{}'.format(norm_layer.__name__, idx, size),
                    norm_layer(size),
                    ))

        # adds the non linear activation function
        layers.append((
                '{}{}-{}'.format(fct_act.__name__, idx, size),
                fct_act(*args_act, **kwargs_act)
                ))

    # at  the end of the loop, we still have the last layer to add, but without
    # activation
    if out_layer:
        layers.append((
            '{}{}-{}-{}'.format(nn.Linear.__name__, idx+1, size, sizes[-1]),
            nn.Linear(size, sizes[-1], *args_linear, **kwargs_linear)
            ))

        if tanh:
            layers.append((
                'tanh', nn.Tanh()
                ))

    return OrderedDict(layers)

# def construct_cnn_layers(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], kwargs_linear={}, tanh=False, out_layer=True, batchnorm=False, batchnorm_in=False):
    # '''Constructs layers  with each layer being of sizes[i]
    # with specified init and end size in sizes[0] sizes[-1]

    # Args:
        # sizes (list): the list [nin, *layers, nout] of dimensions for the layers
        # layer (nn.): the linear transformation for the layers (default: nn.Linear)
        # fct_act: the non linear activation function (default: nn.ReLU)
        # tanh (bool): if true will append a tanh activation layer as the last layer (to map output to -1, 1)
        # batchnorm (bool): if true, use a batch norm

    # return: ordered dict of layers
# '''

    # idx = 0
    # size = sizes[0]
    # layers = []

    # if isinstance(size, tuple):
        # N, R = size
        # size =N-R  # the kept units
        # layers.append((
            # '{}{}-{}/{}'.format(models.classifiers.RandomSampler.__name__, idx, size, N),
                # models.classifiers.RandomSampler(N, R)
                # ))

    # norm_layer = nn.BatchNorm1d

    # for idx, new_size in enumerate(sizes[1:-1] if out_layer else sizes[1:], 1):  # for all layers specified in sizes
        # # switch the sizes
        # prev_size, size = size, new_size
        # # adds the layer
        # layers.append(
                # nn.Conv2d(prev_size, size, *args_linear, **kwargs_linear)
                # )
        # layers.append(
            # nn.Conv2d(
        # #if batchnorm:
            # #if idx == 1 and not out_layer and not batchnorm_in:  # for discriminator network, no normalization at the beginning
            # #    pass
            # #  elif out_layer and not tanh and idx == len(sizes) - 2:  # for generator, no batchnorm at the last layer (following DCGAN)
                # #  pass
            # #else:
                # # layers.append((
                    # # '{}{}-{}'.format(norm_layer.__name__, idx, size),
                    # # norm_layer(size),
                    # # ))

        # # adds the non linear activation function
        # layers.append((
                # '{}{}-{}'.format(fct_act.__name__, idx, size),
                # fct_act(*args_act, **kwargs_act)
                # ))

    # # at  the end of the loop, we still have the last layer to add, but without
    # # activation
    # if out_layer:
        # layers.append((
            # '{}{}-{}-{}'.format(nn.Linear.__name__, idx+1, size, sizes[-1]),
            # nn.Linear(size, sizes[-1], *args_linear, **kwargs_linear)
            # ))

        # if tanh:
            # layers.append((
                # 'tanh', nn.Tanh()
                # ))

    # return OrderedDict(layers)

def construct_mmlp_layers(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], num_tries=10):
    '''Constructs multilinear layers  with each layer being of sizes[i]
    with specified init and end size in sizes[0] sizes[-1]
    The first layer is a LinearParallelMasked, the subsequent ones are MultiLinear

    The first size should be the total number of neurons N and the ones that are remove R in a tuple (N, R)

    Args:
        sizes (list): the list [(nin, Rin), *layers, nout] of dimensions for the layers
        fct_act: the non linear activation function (default: nn.ReLU)

    return: ordered dict of layers
'''

    idx = 0
    (N, R) = sizes[0]  # total vs removed number of neurons for the mask
    layers = []
    norm_layer = nn.BatchNorm1d
    if fct_act is nn.ReLU and args_act == []:
        args_act = [ True ]  # inplace
    MultiLinear = models.classifiers.MultiLinear
    N, R = sizes[0]
    size =N-R  # the kept units
    layers.append((
        '{}{}-{}/{}'.format(models.classifiers.RandomSamplerParallel.__name__, idx, size, N),
            models.classifiers.RandomSamplerParallel(N, R, num_tries)
            ))
    idx = 0

    for idx, new_size in enumerate(sizes[1:], 1):  # for all layers specified in sizes
        # switch the sizes

        prev_size, size = size, new_size
        layers.append((
                '{}{}-{}-{}'.format(MultiLinear.__name__, idx, prev_size, size),
                MultiLinear(prev_size, size, num_tries=num_tries),
                ))

        if idx < len(sizes)-1:  # no activation for the output layer
            layers.append((
                    '{}{}-{}'.format(fct_act.__name__, idx, prev_size),
                    fct_act(*args_act, **kwargs_act)
                    ))
    # adds the non linear activation function

    # at the end, appends the out layer without activation function

    return OrderedDict(layers)

def construct_mlp_net(sizes, fct_act=nn.ReLU, args_act=[], kwargs_act={}, args_linear=[], kwargs_linear={}, tanh=False, out_layer=True, batchnorm=False, batchnorm_in=False):
    layers = construct_mlp_layers(sizes, fct_act, args_act, kwargs_act, args_linear, kwargs_linear, tanh, out_layer, batchnorm, batchnorm_in)
    return nn.Sequential(layers)

def construct_mmlp_net(sizes, num_tries=10, fct_act=nn.ReLU):
    layers = construct_mmlp_layers(sizes, fct_act, num_tries=num_tries)
    return nn.Sequential(layers)

def num_parameters(model, only_require_grad=True):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in model.parameters() if not only_require_grad or p.requires_grad)

def get_grad_to_vector(parameters, zero=False):
    grad = []
    for p in parameters:
        if not p.requires_grad:
            continue
        grad.append(p.grad.view(-1))
        if zero:
            p.grad = None
    return torch.cat(grad)



def get_norm_weights(model):

    norm_squared = 0.
    N = 0  # the total number of parameters
    for p in model.parameters():
        norm_squared += p.pow(2).sum()
        N += p.numel()

    return (norm_squared/N).sqrt()

def parse_transform(fname, *args):
    '''Returns the transform if any'''

    process = Popen(['grep', 'Resize', fname], stdout=PIPE, stderr=PIPE)
    lines = process.communicate()[0].decode('utf-8').strip().splitlines()
    size = None
    transform_lst = []
    transform = None
    depth = 0
    TRANSFORMS = {
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip,
        'RandomCrop': transforms.RandomCrop,
        'RandomAffine' : transforms.RandomAffine,
        'Resize': transforms.Resize,
        'ToTensor': transforms.ToTensor,
        'Normalize': transforms.Normalize,
    }
    with open(fname, 'r') as _f:
        for line in _f:
            if depth == 0:
                if line.find('Transform: Compose') != -1:
                    depth += 1
            elif depth == 1:
                depth += line.count("(")
                depth -= line.count(")")
                if depth == 0:
                    break
                par_pos = line.find('(')
                tfm_name = line.split('(')[0].strip()
                tfm_args = line[par_pos+1:].strip().rstrip(')') # leave trailing )
                args, kwargs = parse_layer_args(tfm_args)
                transform_lst.append(TRANSFORMS[tfm_name](*args, **kwargs))

    if transform_lst:
        transform = transforms.Compose(transform_lst)

    return transform

def to_latex(dirname, quant, table_format, key_err="err", is_vgg=False) :

    if len(quant.columns.names) == 3:
        quant_describe = quant.groupby(level=["stat", "set"], axis=1, group_keys=False).describe()
        N_L = len(quant.columns.unique(level="layer")) # number of layers

        split = N_L>=10
        # if table_format == "wide":
        table = quant_describe[["mean", "std", "min"]].transpose()
        if is_vgg:
            col_names=(N_L-10)*["0"] + ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "fc1", "fc2"]
            table.columns.set_levels(col_names, level="layer", inplace=True)
        for stat in ["loss", key_err]:
            tab = table[stat][["train", "test"]]
            if split:
                for dset in ["train", "test"]:
                    tab = table[stat][dset]
                    if table_format == "long":
                        tab = tab.transpose()
                    tab.to_latex(os.path.join(dirname, f'table_{stat}_{dset}.tex'), float_format="%.2f", na_rep='--')
            else:
                if table_format == "long":
                    tab = tab.transpose()
                tab.to_latex(os.path.join(dirname, f'table_{stat}.tex'), float_format="%.2f", na_rep='--')



    if "experiment" in quant.columns.names:
        quant_max_min = quant.min(axis=0).max(level=["experiment", "stat", "set"]).to_frame().transpose()
        quant_max_min = quant_max_min.reindex(["A", "B"], axis=1, level="experiment")
        quant_max_min = quant_max_min.reindex(["train", "test"], axis=1, level="set")
        quant_max_min = quant_max_min.reindex(["loss", key_err], axis=1, level="stat")
        quant_max_min.to_latex(os.path.join(dirname, f'table_sum.tex'), float_format="%.2f", na_rep='--')
    return


def assert_col_order(df, cols, id_vars=None, values="value"):
    if id_vars is None:
        id_vars=df.index.name
    if df.columns.names != cols:
        # the order is
        # perform pivot
        df = pd.melt(df.reset_index(), id_vars=id_vars).pivot(index=id_vars, columns=cols, values=values)
    df.sort_index(axis=1, inplace=True)
    return df

def get_image_resize(transform):
    '''Return the image from the tranform if any'''

    tfms = transform.transforms
    resize =None
    for t in tfms:
        if type(t) is transforms.Resize:
            resize = t.size  # warning, what if tuple
    return resize

def get_ouput_dim_conv(d_in, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
    '''Assume tuple input'''
    d_out =  tuple( int(math.floor((d + 2 * p - dil*(k -1) - 1) / s + 1)) for d,p,dil,k,s in zip(d_in, padding, dilation, kernel_size, stride))
    return d_out

def get_ouput_dim_conv(d_in, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), output_padding=(0,0)):
    d_out =  tuple( ((d-1) * s  - 2*p + dil * (k - 1 ) + out_p+1) for d,s,p,dil,k,out_p in zip(d_in, stride, padding, dilation, kernel_size, output_padding))
    return d_out

def parse_archi(fname, *args):
    '''Parse a log file containing architectures for networks'''


    nets = {}

    net_re = re.compile('(Model|Linear classifier):')
    LAYERS = {
        'Linear': nn.Linear,
        'BatchNorm1d': nn.BatchNorm1d,
        'LeakyReLU': nn.LeakyReLU,
        'MaxPool2d': nn.MaxPool2d,
        'ConvTranspose2d': nn.ConvTranspose2d,
        'Conv2d': nn.Conv2d,
        'BatchNorm2d': nn.BatchNorm2d,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid,
        'Bilinear': nn.Bilinear,
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
        'RandomSamplerParallel': models.classifiers.RandomSamplerParallel,
        'RandomSampler': models.classifiers.RandomSampler,
        'MultiLinear': models.classifiers.MultiLinear,
    }

    depth = 0
    with open(fname, 'r') as _f:
        for idx, line in enumerate(_f, 1):
            if depth == 0:
                new_net = net_re.match(line)
                if new_net:
                    c_depth = 1
                    net = dict()
                    net_name = line.split(' ')[1].rstrip('(')
                    depth += line.count("(")
                    depth -= line.count(")")

            elif depth == 1:  # definition of the modules (attributes of the network)

                fields = line.split(':')
                if len(fields) == 2:
                    # should always be the case
                    # not the case when closing the sequential module
                    par_pos = fields[1].find('(')
                    module_name = fields[0].strip().rstrip(')').lstrip('(')
                    # will me main, etc.
                    module_type = fields[1][:par_pos].strip()  # Sequential

                    if module_type == 'Sequential':
                        nn_module_type = nn.Sequential
                        nn_layers = []
                    elif module_type == 'Module List':
                        nn.module_type = nn.ModuleList
                    #elif module_type == 'AdaptiveAveragePool2d':
                        #nn.module_type =
                    else:

                        raise NotImplementedError(module_type)

                depth += line.count("(")
                depth -= line.count(")")


            elif depth == 2:

                depth += line.count("(")
                depth -= line.count(")")


                fields = line.split(':')

                if len(fields) == 2:
                    layer_name = fields[0].strip().lstrip('(').rstrip(')')

                    layer_type = fields[1][:fields[1].find('(')].strip()
                    layer_args = fields[1][fields[1].find('(')+1:].strip().rstrip(')') # leave trailing )
                    if layer_type == "RandomSamplerParallel":
                        r = re.compile(',|/')
                        rsp = r.split(layer_args)
                        rsp = [r.strip('()').split('=') for r in rsp]
                        kwargs = {}
                        for kw in rsp:
                            if len(kw) == 2:
                                key,val = kw[0].strip(), kw[1].strip()
                                val = cast(val.strip())
                                kwargs[key] = val
                            else:
                                val, key= kw[0].strip().split(' ')  # e.g. 500 total
                                kwargs[key] = cast(val)
                        kwargs["remove"] = kwargs["total"] - kwargs["kept features"]
                        kwargs.pop("kept features")
                        args = []
                    else:
                        args, kwargs = parse_layer_args(layer_args)
                    try:
                        nn_layers.append((layer_name, LAYERS[layer_type](*args, **kwargs)))
                    except TypeError as e:
                        print(e)
                        print("Layer type:", layer_type)


                if depth == 1:  # end of sequential
                    net[module_name] = nn_module_type(OrderedDict(nn_layers)) # is sequential
                    nn_layers = []


    return net


def construct_FCN(archi):

    class FCN(nn.Module):

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, x):

            return self.main(x.view(x.size(0), -1))

    return FCN(archi['main'])

def construct_classifier(archi):

    class ClassifierFCN(nn.Module):

        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, x):
            x=x.view(x.size(0), -1)
            out = self.network(x)
            return out

    return ClassifierFCN(archi['network'])



def construct_G(archi_G, image_size):
    '''construct the nn.Modules from the dict of archis'''

    class Generator(nn.Module):

        def __init__(self, fc=None, deconv=None, deconv_dim=None, depth=None, image_size=None, inter_shape=None, decoder=None):
            super().__init__()
            if decoder is not None:
                self.decoder = Decoder(decoder)
                self.forward = self._forward_decoder
            else:
                self.fc = fc
                self.deconv = deconv
                self.forward = self._forward_gen
                self.inter_shape = (deconv_dim, image_size // depth, image_size // depth) if inter_shape is None else inter_shape

        def _forward_gen(self, z):
            out = self.fc(z)
            out = out.view(z.shape[0], *self.inter_shape)
            out = self.deconv(out)
            return out

        def _forward_decoder(self, z):
            return self.decoder(z)

    class Decoder(nn.Module):
        '''Simple decoder module'''

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, z):
            out = self.main(z.view(z.size(0), z.size(1), 1, 1))
            return out

    class GeneratorGP(nn.Module):

        def __init__(self, block1,  block2, deconv_out, preprocess, act, DIM=64, IM_DIM=(1, 28, 28)):
            super().__init__()
            self.block1 = block1
            self.block2 = block2
            self.deconv_out = deconv_out
            self.preprocess = preprocess
            self.sigmoid = nn.Sigmoid()
            self.DIM = DIM
            self.IM_DIM = IM_DIM

        def forward(self, z):

            output = self.preprocess(z)
            output = output.view(-1, 4*self.DIM, 4, 4)
            #print output.size()
            output = self.block1(output)
            #print output.size()
            output = output[:, :, :7, :7]
            #print output.size()
            output = self.block2(output)
            #print output.size()
            output = self.deconv_out(output)
            output = self.sigmoid(output)
            #print output.size()
            return output.view(-1, *self.IM_DIM)



    if 'block1' in archi_G.keys():
        return GeneratorGP(archi_G['block1'], archi_G['block2'], archi_G['deconv_out'], archi_G['preprocess'], archi_G['sigmoid'])
    elif archi_G['module_type'] == 'Sequential':
        return Generator(archi_G['fc'], archi_G['deconv'], archi_G['dim'], archi_G['depth'], image_size, archi_G.get('inter_shape', None))
    elif archi_G['module_type']== 'Decoder':
        return Generator(decoder=archi_G['main'])

def construct_F(archi_F, image_size):
    '''construct the nn.Modules from the dict of archis'''

    class Discriminator(nn.Module):

        def __init__(self, conv=None, fc=None,  conv_dim=None, depth=None, image_size=None, inter_dim=None, encoder=None):
            super().__init__()
            if encoder is not None:
                self.encoder = Encoder(encoder)
                self.forward = self._forward_encoder
            else:
                self.conv = conv
                self.fc = fc
                self.depth = depth
                self.forward = self._forward
                self.inter_dim = conv_dim * (image_size // depth) * (image_size // depth) if inter_dim is None else inter_dim

        def _forward(self, x):
            out = self.conv(x)
            out = out.view(x.shape[0], self.inter_dim)
            out = self.fc(out)
            return out

        def _forward_encoder(self, x):
            return self.encoder(x)

    class Encoder(nn.Module):
        '''Simple decoder module'''

        def __init__(self, main):
            super().__init__()
            self.main = main

        def forward(self, x):
            out = self.main(x).view(x.size(0), -1)
            return out

    class DiscriminatorGP(nn.Module):

        def __init__(self, main, output, DIM=64, IM_DIM=(1,28,28)):

            super().__init__()
            self.main = main
            self.output = output
            self.DIM = DIM
            self.IM_DIM = IM_DIM

        def forward(self, x):

            x = x.view(-1, *self.IM_DIM)
            out = self.main(x)
            out = out.view(-1, 4*4*4*self.DIM)
            out = self.output(out)  # might be id

            return out

    if 'output' in archi_F.keys():
        return DiscriminatorGP(archi_F['main'], archi_F['output'])
    elif archi_F['module_type'] == 'Sequential':
        return Discriminator(archi_F['conv'], archi_F['fc'], archi_F['dim'], archi_F['depth'], image_size, archi_F.get('inter_dim', None))
    elif archi_F['module_type'] == 'Encoder':
        return Discriminator(encoder=archi_F['main'])

def cast(s):
    '''casts the string s into apprpriate type'''
    def cast_num(n):

        val = None
        try:
            val = int(n)
        except:
            try:
                val = float(n)
            except:

                submodules = n.split('.')
                if len(submodules) > 1: # of type PIL.xxx.kkk
                    assert submodules[0] == 'PIL'
                    val = PIL.__dict__[submodules[1]].__dict__[submodules[2]]


        return val

    is_tuple = len(s.split(',')) > 1 and s[0] == '('
    is_list = len(s.split(',')) > 1 and s[0] == '['

    if is_tuple:
        val = tuple( cast_num(n.strip('()')) for n in s.split(',') if len(n.strip('()')) > 0)
    elif is_list:
        val = list(cast_num(n.strip('[]')) for n in s.split(',') if len(n.strip('[]')) > 0)
    else:
        if s == 'True':
            val = True
        elif s == 'False':
            val = False
        else:
            val = cast_num(s)
    return val

def parse_layer_args(layer_args_string):


    d = 0  # the parenthesis depth
    buff = []
    key = ''
    kwargs = {}
    args = []
    for idx, c in enumerate(layer_args_string):
        if  c == '(' or c == '[':
            d += 1
        elif c == ')' or c == ']':
            d -= 1
        # else:
        if idx == len(layer_args_string)-1:  # last character
            buff.append(c)
        if (d == 0 and c == ',') or idx == len(layer_args_string)-1:
            # end of argument
            buff = ''.join(buff).strip()
            if key:
                num_words = len(key.split(' '))
                kwargs[key] =  cast(buff)
            else:
                args.append(cast(buff))
            buff = []
            key = ''
        elif c == '=':
            key = ''.join(buff).strip()
            buff = []
        else:
            buff.append(c)

    return args, kwargs

def print_cuda_memory_usage(device, logs=sys.stdout, epoch=None):
    '''The current usage of GPU memory'''



    print(80*'*', file=logs, flush=True)
    print(epoch, file=logs, flush=True)
    print(80*'*', file=logs, flush=True)
    print(torch.cuda.memory_summary(device), file=logs, flush=True)



def count_hidden_layers(model, act=nn.ReLU):

    if hasattr(model, 'main'):
        list_layer = model.main
    elif isinstance(model, torchvision.models.vgg.VGG):
        list_layer = list(model.features) + list(model.classifier)
    elif isinstance(model, nn.DataParallel):
        return count_hidden_layers(model.module, act)
    else:
        raise NotImplementedError


    cnt = 0
    for layer in list_layer:
        cnt += isinstance(layer, act)
    return cnt


if __name__ == '__main__':


    parser = argparse.ArgumentParser('debug')
    parser.add_argument('files', nargs='*')

    argv = parser.parse_args()

    parse_archi(argv.files[0])

