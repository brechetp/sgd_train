import torch.nn as nn
import torch
import numpy as np
import models.pretrained
import utils
import math
import copy
import sys

class FCN(nn.Module):
    '''Fully-connected neural network'''

    def __init__(self, input_dim, num_classes, width=[1024], lrelu=0.01):

        super().__init__()

        if type(width) is int:
            width = [width]
        sizes = [input_dim, *width, num_classes]
        #mlp = utils.construct_mlp_net(sizes, fct_act=nn.LeakyReLU, kwargs_act={'negative_slope': lrelu, 'inplace': True})
        mlp = utils.construct_mlp_net(sizes, fct_act=nn.ReLU, args_act=[True])
        self.widths = width

        N = width[-1]  # size of the last layer
        M = N//2  # the number of units that are selected

        #self.main = nn.Sequential(mlp, nn.Softmax())
        self.main = mlp

        return


    def forward(self, x):

        #vec = torch.cat((is_man.view(-1, 1), x.view(x.size(0), -1)), dim=1)

        out = self.main(x.view(x.size(0), -1))

        return out

    def last_random(self, x):
        '''Returns the last layer of the network'''
        out = self.main[:-1](x.view(x.size(0), -1))
        return out[:, self.random_choice]




    def num_parameters(self, only_require_grad=False):
        '''Return the number of parameters in the model'''
        return sum(p.numel() for p in self.parameters() if not only_require_grad or p.requires_grad)


class Linear(nn.Module):
    '''Simple linear classifier on top of a model'''

    def __init__(self, model, try_num=1, keep_ratio=0.5):
        '''model: a FCN object
        T: the number of tries'''
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.weight_hidden = nn.ModuleList()
        n_classes = self.C = model.main[-1].out_features # the output dimension (i.e. class)
        self.neurons = []
        for layer in self.model.main[:-1]:
            if isinstance (layer, nn.Linear):
                # linear layer
                d_out = layer.out_features
                d_int = layer.in_features
                self.neurons.append(d_out)
                self.weight_hidden.append(nn.Linear(d_out, n_classes, bias=True))

        self.L = len(self.weight_hidden)

        self.N = model.main[-1].in_features  # the number of features to consider
        self.P = int(keep_ratio*self.N+0.5)  # keep  ratio of the features for the random choice
        self.keep_ratio = keep_ratio
        random_perms =torch.cat([torch.randperm(self.N)[:self.P].view(1, -1) for idx in range(try_num)], dim=0)  # the random choice
        # of size  TxP
        #random_perm = random_perms.unsqueeze(1) # Tx1xP
        self.mask = nn.Parameter(torch.zeros((try_num, self.N)), requires_grad=False)
        ones = torch.ones((try_num, self.N))
        self.mask.scatter_(1, random_perms, ones)
        # random_perm is now TxP
        self.random_perm = nn.Parameter(random_perms.unsqueeze(1), requires_grad=False)  # Tx1xP
        self.T = try_num
        #  TxCxP
        self.weight = nn.Parameter(self.mask.unsqueeze(1) * torch.randn((self.T, self.C, self.N)))
        self.bias = nn.Parameter(torch.zeros(self.T, self.C))


    def extra_repr(self):

        return "random tries size: {}, {}".format(self.weight.size(), self.bias.size())

    def forward(self, x):
        '''return size: TxBxC'''

        B = x.size(0) # the batch size
        x = x.view(B, -1)
        out_hidden = x.new_zeros((self.L, B, self.C))
        idx=0
        for layer in self.model.main[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear) and idx < self.L:
                out_hidden[idx] = self.weight_hidden[idx](x.clone())
                idx += 1

        out_last = x
        out_last = out_last.unsqueeze(0).expand(self.T, -1, -1)
        out_mask = self.mask.unsqueeze(1) * out_last
        # of size TxBxP


        # BxTxC = BxTxCxP * BxTxPx1
        #out_einsum = torch.einsum("abcd,abde->abce",
        #                   self.weight.unsqueeze(1).expand(-1, out_rand.size(0),  -1, -1),
        #                   out_rand) #+ self.b
        out_matmul = out_mask.matmul(self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)

        return out_matmul, out_hidden  # TxBxC


class ClassifierVGG(nn.Module):

    def __init__(self, model, num_tries=10, Rs=None) :
        '''model: a pretrained VGG model'''

        super().__init__()

        #self.model = model.eval()  # should set the dropout layers to eval mode ?
        self.model = model
        self.model.requires_grad_(False)
        # the number of classes
        num_classes = self.model.classifier[-1].out_features
        # the different dimensions for the input
        # take the dimensions of the classifier (should be 3?)
        Ns = [layer.in_features for layer in self.model.classifier[::-1] if isinstance(layer, nn.Linear)]

        # Rs are the neurons that we remove from the layers L-1 ...
        self.Rs = Rs = [n // 2 for n in Ns] if Rs is None else Rs  # the R's , i.e. the number of removed neurons
        # indices of the FC layers



        #random_perm = random_perms.unsqueeze(1) # Tx1xP
        # random_perm is now TxP
        self.linear_mask = LinearParallelMasked((Ns[0], Rs[0]), num_classes, num_tries=num_tries)
        self.shallow_net = utils.contruct_mmlp_net([(Ns[1], Rs[1]), Rs[0], num_classes], num_tries=num_tries)
        #nn.Sequential(
        #    LinearParallelMasked((Ns[1], Rs[1]), (Rs[0]), num_tries=num_tries),
        #    nn.ReLU(inplace=True),
        #    MultiLinear(Rs[0], num_classes, num_tries=num_tries)
        #)

    #def to(self, device):
    #    '''Recursively put the parameters on the device'''

    #    self.linear_mask.to(device)
    #    self.shallow_net.to(device)
    #    self.model.to(device)




    def forward(self, x):
        '''first goes through VGG, then classifies'''

        with torch.no_grad():
            out_features = self.model.avgpool(self.model.features(x)).view(x.size(0), -1)  # forward VGG
            out_partial = self.model.classifier[:2](out_features)  # first linear and relu
        # need to branch out for the different random choices
        shallow_branch = self.shallow_net(out_partial.clone())  # the shallow reduced network

        with torch.no_grad():  # frozen VGG
            out_next = self.model.classifier[2:4](out_partial.clone())  # next linear relu (no dropout)
            # only the last classifier is skipped

        linear_branch = self.linear_mask(out_next)

        return linear_branch, shallow_branch

def init_kaiming(weight, bias, fan_in:int=None):
    '''Kaiming (uniform) initialization with for parameters with parallel channels'''

    #a = math.sqrt(5.0)  # initialization of a linear model in pytorch

    # weight is of size TxOxI
    if fan_in is None:
        fan_in = weight.size(1)  # the neurons that are kept (fan_in)
    gain = nn.init.calculate_gain('relu')  # kaiming uniform init
    std = gain / (math.sqrt(fan_in))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        nn.init.uniform_(weight, -bound, bound)
        #bound = 1. / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)


class MultiLinear(nn.Module):
    '''Define a parallel linear layer'''


    def __init__(self, in_features, out_features, num_tries=10):

        super().__init__()

        weight = torch.empty((num_tries, in_features, out_features))
        bias = torch.empty((num_tries, 1, out_features))

        init_kaiming(weight, bias)

        # size of weight: TxNxD
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.num_tries = num_tries
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        ''' Parallel matrix multiplication '''
        # size of x: TxBxN
        # weight of size: TxNxD

        out_matmul = x.matmul(self.weight) + self.bias
        # output of size TxBxD

        return out_matmul


    def extra_repr(self):

        return "in_features={}, out_features={}, num_tries={}".format(self.in_features, self.out_features, self.num_tries)






class ClassifierCopyVGG(nn.Module):
    '''The classifiers from a copy of a trained FCN network'''

    def __init__(self, model, start_idx=1, mult=2, keep=1/2):
        """start_idx is the index for the first hidden layer to be modified"""

        super().__init__()

        # the indices for the hidden layers
        #size_out = [layer.out_features for layer in model.main[:-1] if isinstance(layer, nn.Linear)]



        self.n_layers = L = model.n_layers # the total number of hiden layers
        #self.features = pass
        #self.classifier = pass
        layer_list = []
        c_prev = model.features[0].in_channels  # the first dimension
        select = torch.arange(c_prev).view(-1, 1)  # the first selection i.e. all rows
        self.selected  = [select]  # save the selected neurons
        idx = 1
        r = keep
        self.size_out = []
        self.mult = mult
        feat_layer_list = []

        self.n_classes = model.classifier[-1].out_features

        for layer in model.features:  # for all layer in the original model feature extraction
            # layer can be nn.Covn2d, nn.MaxPool, nn.ReLUL

            if isinstance(layer, nn.Conv2d):
                self.size_out.append(c_prev)

                if idx < start_idx:  # simply copy the layer
                    layer_copy = copy.deepcopy(layer)
                    c_prev = layer.out_channels
                    self.selected.append(torch.arange(c_prev).view(-1, 1))
                elif start_idx <= idx:  # select the different neurons to switch off
                    c_out = layer.out_channels
                    keep_out = round(c_out * r)  #if idx <= L else c_out  # for the last layer keep all the outputs
                    # switch the previously selected rows as columns now
                    select, select_prev = torch.randperm(c_out)[:keep_out].sort().values.view(-1, 1), self.selected[-1].view(1, -1)

                    self.selected.append(select)  # keep track of the selected rows
                    selected_weights =copy.deepcopy(layer.weight[select, select_prev, :, :])
                    layer_copy = nn.Conv2d(in_channels=c_prev, out_channels=keep_out, kernel_size=layer.kernel_size,
                                           stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                           groups=layer.groups, bias=(not layer.bias is None), padding_mode=layer.padding_mode)  # the actual new parameter
                    layer_copy.weight = nn.Parameter(selected_weights)
                    if not layer.bias is None:
                        selected_bias = copy.deepcopy(layer.bias[select.view(-1)])
                        layer_copy.bias = nn.Parameter(selected_bias, requires_grad=False)
                    c_prev = keep_out  # for the next layer
                else:
                    raise NotImplementedError

            elif isinstance(layer, nn.ReLU):
                idx += 1
                layer_copy = layer
            elif isinstance(layer, nn.MaxPool2d):
                layer_copy = layer
            else:
                raise NotImplementedError

            feat_layer_list.append(layer_copy)

        self.features = nn.Sequential(*feat_layer_list)
        self.avgpool = model.avgpool

        # the selected indices at the first fully connected layer
        size_pool = self.avgpool.output_size
        offset = size_pool[0]*size_pool[1]
        #select_mask = torch.zeros(c_out, size_pool[0], size_pool[1])
        #select_mask[select_prev, :, :] = torch.arange(0, offset, dtype=torch.long).view(1, size_pool[0], size_pool[1])
        # selects all contiguous indices for each kernel
        select = self.selected[-1].view(-1, 1, 1)*offset + torch.arange(0, offset, dtype=torch.long).view(1, *size_pool)
        select = torch.flatten(select).view(-1, 1)
        self.selected[-1] = select
        class_layer_list = []
        #select_prev = self.selected[-1].view(1, -1) * offset
        n_prev = c_prev * offset  # the number of kept dimensions by the last convolution


        for layer in model.classifier:  # for all layer in the original model feature extraction
            # layer can be nn.Covn2d, nn.MaxPool, nn.ReLUL
            if isinstance(layer, nn.Linear):
                self.size_out.append(n_prev)

                if idx < start_idx:  # simply copy the layer
                    layer_copy = copy.deepcopy(layer)
                    n_prev = layer.out_features
                    self.selected.append(torch.arange(n_prev).view(-1, 1))
                elif start_idx <= idx:  # select the different neurons to switch off
                    n_out = layer.out_features
                    keep_out = round(n_out *r) if idx <= L else n_out  # for the last layer keep all the outputs
                    # switch the previously selected rows as columns now
                    select, select_prev = torch.randperm(n_out)[:keep_out].sort().values.view(-1, 1), self.selected[-1].view(1, -1)
                    self.selected.append(select)  # keep track of the selected rows
                    selected_weights =copy.deepcopy(layer.weight[select, select_prev])
                    layer_copy = nn.Linear(n_prev, keep_out, bias=layer.bias is not None)  # the actual new parameter
                    layer_copy.weight = nn.Parameter(selected_weights)
                    if layer.bias is not None:
                        selected_bias = copy.deepcopy(layer.bias[select.view(-1)])
                        layer_copy.bias = nn.Parameter(selected_bias)
                    n_prev = keep_out  # for the next layer

                else:
                    raise NotImplementedError

            elif isinstance(layer, nn.ReLU):
                idx += 1
                layer_copy = layer
            elif isinstance(layer, nn.MaxPool2d):
                layer_copy = layer
            else:
                raise NotImplementedError

            class_layer_list.append(layer_copy)


        self.classifier = nn.Sequential(*class_layer_list)




    def forward_no_mult(self, x, no_mult=False):
        """Forward for the copied networks"""

        # go through the different layers inside main
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x, no_mult=False):
        """Forward for the copied networks"""

        x = self.forward_no_mult(x)
        return self.mult*x


class DropoutFCN(nn.Module):
    '''The classifiers from a copy of a trained FCN network'''

    def __init__(self, model: FCN, keep_ratio=1/2):
        """start_idx is the index for the first hidden layer to be modified"""

        super().__init__()

        # the indices for the hidden layers
        #size_out = [layer.out_features for layer in model.main[:-1] if isinstance(layer, nn.Linear)]



        self.size_out = []
        self.keep_ratio=keep_ratio
        self.network = copy.deepcopy(model.main)
        self.n_layers = L = len(list(filter(lambda x: isinstance(x, nn.ReLU), self.network)))

        self._idx_layer = L
        self.lookup_layer = [idx for (idx, layer) in enumerate(self.network) if isinstance(layer, nn.Linear)]

        self.selected=[]
        self.layer_from, self.layer_to =self.layers_from_to()



    @property
    def n_classes(self):
        return self.get_layer(-1).out_features

    @property
    def mult(self):
        # the multiplicative scalar
        return self._mult

    @mult.setter
    def mult(self, mult):
        self._mult = mult

    @property
    def idx_from(self):
        return self.lookup_layer[self._idx_layer-1]

    @property
    def idx_to(self):
        return self.lookup_layer[self._idx_layer]

    @property
    def idx_layer(self):
        return self._idx_layer

    def get_layer(self, idx=None):
        # the index in terms of nn.Sequential of the hidden layer
        if idx is None:
            idx = self._idx_layer-1  # default take the one before the idx
        return self.network[self.lookup_layer[idx]]

    def layers_from_to(self):
        return copy.deepcopy(self.get_layer(self._idx_layer-1)), copy.deepcopy(self.get_layer(self._idx_layer))



    def new_sample(self, select=None):
        # performs a new sample of the current working layer

        M = self.layer_from.out_features
        nin_from, nout_to = self.layer_from.in_features, self.layer_to.out_features
        K = round(self.keep_ratio * M)
        if select is None:
            select = torch.randperm(M)[:K].sort().values
            self.selected.append(select)  # keep track of the selected neurons
        weights_from =(self.layer_from.weight[select, :])
        weights_to = (self.layer_to.weight[:, select])
        new_layer_from = nn.Linear(nin_from, K)
        new_layer_to = nn.Linear(K, nout_to)
        new_layer_from.weight = nn.Parameter(weights_from)
        new_layer_to.weight = nn.Parameter(weights_to)
        if self.layer_from.bias is not None:
            selected_bias = (self.layer_from.bias[select])
            new_layer_from.bias = nn.Parameter(selected_bias, requires_grad=False)

        self.network[self.idx_from] = new_layer_from
        self.network[self.idx_to] = new_layer_to
        # switch the previously selected rows as columns now



    def confirm_sample(self, idx):
        # selects the current or given permutation as best sampling of units
        # index at which operate (in relu): self.idx_sample

        select = self.selected[idx]
        self.new_sample(select)
        self._idx_layer -= 1
        l = self._idx_layer
        self.layer_from, self.layer_to = self.layers_from_to()

    def forward_no_mult(self, x):
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x

    def forward(self, x):

        x = self.forward_no_mult(x)
        x = self.mult*x
        return x

class DropoutVGG(nn.Module):
    '''The classifiers from a copy of a trained FCN network'''

    def __init__(self, model, keep_ratio=1/2):
        """start_idx is the index for the first hidden layer to be modified"""

        super().__init__()

        # the indices for the hidden layers
        #size_out = [layer.out_features for layer in model.main[:-1] if isinstance(layer, nn.Linear)]



        self.size_out = []
        self.keep_ratio=keep_ratio
        self.features = copy.deepcopy(model.features)
        self.classifier = copy.deepcopy(model.classifier)
        self.avgpool = copy.deepcopy(model.avgpool)

        self.n_layers = L = len(list(filter(lambda x: isinstance(x, nn.ReLU), list(self.features) + list(self.classifier))))

        self._idx_layer = L
        self.lookup_feat = [idx for (idx, layer) in enumerate(self.features) if isinstance(layer, nn.Conv2d)]
        self.lookup_class = [idx for (idx, layer) in enumerate(self.classifier) if isinstance(layer, nn.Linear)]

        self.n_layers_feat = L_feat = len(self.lookup_feat)
        self.n_layers_class = L_class = len(self.lookup_class)

        self.selected=[]
        self.layer_from, self.layer_to = self.layers_from_to()
        self.type_from, self.type_to = self.types_from_to()
        self.n_classes = self.classifier[-1].out_features




    @property
    def mult(self):
        # the multiplicative scalar
        return self._mult

    @mult.setter
    def mult(self, mult):
        self._mult = mult

    def lookup_layer(self, idx):
        # idx can be targeting a conv or lin layer
        if self.get_type(idx) == "feat":
            return self.lookup_feat[idx]
        else:
            if idx >=0:
                return self.lookup_class[idx-self.n_layers_feat]
            else: # -1 ...
                return self.lookup_class[idx]

    @property
    def idx_from(self):
        return self.lookup_layer(self._idx_layer-1)

    @property
    def idx_to(self):
        return self.lookup_layer(self._idx_layer)

    @property
    def idx_layer(self):
        return self._idx_layer


    def get_layer(self, idx=None):
        # the index in terms of nn.Sequential of the hidden layer
        if idx is None:
            idx = self._idx_layer  # default take the one before the idx
        idx_seq = self.lookup_layer(idx)
        layer= self.features[idx_seq] if self.get_type(idx)=='feat' else self.classifier[idx_seq]
        return layer

    def get_type(self, idx=None):
        # the type (feat or class) of the layer at idx
        if idx is None:
            idx = self._idx_layer-1  # default take the one before the idx
        if  0<=idx < self.n_layers_feat:
            return 'feat'
        else:
            return 'class'

    def layers_from_to(self):
        return copy.deepcopy(self.get_layer(self._idx_layer-1)), copy.deepcopy(self.get_layer(self._idx_layer))

    def types_from_to(self):
        return self.get_type(self._idx_layer-1), self.get_type(self._idx_layer)

    def new_sample(self, select=None):
        # performs a new sample of the current working layer

        if self.type_from == self.type_to == "class":
            M = self.layer_from.out_features
            nin_from, nout_to = self.layer_from.in_features, self.layer_to.out_features
            K = round(self.keep_ratio * M)
            if select is None:
                select = torch.randperm(M)[:K].sort().values
                self.selected.append(select)  # keep track of the selected neurons
            weights_from =(self.layer_from.weight[select, :])
            weights_to = (self.layer_to.weight[:, select])
            new_layer_from = nn.Linear(nin_from, K, bias=(not self.layer_to.bias is None))
            new_layer_to = nn.Linear(K, nout_to, bias=(not self.layer_from.bias is None))
            new_layer_from.weight = nn.Parameter(weights_from)
            new_layer_to.weight = nn.Parameter(weights_to)
            if self.layer_from.bias is not None:
                selected_bias = (self.layer_from.bias[select])
                new_layer_from.bias = nn.Parameter(selected_bias, requires_grad=False)

            self.classifier[self.idx_from] = new_layer_from
            self.classifier[self.idx_to] = new_layer_to

        elif self.type_from == self.type_to == "feat":
        # switch the previously selected rows as columns now
            M = self.layer_from.out_channels
            nin_from, nout_to = self.layer_from.in_channels, self.layer_to.out_channels
            K = round(self.keep_ratio * M)
            if select is None:
                select = torch.randperm(M)[:K].sort().values
                self.selected.append(select)  # keep track of the selected neurons
            weights_from =(self.layer_from.weight[select, :, :, :])
            weights_to = (self.layer_to.weight[:, select, :, :])
            new_layer_from = nn.Conv2d(in_channels=nin_from, out_channels=K,kernel_size=self.layer_from.kernel_size,
                                           stride=self.layer_from.stride, padding=self.layer_from.padding, dilation=self.layer_from.dilation,
                                           groups=self.layer_from.groups, bias=(not self.layer_from.bias is None), padding_mode=self.layer_from.padding_mode)  # the actual new parameter
            new_layer_to = nn.Conv2d(in_channels=K, out_channels=nout_to, kernel_size=self.layer_to.kernel_size,
                                           stride=self.layer_to.stride, padding=self.layer_to.padding, dilation=self.layer_to.dilation,
                                           groups=self.layer_to.groups, bias=(not self.layer_to.bias is None), padding_mode=self.layer_to.padding_mode)  # the actual new parameter
            new_layer_from.weight = nn.Parameter(weights_from)
            new_layer_to.weight = nn.Parameter(weights_to)
            if self.layer_from.bias is not None:
                selected_bias = (self.layer_from.bias[select])
                new_layer_from.bias = nn.Parameter(selected_bias, requires_grad=False)

            self.features[self.idx_from] = new_layer_from
            self.features[self.idx_to] = new_layer_to

        elif self.type_from == "feat" and self.type_to == "class":
            # has to bridge between the two
            M = self.layer_from.out_channels
            nin_from, nout_to = self.layer_from.in_channels, self.layer_to.out_features
            K = round(self.keep_ratio * M)
            if select is None:
                select = torch.randperm(M)[:K].sort().values
                self.selected.append(select)  # keep track of the selected neurons
            size_pool = self.avgpool.output_size
            offset = size_pool[0]*size_pool[1]
            select_lin = select.view(-1, 1, 1)*offset + torch.arange(0, offset, dtype=torch.long).view(1, *size_pool)
            select_lin = torch.flatten(select_lin)
            weights_from =(self.layer_from.weight[select, :, :, :])
            weights_to = (self.layer_to.weight[:, select_lin])
            new_layer_from = nn.Conv2d(in_channels=nin_from, out_channels=K,kernel_size=self.layer_from.kernel_size,
                                           stride=self.layer_from.stride, padding=self.layer_from.padding, dilation=self.layer_from.dilation,
                                           groups=self.layer_from.groups, bias=(not self.layer_from.bias is None), padding_mode=self.layer_from.padding_mode)  # the actual new parameter
            new_layer_to = nn.Linear(len(select_lin), nout_to, bias=(not self.layer_from.bias is None))
            new_layer_from.weight = nn.Parameter(weights_from)
            new_layer_to.weight = nn.Parameter(weights_to)
            if self.layer_from.bias is not None:
                selected_bias = (self.layer_from.bias[select])
                new_layer_from.bias = nn.Parameter(selected_bias, requires_grad=False)

            self.features[self.idx_from] = new_layer_from
            self.classifier[self.idx_to] = new_layer_to

        else:
            raise NotImplementedError

    def confirm_sample(self, idx):
        # selects the current or given permutation as best sampling of units
        # index at which operate (in relu): self.idx_sample

        select = self.selected[idx]
        self.new_sample(select)
        self._idx_layer -= 1
        self.layer_from, self.layer_to = self.layers_from_to()
        self.type_from, self.type_to = self.types_from_to()

    def forward_no_mult(self, x):
        """Forward for the copied networks"""

        # go through the different layers inside main
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x, no_mult=False):
        """Forward for the copied networks"""

        x = self.forward_no_mult(x)
        return self.mult*x




class ClassifierCopyFCN(nn.Module):
    '''The classifiers from a copy of a trained FCN network'''

    def __init__(self, model: FCN, start_idx=1, mult=2, keep=1/2):
        """start_idx is the index for the first hidden layer to be modified"""

        super().__init__()

        # the indices for the hidden layers
        #size_out = [layer.out_features for layer in model.main[:-1] if isinstance(layer, nn.Linear)]



        self.n_layers = L = len([layer for layer in model.main if isinstance(layer, nn.ReLU)])  # the total number of hiden layers
        layer_list = []
        n_prev = model.main[0].in_features  # the first dimension
        select = torch.arange(n_prev).view(-1, 1)  # the first selection i.e. all rows
        self.selected  = [select]  # save the selected neurons
        idx = 1
        self.size_out = []
        self.mult = mult

        for layer in model.main:  # for all layer in the original model

            if isinstance(layer, nn.Linear):
                self.size_out.append(n_prev)

                if idx < start_idx:  # simply copy the layer
                    layer_copy = copy.deepcopy(layer)
                    n_prev = layer.out_features
                    self.selected.append(torch.arange(n_prev).view(-1, 1))
                elif start_idx <= idx:  # select the different neurons to switch off
                    n_out = layer.out_features
                    keep_out = round(keep * n_out)  if idx <= L else n_out  # for the last layer keep all the outputs
                    # switch the previously selected rows as columns now
                    select, select_prev = torch.randperm(n_out)[:keep_out].sort().values.view(-1, 1), self.selected[-1].view(1, -1)

                    self.selected.append(select)  # keep track of the selected rows
                    selected_weights =copy.deepcopy(layer.weight[select, select_prev])
                    layer_copy = nn.Linear(n_prev, keep_out)  # the actual new parameter
                    layer_copy.weight = nn.Parameter(selected_weights)
                    n_prev = keep_out  # for the next layer
                else:
                    raise NotImplementedError

            elif isinstance(layer, nn.ReLU):
                idx += 1
                layer_copy = layer
            else:
                raise NotImplementedError

            layer_list.append(layer_copy)

        self.n_classes = n_out

        self.network = nn.Sequential(*layer_list)




    def forward_no_mult(self, x, prime=False):

        # go through the different layers inside main
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x

    def forward(self, x, prime=False):
        """Forward for the copied networks"""

        x = self.forward_no_mult(x)
        return x * self.mult




class ClassifierFCN(nn.Module):
    '''The classifiers plugged into a FCN network'''

    def __init__(self, model: FCN, num_tries, Rs, depth_max=None):

        super().__init__()
        # the indices of the features (i.e. after the activations
        # and the size of the output network
        indices  =[ idx+1 for idx, layer in enumerate(model.main) if isinstance(layer, nn.ReLU)]
        size_out = [layer.out_features for layer in model.main[:-1] if isinstance(layer, nn.Linear)]

        if depth_max is None:
            depth_max = len(indices)

        self.indices = indices[:depth_max]
        self.size_out = size_out[:depth_max]
        self.n_tries = num_tries

        self.model = model

        L = self.n_layers = len(self.indices)  # the total number of layers to plug into
        n_classes = self.n_classes = model.main[-1].out_features

        if isinstance(Rs, int):
            Rs = L*[Rs]

        # each network idx will have
        # 1. a random mask removing R neurons
        # 2. idx-1 depth with R neurons
        # 3. output layer into n_classes

        self.networks = nn.ModuleList()
        sizes = []



        for idx, (N, R) in enumerate(zip(self.size_out, Rs), 1):
            depth = L-idx
            sizes = [(N, R)] + depth * [R] + [n_classes]
            net = utils.construct_mmlp_net(sizes, fct_act=nn.ReLU, num_tries=num_tries)
            self.networks.append(net)

            #sizes =

    def forward(self, x, stop=None):
        '''Forward of the different layers
        stop: the index of the layer at which to stop the computation (between 1 and L)'''
        # stop

        # go through the different layers inside main

        feats=[x.view(x.size(0), -1)]
        idx_prev=0
        #out is of size LxTxBxC
        #out = x.new_ones((stop, self.n_tries, x.size(0), self.n_classes))  # new tensor with same device and dtype
        with torch.no_grad():
            for i, idx in enumerate(self.indices[:stop]):
                # for all the linear layers (marked by their indices in
                # self.indices)
                #feats. = self.model.main[idx_prev:idx](feat)  # the new input
                feats.append(self.model.main[idx_prev:idx](feats[-1]))  # the new input
                #out[i, :, :, :] = self.networks[i](feat.clone())  # output of the tunel
                idx_prev = idx

        out = []
        for i in range(len(feats)-1):

            out.append(self.networks[i](feats[1+i].clone()).unsqueeze(0))

        # in the forward order

        #return out

        return torch.cat(out)

class CopyFCN(nn.Module):
    '''A dummy copy of a FCN model to perform linear paths'''

    def __init__(self, model):

        for l in model.main: # for all the layers
            self.main.append(copy.deepcopy(l))

    def change_output_weights(self):

        self.main[-1].weights[:, select] = 0  # the weights are laid out as input x output
        self.main[1].bias[select] = 0


    """
    update of the weights that are "incomping"
    """
    def change_input_weights(self, idx):  # 0 means top, 1 means bottom



        pass

    """
    Update the weights according to a path (i.e. list of points) in parameter space
    """
    def change_weights_path(self, path):
        pass


    """
    """





class ClassifierVGGAnnex(nn.Module):

    '''An annex classifier based on a previously trained VGG network'''

    def __init__(self, model,  F=1/2, idx_entry=1):

        super().__init__()


        # the indices of the linear layers
        indices_feat  =[ (idx, layer.out_channels) for idx, layer in enumerate(model.features) if isinstance(layer, nn.Conv2d)]
        indices_class  =[ (idx, layer.out_features) for idx, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear)]
        #widths_feat = [layer.in_channels for layer in model.features if isinstance(layer, nn.Conv2d)]
        #widths_class = [layer. for layer in model.features if isinstance(layer, nn.Conv2d)]
        self.indices_feat = indices_feat
        self.indices_class = indices_class

        L_feat = self.n_layers_feat = len(self.indices_feat)  # the total number of layers of the original model (including input and output)
        L_class = self.n_layers_class = len(self.indices_class)  # the total number of layers of the original model (including input and output)

        layers_feat = []
        layers_class = []

        self.idx_entry = idx_entry
        if idx_entry == 0:
            entry_layer = None  # no entry layer in this case, simply the network with severed widths
            idx_feat = -1
            idx_class = -1
        elif 0 <idx_entry <= L_feat:  # entry into the feature section
            entry_layer_idx = indices_feat[idx_entry-1][0]
            entry_layer  = model.features[entry_layer_idx]  # the layer to sample from
            idx_feat = entry_layer_idx
            idx_class = -1
        elif idx_entry <= L_feat+L_class-1: # L_feat+L_class -2 is the total number of hidden layers
            entry_layer_idx = indices_class[idx_entry-L_feat-1][0]
            entry_layer = model.classifier[entry_layer_idx]
            idx_feat = len(model.features)  # copy all the features
            idx_class = entry_layer_idx  # copy until the linear layer
        else:
            raise ValueError(idx_entry, 'layer index too big')
            sys.exit(1)


        remain_feat = []
        remain_class = []


        if idx_feat>=0:
            layers_feat.extend(copy.deepcopy(model.features[:idx_feat]))  # all layers before idx_entry are simply copied
            # do NOT copy the entry layer

        remain_feat = model.features[idx_feat+1:]  # the remaining are not copied but their architectures matter


        if idx_class >= 0:
            layers_class.extend(copy.deepcopy(model.classifier[:idx_class]))

        remain_class = model.classifier[idx_class+1:]  # to be proccessed
        #map(lambda p: p.requires_grad_(False), [[p for p in layer.parameters()] for layer in layers])


        #self.size_out = size_in
        #self.n_tries = num_tries

        n_classes = self.n_classes = model.classifier[-1].out_features
        #n_classes = model.main[-1].out_features
        #depth_tunnel = L_-idx_entry-1  # the total idx_entry of the tunnel, excluding input and output ?
        #if idx_entry >=1:

        #else:  # idx_entry == 0
        #N = model.main[].in_features  # take the last width, i.e. smallest one


        #Rs_feat =  [int(round(F * w[1])) for w in indices_feat[i_f:]]
        #Rs_class = [int(round(F * w[1])) for w in indices_class[i_c:]] + [n_classes]

        if idx_entry == 0:
            c_out = model.features[0].in_channels
            R=0  # removes 0 channels from the image
            self.idx_rs = len(layers_feat)
            #self.random_sampler = None
        elif 0<idx_entry <=L_feat:
            c_in = entry_layer.in_channels  # to total depth without modification
            N = entry_layer.out_channels
            R  = int(round(F*N))
            #c_in = previous_layer.out_channels  # the previous
            c_out = N-R  # new output size
            #self.random_sampler = RandomSampler(N, R)
            self.idx_rs = len(layers_feat)
            self.random_perm =torch.randperm(N)[:N-R].sort().values.view(-1)  # select the columns
            #layers_feat.append(self.random_sampler)
            layer_copy = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=entry_layer.kernel_size,
                                    stride=entry_layer.stride, padding=entry_layer.padding, dilation=entry_layer.dilation,
                                    groups=entry_layer.groups, bias=(not entry_layer.bias is None), padding_mode=entry_layer.padding_mode)  # the actual new parameter
            selected_weights =copy.deepcopy(entry_layer.weight[self.random_perm, :, :, :])
            layer_copy.weight = nn.Parameter(selected_weights, requires_grad=False)
            if not entry_layer.bias is None:
                selected_bias = copy.deepcopy(entry_layer.bias[self.random_perm])
                layer_copy.bias = nn.Parameter(selected_bias, requires_grad=False)
            layers_feat.append(layer_copy)
        else:
            assert not remain_feat, idx_entry


        #idx=0
        for layer in remain_feat:
            if isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                layer_copy = layer
            elif isinstance(layer, nn.Conv2d):
                c_in, c_out = c_out, round(F*layer.out_channels) #Rs_feat[idx]
                layer_copy = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=layer.kernel_size,
                                       stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                       groups=layer.groups, bias=(not layer.bias is None), padding_mode=layer.padding_mode)  # the actual new parameter
                #idx += 1
            layers_feat.append(layer_copy)

        self.avgpool = model.avgpool

        # the selected indices at the first fully connected layer
        size_pool = self.avgpool.output_size
        offset = size_pool[0]*size_pool[1]
        #select_mask = torch.zeros(c_out, size_pool[0], size_pool[1])
        #select_mask[select_prev, :, :] = torch.arange(0, offset, dtype=torch.long).view(1, size_pool[0], size_pool[1])
        # selects all contiguous indices for each kernel
        #select = select_prev.view(-1, 1, 1)*offset + torch.arange(0, offset, dtype=torch.long).view(1, *size_pool)
        #select = torch.flatten(select).view(-1, 1)
        #self.selected.append(select)
        class_layer_list = []
        #num_classes = model.classifier[-1].out_features
        #select_prev = self.selected[-1].view(1, -1) * offset
        if idx_entry <= L_feat:  # if the entry layer was in the featuer section
            n_out = round(c_out*offset)
        elif L_feat < idx_entry:
            #N = entry_layer.in_features
            n_in = entry_layer.in_features
            N = entry_layer.out_features
            R  = int(round(F*N))
            n_out = N-R
            #self.random_sampler = RandomSampler(N, R)
            #self.idx_rs = len(layers_class)
            #layers_class.append(self.random_sampler)
            #n_out = layer.out_features
            #keep_out = round(n_out *r) if idx <= L else n_out  # for the last layer keep all the outputs
            # switch the previously selected rows as columns now
            #select, select_prev = torch.randperm(n_out)[:keep_out].sort().values.view(-1, 1), self.selected[-1].view(1, -1)
            #select, select_prev = torch.randperm(n_out)[:keep_out].sort().values.view(-1, 1), self.selected[-1].view(1, -1)
            self.random_perm =torch.randperm(N)[:N-R].sort().values.view(-1)  # select the columns
            #self.selected.append(select)  # keep track of the selected rows
            selected_weights =copy.deepcopy(entry_layer.weight[self.random_perm, :])
            layer_copy = nn.Linear(n_in, n_out, bias=entry_layer.bias is not None)  # the actual new parameter
            layer_copy.weight = nn.Parameter(selected_weights, requires_grad=False)
            if entry_layer.bias is not None:
                selected_bias = copy.deepcopy(entry_layer.bias[self.random_perm])
                layer_copy.bias = nn.Parameter(selected_bias, requires_grad=False)

            layers_class.append(layer_copy)
        #n_out = Rs_class[0]  if idx_entry < L_feat else N-R# the number of kept dimensions by the last convolution
        #idx=1
# the last eveleemnt in the number of classes which we don not shrink
        # Rs_class = [int(round(F * w[1])) for w in indices_class[i_c+1:-1]] + [indices_class[-1][1]]
        # idx=0
        for idx, layer in enumerate(remain_class):
            if isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                layer_copy = layer
            elif isinstance(layer, nn.Linear):
                n_in, n_out = (n_out, round(F*layer.out_features) if (idx<len(remain_class)-1) else layer.out_features) # last layer keep the number of classes
                layer_copy = nn.Linear(n_in, n_out, bias=(layer.bias is not None))
                #idx += 1
            layers_class.append(layer_copy)
    #if num_tries > 1:  # multiple tries in parallel
        #    nn_layers_feat = utils.construct_mcnn(sizes_feat, fct_act=nn.ReLU, num_tries=num_tries)
        #    nn_layers_class = utils.construct_mmlp_layers(sizes_feat, fct_act=nn.ReLU, num_tries=num_tries)
        #else:  # construct sequential
        #    nn_layers_feat = utils.construct_cnn_layers(sizes_feat, fct_act=nn.ReLU)
        #    nn_layers_class = utils.construct_mlp_layers(sizes_class, fct_act=nn.ReLU)

        #layers_feat.extend(nn_layers_feat.values())
        #layers_class.extend(nn_layers_feat.values())
        self.features = nn.Sequential(*layers_feat)
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*layers_class)
        # #self.network = nn.Sequential(*layers)

    #@property
    # def random_sampler(self):
        # if self.entry_in_features():
            # return self.samplerfeatures[self.idx_rs]
        # else:
            # return self.classifier[self.idx_rs]

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


    # def new_sample(self):
        # if self.idx_entry >0:  # nothing to redraw otherwise
            # self.random_sampler.draw()


    def entry_in_features(self):
        return self.idx_entry < self.n_layers_feat




class ClassifierFCNSimple(nn.Module):

    '''A Simple classifier based on a previously trained network'''

    def __init__(self, model:FCN, num_tries:int, R=None, depth=1):

        super().__init__()


        # the indices of the linear layers
        indices  =[ idx for idx, layer in enumerate(model.main) if isinstance(layer, nn.Linear)]
        size_in = [layer.in_features for layer in model.main if isinstance(layer, nn.Linear)]
        layers = []
        layers.extend(copy.deepcopy(model.main[:indices[depth]]))  # all layers before depth are simply copied
        #map(lambda p: p.requires_grad_(False), [[p for p in layer.parameters()] for layer in layers])

        self.indices = indices
        self.size_out = size_in
        self.n_tries = num_tries

        L = self.n_layers = len(self.indices)  # the total number of layers of the original model (including input and output)
        n_classes = self.n_classes = model.main[-1].out_features
        n_classes = model.main[-1].out_features
        depth_tunnel = L-depth-1  # the total depth of the tunnel, excluding input and output ?
        #if depth >=1:
        N = model.main[indices[depth]].in_features

        if depth == 0:
            width = model.main[-1].in_features
        else:
            width = N
        #else:  # depth == 0
        #N = model.main[].in_features  # take the last width, i.e. smallest one

        if R is None:  # default: remove half the neurons
            R = int(round(1/2 * width))
        elif isinstance(R, float) and 0 <R <=1:
            R = int(round(R * width))
        else:
            assert 0 < R <= N


        if depth >= 1:  # construct a parallel network
            sizes = [(N, R)] + depth_tunnel * [R] + [n_classes]
        else:
            sizes = [N] + depth_tunnel * [R] + [n_classes]

        if num_tries > 1:  # multiple tries in parallel
            nn_layers = utils.construct_mmlp_layers(sizes, fct_act=nn.ReLU, num_tries=num_tries)
        else:  # construct sequential
            nn_layers = utils.construct_mlp_layers(sizes, fct_act=nn.ReLU)

        layers.extend(nn_layers.values())
        self.network = nn.Sequential(*layers)


    def forward(self, x):

        x=x.view(x.size(0), -1)
        out = self.network(x)
        return out





class RandomSampler(nn.Module):

    def __init__(self, total, remove):
        '''in_features: tuple total / remove
        '''


        super().__init__()

        self.N = N = total
        self.R = R = remove

        self.draw()

    def forward(self, x):
        return x[:, self.random_perm]

    def draw(self):
        self.random_perm =torch.randperm(self.N)[:self.N-self.R].sort().values  # select the columns

    def extra_repr(self, **kwargs):

        return "kept features={} (/ {} total)".format(self.N-self.R, self.N)

class RandomSamplerParallel(nn.Module):

    def __init__(self, total, remove, num_tries=10):
        '''in_features: tuple total / remove
        '''


        super().__init__()

        self.N = N = total
        self.R = R = remove
        self.T = num_tries
        K = N-R  # the number of kept units

        self.random_perms =nn.Parameter(torch.cat([torch.randperm(N)[:N-R].sort().values.unsqueeze(0) for idx
                    in range(num_tries)], dim=0).view(num_tries, 1, N-R), requires_grad=False)  # select the columns


    def forward(self, x):
        x = x.unsqueeze(0).expand(self.T, -1, -1)
        return x.gather(2, self.random_perms.expand(-1, x.size(1), -1))

    def extra_repr(self, **kwargs):

        return "kept features={} (/ {} total), num_tries={}".format(self.N-self.R, self.N, self.T)

class LinearParallelMasked(nn.Module):

    def __init__(self, in_features, out_features, num_tries=10):
        '''in_features: tuple total / remove
        '''

        super().__init__()
        self.num_tries = num_tries
        self.out_features = out_features
        self.in_features = in_features
        self.N = N = in_features[0]  # total number of neurons
        self.R = R = in_features[1]  # the neurons that are removed

        # of size TxN
        self.mask = nn.Parameter(torch.ones((num_tries, N)), requires_grad=False)  # all ones, i.e. selecting, need to remove the random neurons
        random_perms =torch.cat([torch.randperm(N)[:R].view(1, -1) for idx in range(num_tries)], dim=0)  # the random choices, as many as num_tries
        self.random_perm = nn.Parameter(random_perms.unsqueeze(1), requires_grad=False)  # Tx1xP
        zeros = torch.zeros((num_tries, N))  # the filling zeros
        self.mask.scatter_(1, random_perms, zeros)
        self.mask.unsqueeze_(1)
        # size Tx1xN

        # of size num_tries x out_features x in_features , masked ! will need to
        # apply the mask again later on as well
        weight = torch.empty((self.num_tries, N, self.out_features))
        bias =torch.empty((self.num_tries, 1, self.out_features))
        init_kaiming(weight, bias, fan_in=N-R)

        self.weight = nn.Parameter(weight)
        # sizes TxNxP
        self.bias = nn.Parameter(bias)

        #self.register_parameter('weight', self.weight)
        #self.register_parameter('bias', self.bias)
        #self.register_parameter('mask', self.mask)



    def forward(self, x):

        out_last = x.unsqueeze(0).expand(self.num_tries, -1, -1)
        # size TxBxN
        #out_last = out_last.unsqueeze(0).expand(self.num_tries, -1, -1)
        #
        out_mask = self.mask * out_last  # selecting the activations
        # size TxBxN


        # performing the forward pass
        # TxBxN * TxNxP ->  TxBxP
        out_matmul = out_mask.matmul(self.weight) + self.bias
        # of size num_tries x B x out_features
        return out_matmul

    #def to(self, device):

    #    self.weight.to(device)
    #    self.bias.to(device)

    def extra_repr(self, **kwargs):

        return "in_features={} (/ {}), out_features={}, num_tries={}".format(self.N-self.R, self.N, self.out_features, self.num_tries)








def FCNHelper(num_layers, input_dim, num_classes, min_width, max_width=None,
              shape='linear', first_layer=None, last_layer=None):

    if shape == 'linear':
        max_width = 2*min_width
        widths = list(np.linspace(max_width, min_width, num_layers, dtype=int))  # need three steps
    elif shape == 'square':
        widths = num_layers * [min_width]

    if last_layer is not None:
        widths[-1] = last_layer
    if first_layer is not None:
        widths[0] = first_layer

    return FCN(input_dim, num_classes, widths)



def FCN3(input_dim, num_classes, min_width, max_width, interpolation='linear'):


    if interpolation == 'linear':
        # linear interpolation between the two extrema
        widths = list(np.linspace(max_width, min_width, 3, dtype=int))  # need three steps

    return FCN(input_dim, num_classes, widths)


