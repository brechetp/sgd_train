import torch.nn as nn
import torch
import numpy as np
import models.pretrained
import utils
import math

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
                                    # the VGG model
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
        self.linear_mask = LinearMasked((Ns[0], Rs[0]), num_classes, num_tries=num_tries)
        self.shallow_net = utils.contruct_mmlp_net([(Ns[1], Rs[1]), Rs[0], num_classes], num_tries=num_tries)
        #nn.Sequential(
        #    LinearMasked((Ns[1], Rs[1]), (Rs[0]), num_tries=num_tries),
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


class LinearMasked(nn.Module):

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


