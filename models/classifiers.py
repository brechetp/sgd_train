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
        self.fcn_idx = [idx for (idx, layer) in enumerate(self.model.classifier) if not isinstance(layer, nn.Dropout)]  # remove the dropout layers



        #random_perm = random_perms.unsqueeze(1) # Tx1xP
        # random_perm is now TxP
        self.linear_mask = LinearMasked((Ns[0], Rs[0]), num_classes, n_try=num_tries)
        self.shallow_net = nn.Sequential(
            LinearMasked((Ns[1], Rs[1]), (Ns[0]-Rs[0]), n_try=num_tries),
            nn.ReLU(inplace=True),
            MultiLinear(num_tries, (Ns[0]-Rs[0]), num_classes)
        )

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

def init_kaiming(weight, bias, fan_in=None):
    '''Kaiming (uniform) initialization with for parameters with parallel channels'''

    a = math.sqrt(5.0)  # initialization of a linear model in pytorch

    if fan_in is None:
        fan_in = weight.size(1)  # the neurons that are kept (fan_in)
    gain = nn.init.calculate_gain('relu', a)  # kaiming uniform init
    std = gain / (math.sqrt(fan_in))
    bound = math.sqrt(3.0) * std
    nn.init.uniform_(weight, -bound, bound)
    bound = 1. / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


class MultiLinear(nn.Module):
    '''Define a parallel linear layer'''


    def __init__(self, n_tries, in_features, out_features):

        super().__init__()

        weight = torch.empty((n_tries, in_features, out_features))
        bias = torch.empty((n_tries, 1, out_features))

        init_kaiming(weight, bias)

        # size of weight: TxNxD
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        ''' Parallel matrix multiplication '''
        # size of x: TxBxN

        out_matmul = x.matmul(self.weight) + self.bias
        # output of size TxBxD

        return out_matmul


class LinearMasked(nn.Module):

    def __init__(self, in_features, out_features, n_try=10):
        '''in_features: tuple total / remove
        '''

        super().__init__()
        self.n_try = n_try
        self.out_features = out_features
        self.in_features = in_features
        self.N = N = in_features[0]  # total number of neurons
        self.R = R = in_features[1]  # the neurons that are removed

        self.mask = nn.Parameter(torch.ones((n_try, N)), requires_grad=False)  # all ones, i.e. selecting, need to remove the random neurons
        random_perms =torch.cat([torch.randperm(N)[:R].view(1, -1) for idx in range(n_try)], dim=0)  # the random choices, as many as n_try
        self.random_perm = nn.Parameter(random_perms.unsqueeze(1), requires_grad=False)  # Tx1xP
        zeros = torch.zeros((n_try, N))  # the filling zeros
        self.mask.scatter_(1, random_perms, zeros)
        self.mask.unsqueeze_(1)
        # size Tx1xN

        # of size n_try x out_features x in_features , masked ! will need to
        # apply the mask again later on as well
        weight = torch.empty((self.n_try, N, self.out_features))
        bias =torch.empty((self.n_try, 1, self.out_features))
        init_kaiming(weight, bias, fan_in=N-R)

        self.weight = nn.Parameter(weight)
        # sizes TxNxP
        self.bias = nn.Parameter(bias)

        #self.register_parameter('weight', self.weight)
        #self.register_parameter('bias', self.bias)
        #self.register_parameter('mask', self.mask)



    def forward(self, x):

        out_last = x.unsqueeze(0).expand(self.n_try, -1, -1)
        # size TxBxN
        #out_last = out_last.unsqueeze(0).expand(self.n_try, -1, -1)
        #
        out_mask = self.mask * out_last  # selecting the activations
        # size TxBxN


        # performing the forward pass
        # TxBxN * TxNxP ->  TxBxP
        out_matmul = out_mask.matmul(self.weight) + self.bias
        # of size n_try x B x out_features
        return out_matmul

    #def to(self, device):

    #    self.weight.to(device)
    #    self.bias.to(device)

    def extra_repr(self, **kwargs):

        return "in_features={}, removed: {}".format(self.N, self.R)








def FCNHelper(num_layers, input_dim, num_classes, min_width, max_width=None, shape='linear', last_layer=None):

    if shape == 'linear':
        max_width = 2*min_width
        widths = list(np.linspace(max_width, min_width, num_layers, dtype=int))  # need three steps
    elif shape == 'square':
        widths = num_layers * [min_width]

    if last_layer is not None:
        widths[-1] = last_layer

    return FCN(input_dim, num_classes, widths)



def FCN3(input_dim, num_classes, min_width, max_width, interpolation='linear'):


    if interpolation == 'linear':
        # linear interpolation between the two extrema
        widths = list(np.linspace(max_width, min_width, 3, dtype=int))  # need three steps

    return FCN(input_dim, num_classes, widths)


