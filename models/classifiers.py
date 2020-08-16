import torch.nn as nn
import torch
import utils
import numpy as np

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

    def __init__(self, model, try_num=1):
        '''model: a FCN object
        T: the number of tries'''
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.C = model.main[-1].out_features # the output dimension (i.e. class)
        self.N = model.main[-1].in_features  # the number of features to consider
        self.P = self.N//2  # sample half of the features
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

    def forward(self, x):
        '''return size: TxBxC'''

        B = x.size(0) # the batch size
        out_last = self.model.main[:-2](x.view(B, -1)) # also go through the ReLU (?)
        out_last = out_last.unsqueeze(0).expand(self.T, -1, -1)
        out_mask = self.mask.unsqueeze(1) * out_last
        # of size TxBxP


        # BxTxC = BxTxCxP * BxTxPx1
        #out_einsum = torch.einsum("abcd,abde->abce",
        #                   self.weight.unsqueeze(1).expand(-1, out_rand.size(0),  -1, -1),
        #                   out_rand) #+ self.b
        out_matmul = out_mask.matmul(self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)

        return out_matmul  # TxBxC



def FCN3(input_dim, num_classes, min_width, max_width, interpolation='linear'):


    if interpolation == 'linear':
        # linear interpolation between the two extrema
        widths = list(np.linspace(max_width, min_width, 3, dtype=int))  # need three steps

    return FCN(input_dim, num_classes, widths)
