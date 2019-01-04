'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1, self).__init__()
        self.reduction = reduction
    def forward(self, output, target):
        lossvalue = torch.abs(output - target)
        if self.reduction == 'mean':
            lossvalue = lossvalue.mean()
        elif self.reduction == 'sum':
            lossvalue = lossvalue.sum() / lossvalue.size(0)
        return lossvalue

class L2(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2, self).__init__()
        self.reduction = reduction
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1)
        if self.reduction == 'mean':
            lossvalue = lossvalue.mean()
        elif self.reduction == 'sum':
            lossvalue = lossvalue.sum() / lossvalue.size(0)
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1(reduction)
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        epevalue = EPE(output, target)
        lossvalue = self.loss(output, target)
        if self.args.cuda:
            epevalue = epevalue.to('cuda:0')
            lossvalue = lossvalue.to('cuda:0')
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2(reduction)
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        epevalue = EPE(output, target)
        lossvalue = self.loss(output, target)
        if self.args.cuda:
            epevalue = epevalue.to('cuda:0')
            lossvalue = lossvalue.to('cuda:0')
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale=4, numScales=5, l_weight=0.32, norm='L1', reduction='mean'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        if reduction == 'mean':
            self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        elif reduction == 'sum':
            # Weight magnitudes are reversed from reduction=='mean'.
            # The weights below are based on PWC-Net configuration.
            self.loss_weights = torch.FloatTensor([l_weight / 64, l_weight / 32, l_weight / 16, l_weight / 4, l_weight])
        if args.cuda:
            self.loss_weights = self.loss_weights.to('cuda:0')
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1(reduction)
        else:
            self.loss = L2(reduction)

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE']

    def forward(self, output, target):
        if type(output) is tuple:
            lossvalue = 0
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                loss = self.loss(output_, target_)
                if self.args.cuda:
                    loss = loss.to('cuda:0')
                lossvalue += self.loss_weights[i]*loss
                if i == 0:
                    epevalue = EPE(output_ / self.div_flow, target_ / self.div_flow)
                    if self.args.cuda:
                        epevalue = epevalue.to('cuda:0')
        else:
            epevalue = EPE(output, target)
            lossvalue = self.loss(output, target)
            if self.args.cuda:
                epevalue = epevalue.to('cuda:0')
                lossvalue = lossvalue.to('cuda:0')
        return  [lossvalue, epevalue]
