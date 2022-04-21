import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

# flexible CNN architecture       
class MyCNNflex_Regr(nn.Module):
    def __init__(self,  input_shape, num_classes=3, drop_p_conv=.2, drop_p_fc=.5, kernel_size=51, stride=2, pool_size=1, filters=32, num_layers=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.drop_p_conv = drop_p_conv
        self.drop_p_fc = drop_p_fc
        self.filters = filters
        
        # block of conv, relu, drop
        def one_conv(in_c, out_c, kernel_size, stride, drop_p):
            conv = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                nn.ReLU(),
                nn.Dropout(drop_p)
            )
            return conv
            
        layers = [] # all conv layers 
        layers.append( one_conv(input_shape[1], self.filters, self.kernel_size, self.stride, self.drop_p_conv) )
        self.outshape = int( (input_shape[2]-self.kernel_size)/self.stride +1 )
        for i in range(num_layers-1):
            block = one_conv(self.filters, self.filters, self.kernel_size, self.stride, self.drop_p_conv)
            layers.append(block)
            self.outshape = int( (self.outshape-self.kernel_size)/self.stride +1)
            if self.pool_size > 1:
                layers.append( nn.MaxPool1d(2,stride=self.pool_size) )
                self.outshape = int(self.outshape/self.pool_size)
        self.features = nn.Sequential(*layers)
        
        fc = [] # all fc layers
        fc.append( nn.Dropout(self.drop_p_fc) )
        fc.append( nn.Linear(self.outshape*self.filters, self.filters) ) 
        fc.append( nn.Linear(self.filters, num_classes) )
        self.fc_block = nn.Sequential(*fc)     
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc_block(x)