import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, input_channels = 1):
        super(Extractor, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2, stride=2, padding =0),
                                     nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2, stride=2, padding =0),
                                     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(100),
                                     nn.Flatten()
                                    )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        
        self.predictor = nn.Sequential(nn.Linear(100*64, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 4)
                                    )
        

    def forward(self, x):
        x = self.predictor(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(nn.Linear(100*64, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid()
                                    )
        
    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x

from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None