# 
# @rajp
#

# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html

import torch
import torchsummary
torch.manual_seed(13)


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def run1():
    """ modules and parameters """
    global TinyModel
        
    tinymodel = TinyModel()

    print('The model:')
    #print(tinymodel)
    print(torchsummary.summary(tinymodel.to("cuda"), (1, 100)))
    """
        x = self.softmax(x)
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Linear-1               [-1, 1, 200]          20,200
                    ReLU-2               [-1, 1, 200]               0
                    Linear-3                [-1, 1, 10]           2,010
                Softmax-4                [-1, 1, 10]               0
        ================================================================
        Total params: 22,210
        Trainable params: 22,210
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.00
        Forward/backward pass size (MB): 0.00
        Params size (MB): 0.08
        Estimated Total Size (MB): 0.09
        ----------------------------------------------------------------
        None

    """
    print('\n\nJust one layer:')
    print(tinymodel.linear2)

    print('\n\nModel params:')
    for param in tinymodel.parameters():
        #print(param)
        print(param.shape)

    print('\n\nLayer params:')
    for param in tinymodel.linear2.parameters():
        #print(param)
        print(param.shape)

if __name__ == "__main__":
    run1()