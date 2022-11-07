#
# @rajp
#

# https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
# https://www.youtube.com/watch?v=Dk88zv1KYMI&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=8 

# Production inference deployment with Pytorch
#  Evaludaiton mode for pytorch models
#  Torchscript
#  Torchscript and C++
#  Deploying with Torchserve

import torch
import torchvision
import matplotlib.pyplot as plt 
import numpy as np 
import torchsummary 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from lab5 import LeNet5

torch.manual_seed(876)



def load_model():
    # No matter which deployment method you use, first put model in eval mode.
    # this turns off autograd and other modes that are not needed in eval mode
    # autograd is expensive is memory and compute
    # dropout are only active during training time. 
    # batch norm is turned off
    model = LeNet5()
    model.load_state_dict(torch.load("./model_20221105_203254_epoch=1.pth"))
    model.eval() # it is alias for model.train(False)
    return model

def run2(model):
    """ torch.jit.trace() """
    """ torch.jit.trace() requires sample input
        does NOT preserve control flow
        works with just about any code
    """
    sample_input = torch.rand(3,32,32).unsqueeze(0)
    traced_model = torch.jit.trace(model, sample_input) 
    print(traced_model)
    """
        LeNet5(
        original_name=LeNet5
        (featureExtractor): Sequential(
            original_name=Sequential
            (0): Conv2d(original_name=Conv2d)
            (1): AvgPool2d(original_name=AvgPool2d)
            (2): Conv2d(original_name=Conv2d)
            (3): AvgPool2d(original_name=AvgPool2d)
        )
        (classifier): Sequential(
            original_name=Sequential
            (0): Linear(original_name=Linear)
            (1): ReLU(original_name=ReLU)
            (2): Linear(original_name=Linear)
            (3): ReLU(original_name=ReLU)
            (4): Linear(original_name=Linear)
        )
        )
    """
    print(traced_model.code)
    """
        def forward(self,
            x: Tensor) -> Tensor:
            classifier = self.classifier
            featureExtractor = self.featureExtractor
            input = torch.flatten((featureExtractor).forward(x, ), 1)
            return (classifier).forward(input, )

    """
    # if the model had any conditional flows, they would be gone in the above repr as 
    # trace just trarces through compile time constant values depending on sample inputs

    pred = traced_model(sample_input)
    print(pred.shape)

def run1(model):
    """ torch.jit.script() """
    # torchscript is: a statically typed subset of Python meant for ML
    # meant for consumption by the PyTorch JIT compiler that performs runtime opt on Torchscript model (operator fusing, batching matmults)
    # perferred method for serializing    
    # to convert models to torchscript, use torch.jit.trace() or torch.jit.script()
    # trace mode: rapid tests
    # script mode: production
    # saved model has both weights and graph
    
    """ torch.jit.script() preserves control flow
        converts model by analyzing control flow
        accommodates list, dict, typle
        may not cover 100% of operators - some models wont be convertable
    """
    scripted_model = torch.jit.script(model)
    scripted_model.save("my_scripted_module.pt")
    # this saves both computaiton graph and weights - no need to ship python file for production

    #inference
    model = torch.jit.load("my_scripted_module.pt")
    pred = model(torch.rand(4,3,32,32))
    print(pred.shape)
    print(model)
    """
        RecursiveScriptModule(
        original_name=LeNet5
        (featureExtractor): RecursiveScriptModule(
            original_name=Sequential
            (0): RecursiveScriptModule(original_name=Conv2d)
            (1): RecursiveScriptModule(original_name=AvgPool2d)
            (2): RecursiveScriptModule(original_name=Conv2d)
            (3): RecursiveScriptModule(original_name=AvgPool2d)
        )
        (classifier): RecursiveScriptModule(
            original_name=Sequential
            (0): RecursiveScriptModule(original_name=Linear)
            (1): RecursiveScriptModule(original_name=ReLU)
            (2): RecursiveScriptModule(original_name=Linear)
            (3): RecursiveScriptModule(original_name=ReLU)
            (4): RecursiveScriptModule(original_name=Linear)
        )
    """
    print(model.code)
    """
        def forward(self,
            x: Tensor) -> Tensor:
            featureExtractor = self.featureExtractor
            x0 = (featureExtractor).forward(x, )
            x1 = torch.flatten(x0, 1)
            classifier = self.classifier
            return (classifier).forward(x1, )
    """


    """
    you can load torchscript into C++ without any other python dependencies
    minimum cmake file for LibTorch is as follows: C++ 14 or higher

    cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
    project(custom_ops)

    find_package(Torch REQUIRED)

    add_Executable(example-app example-app.cpp)
    target_link_libraries(Example-app "${TORCH_LIBRARIES}")
    set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
    """


if __name__ == "__main__":
    model = load_model()
    run1(model)
    run2(model)