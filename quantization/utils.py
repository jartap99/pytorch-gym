#
# @rajp
# 

import torch
import torchsummary
import torchvision

import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+"/../intro/")

from lab5 import LeNet5
from lab5 import load_data

import timeit

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
""" as on nov 20222 cuda gives error
Pytorch doe snot support quant on GPU 
NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'quantized::linear_dynamic' is only available for these backends: [CPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].
"""
device = torch.device("cpu")
print("device: ", device)
checkpoint = "model_20221108_223519_epoch=9.pth"
    

def print_model_parameters(model, shape=True):
    print("*****")
    for param in model.parameters():
        if shape is True:
            print(param.shape, param.dtype)    
        else:
            print(param)

def print_model_size(model, label=""):
    print("*****")
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("Model size: ", label, " \t Size: ", size/1e3, "(KB)" )
    os.remove("temp.p")
    return size


def accuracy(model, dataLoader, withLoss=True):
    if (withLoss): 
        criterion = torch.nn.CrossEntropyLoss()
    vloss = 0
    misses = 0
    for j, vdata in enumerate(dataLoader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs.to(device)) 
        if (withLoss): 
            vloss_ = criterion(voutputs, vlabels.to(device))
            vloss += vloss_
        d = vlabels - torch.argmax(voutputs,1)
        d = list(torch.nonzero(d).squeeze().size())
        if len(d)>0:
            misses += d[0]
        
    acc = 1-(misses/(len(dataLoader)*4))
    return ( vloss, acc)
