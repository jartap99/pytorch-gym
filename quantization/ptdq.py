#
# @rajp
# 

from utils import * 


def latency ():
    # Step 4 of 5: Compare model latencies
    SETUP_CODE = '''
import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+"/../intro/")
checkpoint = "model_20221108_223519_epoch=9.pth"
import torch
from lab5 import LeNet5
device = torch.device("cpu")
model_fp32 = LeNet5().to(device)
model_fp32.load_state_dict(torch.load(cwd+"/../intro/"+checkpoint))    
model_int8 = torch.quantization.quantize_dynamic(model=model_fp32, 
                        qconfig_spec={torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.Linear},
                        dtype=torch.qint8)
some_input = torch.rand(1, 3, 32, 32)
    '''
    TEST_CODE = '''
model_fp32.forward(some_input)
    '''
    TEST_CODE_INT8 = '''
model_int8.forward(some_input)
    '''

    time_fp32 = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE_INT8, repeat=3, number=10000)
    time_int8 = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE_INT8, repeat=3, number=10000)

    print("time_fp32: ", time_fp32)
    print("time_int8: ", time_int8)
    print("\n{0:.2f} times faster".format(time_fp32[1]/time_int8[1]))


def ptdq():
    """ post training dynamic quantization
    https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html
    """
    global cwd, checkpoint
    
    # Step 1 of 5: Setup
    model_fp32 = LeNet5().to(device)
    model_fp32.load_state_dict(torch.load(cwd+"/../intro/"+checkpoint))
    print_model_parameters(model_fp32)
    # torch summary only works on GPU an with current version of pytorch, dynamic quant does not work on gpu 
    #torchsummary.summary(model_fp32, (3,32,32))

    # Step 2 of 5: Quantization
    model_int8 = torch.quantization.quantize_dynamic(model=model_fp32, 
                        qconfig_spec={torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.Linear},
                        dtype=torch.qint8)
    print_model_parameters(model_int8)
    
    # Step 3 of 5: Compare model sizes
    size_fp32 = print_model_size(model_fp32, label="model_fp32")
    size_int8 = print_model_size(model_int8, label="model_int8")
    print("\n{0:.2f} times smaller".format(size_fp32/size_int8))
    return (model_fp32, model_int8)

if __name__ == "__main__":
    (model_fp32, model_int8) = ptdq()

    (trainDataSet, trainDataLoader, testDataSet, testDataLoader, classes) = load_data("./../intro/data/cifar10/", writer=None)
    
    latency()
    
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    (vloss_int8, acc_int8) = accuracy(model_int8, testDataLoader, withLoss=True)  
    print ("LOSSES:   Model-1: ", vloss_fp32, "Model-2: ", vloss_int8)
    print ("ACCURACY: Model-1: ", acc_fp32, "Model-2: ", acc_int8)
    
    """
        LOSSES:   Model-1:  tensor(3284.2734, grad_fn=<AddBackward0>) Model-2:  tensor(3282.9954)
        ACCURACY: Model-1:  0.5942000000000001 Model-2:  0.5920000000000001
    """