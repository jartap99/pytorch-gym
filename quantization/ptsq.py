#
# @rajp
# 

from utils import * 
import copy

""" post training static quantization
    https://pytorch.org/blog/introduction-to-quantization-on-pytorch/ 
    https://pytorch.org/docs/stable/quantization.html
    https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html 

"""
    

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.featureExtractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5*5*16, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.featureExtractor(x)
        # either add flatten here on within the single sequential block of model
        x = torch.flatten(x, 1) # flatten everything but batch dimension
        logits = self.classifier(x)
        #probabilities = torch.nn.functional.softmax(logits, dim=0)
        #label = torch.argmax(probabilities, dim=1)
        #return label
        logits = self.dequant(logits)
        return logits

                      
def step1(qat=False):
    """
    Step 1: (a) Add quant dequant stubs to origin al model as shown in redefined class above
            (b) Fuse Conv+BN, Conv+ReLU and Conv+BN+Relu modules prior to quantization
            This operation does not change the numerics
    
    # torch summary only works on GPU an with current version of pytorch, dynamic quant does not work on gpu 
    #torchsummary.summary(model_fp32, (3,32,32))

    """
    global cwd, checkpoint
    
    model_fp32 = LeNet5().to(device)
    model_fp32.load_state_dict(torch.load(cwd+"/../intro/"+checkpoint))
    print_model_parameters(model_fp32)
    model_fp32_original = copy.deepcopy(model_fp32) # clone 
    
    #print(model_fp32)
    #print(model_fp32.featureExtractor[0])
    """
        LeNet5(
            (featureExtractor): Sequential(
                (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
                (1): ReLU()
                (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
                (4): ReLU()
                (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (classifier): Sequential(
                (0): Linear(in_features=400, out_features=120, bias=True)
                (1): ReLU()
                (2): Linear(in_features=120, out_features=84, bias=True)
                (3): ReLU()
                (4): Linear(in_features=84, out_features=10, bias=True)
            )
            (quant): QuantStub()
            (dequant): DeQuantStub()
            )
    """
    if qat is False:
        model_fp32.eval()
    
    #for m in model_fp32.modules():
    #    print(type(m))
    torch.ao.quantization.fuse_modules(model_fp32.featureExtractor, \
            [["0", "1"], ["3", "4"]], \
            inplace=True) # fuse conv + relu
    #print(model_fp32)
    #print(model_fp32.featureExtractor[0])
    """
        LeNet5(
            (featureExtractor): Sequential(
                (0): ConvReLU2d(
                (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
                (1): ReLU()
                )
                (1): Identity()
                (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (3): ConvReLU2d(
                (0): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
                (1): ReLU()
                )
                (4): Identity()
                (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (classifier): Sequential(
                (0): Linear(in_features=400, out_features=120, bias=True)
                (1): ReLU()
                (2): Linear(in_features=120, out_features=84, bias=True)
                (3): ReLU()
                (4): Linear(in_features=84, out_features=10, bias=True)
            )
            (quant): QuantStub()
            (dequant): DeQuantStub()
            )
    """
    return (model_fp32_original, model_fp32)

def step2(model_fp32):
    model_fp32.qconfig = torch.ao.quantization.default_qconfig
    qmodel = torch.ao.quantization.prepare(model_fp32, inplace=False)
    print("model_fp32.qconfig: ", model_fp32.qconfig)
    """
    model_fp32.qconfig:  QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, quant_min=0, quant_max=127){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric){})
    """
    print("qmodel.qconfig: ", qmodel.qconfig)
    """
    qmodel.qconfig:  QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, quant_min=0, quant_max=127){'factory_kwargs': <function add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7fcedd218160>}, weight=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric){'factory_kwargs': <function add_module_to_qconfig_obs_ctr.<locals>.get_factory_kwargs_based_on_module_device at 0x7fcedd218160>})
    """
    if True:
        # per channel quantization
        qmodel.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        print("qmodel.qconfig: ", qmodel.qconfig)
        """
        qmodel.qconfig:  QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.HistogramObserver'>, reduce_range=True){}, weight=functools.partial(<class 'torch.ao.quantization.observer.PerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){})
        """
    print('Post Training Quantization Prepare: Inserting Observers')
    print(qmodel)
    """
        LeNet5(
            (featureExtractor): Sequential(
                (0): ConvReLU2d(
                (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
                (1): ReLU()
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
                )
                (1): Identity()
                (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (3): ConvReLU2d(
                (0): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
                (1): ReLU()
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
                )
                (4): Identity()
                (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (classifier): Sequential(
                (0): Linear(
                in_features=400, out_features=120, bias=True
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
                )
                (1): ReLU()
                (2): Linear(
                in_features=120, out_features=84, bias=True
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
                )
                (3): ReLU()
                (4): Linear(
                in_features=84, out_features=10, bias=True
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
                )
            )
            (quant): QuantStub(
                (activation_post_process): MinMaxObserver(min_val=inf, max_val=-inf)
            )
            (dequant): DeQuantStub()
            )
    """
    return qmodel

if __name__ == "__main__":

    (model_fp32_original, model_fp32) = step1()
    
    (trainDataSet, trainDataLoader, testDataSet, testDataLoader, classes) = load_data("./../intro/data/cifar10/", writer=None)

    # Baseline accuracy: Comparing model acuracy before and after fuse model
    (vloss_fp32o, acc_fp32o) = accuracy(model_fp32_original, testDataLoader, withLoss=True)
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    
    print ("1 LOSSES:   Model-1: ", vloss_fp32o, "Model-2: ", vloss_fp32)
    print ("1 ACCURACY: Model-1: ", acc_fp32o, "Model-2: ", acc_fp32)

    """
        1 LOSSES:   Model-1:  tensor(3284.2749, grad_fn=<AddBackward0>) Model-2:  tensor(3284.2727, grad_fn=<AddBackward0>)
        1 ACCURACY: Model-1:  0.5913999999999999 Model-2:  0.5929   
    """

    # prepare
    qmodel = step2(model_fp32)

    # Step 3 - Calibrate
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    (vloss_q, acc_q) = accuracy(qmodel, testDataLoader, withLoss=True)
       
    print ("2 LOSSES:   Model-1: ", vloss_fp32, "Model-2: ", vloss_q)
    print ("2 ACCURACY: Model-1: ", acc_fp32, "Model-2: ", acc_q)

    """
        2 LOSSES:   Model-1:  tensor(3284.2766, grad_fn=<AddBackward0>) Model-2:  tensor(3284.2803, grad_fn=<AddBackward0>)
        2 ACCURACY: Model-1:  0.5903 Model-2:  0.5926
    """
    # Step 4 - convert
    torch.ao.quantization.convert(qmodel, inplace=True)
    print('Post Training Quantization: Convert done')
    print(qmodel)
    """
        LeNet5(
            (featureExtractor): Sequential(
                (0): QuantizedConvReLU2d(3, 6, kernel_size=(5, 5), stride=(1, 1), scale=0.07177824527025223, zero_point=0)
                (1): Identity()
                (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
                (3): QuantizedConvReLU2d(6, 16, kernel_size=(5, 5), stride=(1, 1), scale=0.15653109550476074, zero_point=0)
                (4): Identity()
                (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            (classifier): Sequential(
                (0): QuantizedLinear(in_features=400, out_features=120, scale=0.22000500559806824, zero_point=62, qscheme=torch.per_tensor_affine)
                (1): ReLU()
                (2): QuantizedLinear(in_features=120, out_features=84, scale=0.11531691998243332, zero_point=53, qscheme=torch.per_tensor_affine)
                (3): ReLU()
                (4): QuantizedLinear(in_features=84, out_features=10, scale=0.1604081392288208, zero_point=59, qscheme=torch.per_tensor_affine)
            )
            (quant): Quantize(scale=tensor([0.0157]), zero_point=tensor([64]), dtype=torch.quint8)
            (dequant): DeQuantize()
            )
    """

    # Model sizes
    print_model_size(model_fp32_original, label="model_fp32_original")
    print_model_size(model_fp32, label="model_fp32")
    print_model_size(qmodel, label="qmodel")
    """
        Model size:  model_fp32_original         Size:  251.235 (KB)
        Model size:  model_fp32                  Size:  251.427 (KB)
        Model size:  qmodel                      Size:  69.659 (KB)
    """

    # Check acuracy after conversion
    (vloss_fp32, acc_fp32) = accuracy(model_fp32, testDataLoader, withLoss=True)
    (vloss_q, acc_q) = accuracy(qmodel, testDataLoader, withLoss=True)
       
    print ("3 LOSSES:   Model-1: ", vloss_fp32, "Model-2: ", vloss_q)
    print ("3 ACCURACY: Model-1: ", acc_fp32, "Model-2: ", acc_q)

    """
    WITH PER TENSOR QUANT
        3 LOSSES:   Model-1:  tensor(3284.2732, grad_fn=<AddBackward0>) Model-2:  tensor(3286.6116)
        3 ACCURACY: Model-1:  0.5986 Model-2:  0.5952

    WITH PER CHANNEL QUANT
        3 LOSSES:   Model-1:  tensor(3284.2732, grad_fn=<AddBackward0>) Model-2:  tensor(3286.6116)
        3 ACCURACY: Model-1:  0.5986 Model-2:  0.5952
    """