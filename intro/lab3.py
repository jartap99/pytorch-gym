#
# @rajp
#

# https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html

import torch
import math
import matplotlib.pyplot as plt

torch.manual_seed(876)

def run1():
    """ autograd """
    a = torch.linspace(0, 2*math.pi, 25, requires_grad=True)
    print(a)
    b = torch.sin(a)

    plt.figure(1)
    plt.plot(a.detach(), b.detach())
    # detach the tensor from autograd and then plot it to give it to matplotlib or numpy
    plt.savefig("./lab3_fig0.png")
    plt.close()

    print(b)

    c = 2 * b
    print(c)

    d = c + 1
    print(d)

    out = d.sum()
    print(out)
    # x -> sin(x) -> 2*sin(x) -> 2*sin(x) + 1 -> sum()
    # now we are interested in backprop of this computation graph

    print("*"*10)
    print(d.grad_fn)
    # <AddBackward0 object at 0x7fe014912230>

    print(d.grad_fn.next_functions)
    # ((<MulBackward0 object at 0x7fe014912b60>, 0), (None, 0))

    print(d.grad_fn.next_functions[0][0].next_functions)
    # ((<SinBackward0 object at 0x7fe014912b60>, 0), (None, 0))

    print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
    # ((<AccumulateGrad object at 0x7fe014912230>, 0),)

    print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
    # ()


    print("*"*10)
    print(d.grad_fn)
    # <AddBackward0 object at 0x7f72adc55000>
    # ((<MulBackward0 object at 0x7f72adc55030>, 0), (None, 0))
    # ((<SinBackward0 object at 0x7f72adc55030>, 0), (None, 0))
    # ((<AccumulateGrad object at 0x7f72adc55000>, 0),)
    # ()

    print("*"*10)
    print(c.grad_fn)
    # <MulBackward0 object at 0x7f526339ed40>

    print("*"*10)
    print(b.grad_fn)
    # <SinBackward0 object at 0x7f001f8a2c50>

    print("*"*10)
    print(a.grad_fn)
    # None

    print("*"*10)
    print(b.grad, c.grad)
    # None None

    # everything described above is gradient funcs and how they are denoted in pytorch
    # now lets compute gradiesnt
    # gradiesnt are calculated only for leaf nodes of computation graph.
    # they are not computed for intermediate nodes. 
    c.retain_grad()
    out.backward()

    print("*"*10)
    print(b.grad, c.grad)
    # These ar eno longer None. Try commenting line 83 and check ... they will be None
    
    
    plt.figure(2)
    plt.plot(a.detach(), a.grad.detach())
    plt.savefig("./lab3_fig1.png")
    plt.close()

def run2():
    """ autograd in trianing """
    BATCH_SIZE = 16
    DIM_IN = 1000
    HIDDEN_SIZE = 100
    DIM_OUT = 10

    class TinyModel(torch.nn.Module):

        def __init__(self):
            super(TinyModel, self).__init__()

            self.layer1 = torch.nn.Linear(1000, 100)
            self.relu = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(100, 10)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

    model = TinyModel()

    print("*"*10)
    print("initial weight shape: ", model.layer2.weight.shape)
    print("initial weight: ", model.layer2.weight[0][0:10]) # just a small slice
    print("initial weight grad: ", model.layer2.weight.grad) # None - no training yet

    # 1 step of training and backprop
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    pred = model(some_input)
    loss = (ideal_output - pred).pow(2).sum()
    print("loss: ", loss)
    loss.backward()
    
    print("*"*10)
    print("weight after loss shape: ", model.layer2.weight.shape)
    print("weight after loss: ", model.layer2.weight[0][0:10])  # weights do not change as the optimizer update did not happen. gradiesnt are calculated already
    print("weight after loss grad: ", model.layer2.weight.grad.shape) # torch.Size([10, 100])
    print("weight after loss grad: ", model.layer2.weight.grad[0][0:10]) # grads are computed

    # run optimizer step - this step updates weights
    optimizer.step()

    print("*"*10)
    print("weight after loss shape: ", model.layer2.weight.shape)
    print("weight after loss: ", model.layer2.weight[0][0:10]) 
    print("weight after loss grad: ", model.layer2.weight.grad.shape) # torch.Size([10, 100])
    print("weight after loss grad: ", model.layer2.weight.grad[0][0:10]) # grads are same from before optimizer step

    # demonstrating gradient explosion if optimizer is not seroed out
    for i in range(5):
        pred = model(some_input)
        loss = (ideal_output - pred).pow(2).sum()
        loss.backward()
        print("i: ", i, model.layer2.weight.grad[0][:5])
        optimizer.step()
        """
            i:  0 tensor([11.5412,  3.7158,  3.5801,  5.8046, -0.2148])
            i:  1 tensor([12.7425,  4.2184,  4.5420,  6.2478, -0.9201])
            i:  2 tensor([ 1.9417, -6.2815, -1.0454,  3.1178, -2.3064])
            i:  3 tensor([ -9.8013, -19.6989,  -8.8428,   1.3888,  -2.3483])
            i:  4 tensor([ -6.4179, -20.3264,  -8.9991,   5.0063,  -0.0990])
        """ 
        optimizer.zero_grad() # DO NOT FORGET THIS STEP
        """
            i:  0 tensor([11.5412,  3.7158,  3.5801,  5.8046, -0.2148])
            i:  1 tensor([ 1.2014,  0.5026,  0.9618,  0.4433, -0.7053])
            i:  2 tensor([ 1.9507,  1.4353,  1.2873,  1.0689, -0.0353])
            i:  3 tensor([1.5965, 1.0298, 1.0573, 1.0120, 0.0720])
            i:  4 tensor([1.1809, 0.5886, 0.8135, 0.8159, 0.0791])
        """
        
def run3():
    """ turn autograd on and off """
    a = torch.ones((2,3), requires_grad=True)
    print(a)
    # tensor([[1., 1., 1.],
    #         [1., 1., 1.]], requires_grad=True)
    
    b1 = 2 * a
    print(b1)
    # tensor([[2., 2., 2.],
    #         [2., 2., 2.]], grad_fn=<MulBackward0>)

    a.requires_grad = False
    b2 = 2*a 
    print(b2)
    # tensor([[2., 2., 2.],
    #         [2., 2., 2.]])

    # can use torch.no_grad() to disable autograd temporarily
    b = torch.rand((2,3), requires_grad=True)
    with torch.no_grad():
        c = a + b
    print(c) # grad turned off

    @torch.no_grad()
    def oper (x, y):
        return x+y
    
    c2 = oper(a, b)
    print(c2)

    # similarly there is torch.enable_grad()
    # do not perform in-place operations on tensors with autograd on 

def run4():
    """ autograd profiler """
    device = torch.device("cpu")
    run_on_cpu = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        run_on_gpu = True

    x = torch.randn(2, 3, requires_grad=True)
    y = torch.rand(2, 3, requires_grad=True)
    z = torch.ones(2, 3, requires_grad=True)

    with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
        for _ in range(1000):
            z = (z / x) * y

    print(prf.key_averages().table(sort_by='self_cpu_time_total'))     
    """
        WARNING:2022-11-05 16:35:02 8292:8292 init.cpp:111] function cbapi.getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
        WARNING:2022-11-05 16:35:02 8292:8292 init.cpp:112] CUPTI initialization failed - CUDA profiler activities will be missing
        INFO:2022-11-05 16:35:02 8292:8292 init.cpp:114] If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti
        STAGE:2022-11-05 16:35:02 8292:8292 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
        STAGE:2022-11-05 16:35:04 8292:8292 ActivityProfilerController.cpp:300] Completed Stage: Collection
        -------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
        -------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
            aten::mul        50.63%      11.427ms        50.63%      11.427ms      11.648us       1.092ms        49.23%       1.092ms       1.113us           981
            aten::div        48.03%      10.840ms        48.03%      10.840ms      11.061us       1.126ms        50.77%       1.126ms       1.149us           980
            aten::mul         0.70%     157.000us         0.70%     157.000us       8.263us       0.000us         0.00%       0.000us       0.000us            19
            aten::div         0.65%     147.000us         0.65%     147.000us       7.350us       0.000us         0.00%       0.000us       0.000us            20
        -------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
        Self CPU time total: 22.571ms
        Self CUDA time total: 2.218ms
    """

def run5():
    # autograd runs with the principle of jacobians and chain-rule matmul of jacobians
    # so if we provide a vector to the backward() call, it requires user to provide a vector of initial derivatives with it
    # if y is a vector, y.backward(v) where v is jacobian of y must be provided. 
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) # stand-in for gradients
    y.backward(v)

    print(x.grad)

def run6():
    """ high level api to calculate jacobian, hessian, etc. """
    def exp_adder(x, y):
        return 2 * x.exp() + 3 * y

    inputs = (torch.rand(1), torch.rand(1)) # arguments for the function
    print(inputs)
    # (tensor([0.8825]), tensor([0.6023]))

    a = torch.autograd.functional.jacobian(exp_adder, inputs)
    print(a)
    # (tensor([[4.8337]]), tensor([[3.]]))

    a = torch.autograd.functional.hessian(exp_adder, inputs)
    print(a)
    # ((tensor([[4.8337]]), tensor([[0.]])), (tensor([[0.]]), tensor([[0.]])))

    inputs = (torch.rand(3), torch.rand(3)) # arguments for the function
    a = torch.autograd.functional.jacobian(exp_adder, inputs)
    print(a)
    # can also calculate vjp (vector jacobian product), jvp hvp, vhp, etc.

if __name__ == "__main__":
    #run1()
    #run2()
    #run3()
    #run4()
    #run5()
    run6()