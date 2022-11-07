#
# @rajp
#

# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

import torch
torch.manual_seed(8623)

def run1():
    """ tensor variables"""
    x = torch.rand(2,2)
    z = torch.zeros(2,2)
    o = torch.ones((2,2), dtype=torch.int16)
    print(x)
    print(z)
    print(o)
    print(torch.svd(x))
    print(torch.det(x))


def run2():
    """ autograd """
    x = torch.rand(1, 10)
    prev_h = torch.rand(1, 20)
    # if the requires_grad is not set, loss.backward() will fail
    # when torch.nn.Module is used, all tensors are managed by ptrch and their grad is set to true automatically
    w_x = torch.rand((20, 10), requires_grad=True)
    w_h = torch.rand((20, 20), requires_grad=True)

    t0 = torch.mm(w_x, x.t()) # matmul
    t1 = torch.mm(w_h, prev_h.t()) # matmul
    t2 = t0 + t1
    next_h = torch.tanh(t2)
    print(next_h)

    loss = next_h.sum()
    print(loss)

    loss.backward() # -> performs autograd

if __name__ == "__main__":
    run1()
    run2()
