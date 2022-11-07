#
# @rajp
#

# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

import torch
import math
torch.manual_seed(876)

def run1():
    """ tensor variables """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creation of tensors
    e = torch.empty((2,2), dtype=torch.int16, device = None) # like malloc
    z = torch.zeros(2,2)
    o = torch.ones((2,2), dtype=torch.int16, device=device)
    r = torch.rand((2,2))
    print(z)
    print(o)
    print(e)
    print(r)
    print(torch.svd(r))
    print(torch.det(r))

    # create tensor like other tensors
    e_like = torch.empty_like(e)
    z_like = torch.zeros_like(z)
    o_like = torch.ones_like(o)
    r_like = torch.rand_like(r)
    print(e_like)
    print(r_like)
    print(z_like)
    print(o_like)

    print(r)
    print(r_like)
    r[0][0] = 444
    print(r)
    print(r_like)
    # x_like is not alias of x. it is a deep copy/clone, different tensor

    a = torch.tensor([1.258, 88.6]) # creates a new copy of elemnts into a tensor
    print(a)
    b = a.to(torch.int16)
    print(b)

def run2():
    """ math and logic """
    # over 300 math and logic operations - abs, ceil, bitwise, comparisons, reductions, etc.
    ones = torch.zeros((2,2)) + 1 # elemw add
    twos = ones * 2 #  elemw mult
    print(twos)
    a = twos ** torch.tensor([[1, 2], [3, 4]])
    print(a)
    #tensor([[ 2.,  4.],
    #    [ 8., 16.]])

    b = twos ** torch.tensor([1,2])
    print(b)
    #tensor([[2., 4.],
    #    [2., 4.]])

def run3():
    """ broadcasting 
    broadcasting works in the following 3 situations
    comparing dimension from first to last
    1. each dimension must be equal
    2. one of the dimension must be 1
    3. the dimension of broadcast does not exist in tensor
    """
    r = torch.rand(2, 5)
    print(r)
    r2 = r * (torch.ones(1,5) * 2)
    print(r2)

    a =     torch.ones(4, 3, 2)
    b = a * torch.rand(   3, 2)
    print("*"*10)
    print(b)

    b = a * torch.rand(   3, 1)
    print("*"*10)
    print(b)
    
    b = a * torch.rand(5, 4, 3, 2)
    print("*"*10)
    print(b)
    
    b = a * torch.rand(   1, 2)
    print("*"*10)
    print(b)
    
    b = torch.rand(2, 2)
    print("*"*10)
    print(b)
    print(b.mul(b)) # not in place
    print(b) 
    print(b.mul_(b)) # in place - works for many othe operations
    print(b) 

def run4():
    """ squeezing and unaqueezing """
    a = torch.rand(3, 2) 
    print (a.shape ,a) # torch.Size([3, 2])
    
    b = a.unsqueeze(0)
    print(b.shape, b) # torch.Size([1, 3, 2])

    b = a.squeeze(0)
    print(b.shape, b) # torch.Size([3, 2])
    # squeeze is only called on dimension of extent 1
    
    b = a.unsqueeze(1)
    print(b.shape, b) # torch.Size([3, 1, 2]) 
    

if __name__ == "__main__":
    #run1()
    #run2()
    #run3()
    run4()
