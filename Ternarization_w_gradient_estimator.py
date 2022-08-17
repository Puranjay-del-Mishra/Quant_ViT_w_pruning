import torch
import math
import numpy as np
from torch import nn
from torch.nn import Parameter
from torch.nn.init import Tensor
from torch.nn import init as init
from torch.nn import functional as F

class TNlayer_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,layer,inp_list,alpha_list):
        #obtaining the alpha and input values from the lists passed as arguments
        alpha = alpha_list[0]
        inp = inp_list[0]
        ####

        inp = torch.mul(inp,alpha)  #multiplying alpha to the input
        inp = torch.mul(layer.weight.data,inp)  ## this should not be a regular multiplication,we ideally need an only accumulation multiplication
        layer.weight.data = torch.t(torch.mul(torch.t(layer.weight.data), alpha))
        return inp

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output,None,None  # vanilla STE idea

    @staticmethod
    def Ternarize(tensor):
        tensor = tensor.cuda()
        # print(tensor[0])

        new_tensor = tensor.abs()
        delta = torch.mul(0.75, torch.mean(new_tensor, dim=1))
        # print(delta[0])

        new_tensor = torch.t(new_tensor)

        t = torch.greater_equal(new_tensor, delta).type(torch.cuda.FloatTensor)
        # print(t[0])
        x = torch.greater(tensor, 0).type(torch.cuda.FloatTensor)
        y = torch.less(tensor, 0).type(torch.cuda.FloatTensor)
        y = torch.mul(y, -1)
        z = torch.add(x, y)
        t = torch.t(t)
        final = torch.mul(t, z)

        new_tensor = torch.t(new_tensor)

        final.cuda()
        alpha = torch.mean(torch.mul(final, new_tensor), dim=1)

        # print(output[0])

        return (final, alpha)


class Ternary_lin(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Ternary_lin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ## weights ternarize
        self.weight.data,alpha = TNlayer_func.Ternarize(self.weight.data) ##this ternarization just numerically converts the weights,
        # we will have to check if we need it in int8 (or even smaller not representation)

        # we need to pass the alpha, input and the weights of the model through the autograd subclass.
        # we pass the alpha and inputs as a list of tensors as we don't want the TNlayer_func class to return gradients for them
        alpha_list = [alpha]
        input_list = [input]
        ####
        ## performing the linear step  by calling TNlayer_func.apply() to implement STE (note we are taking bias as 0)
        out = TNlayer_func.apply(self.weight,alpha_list,input_list)

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )