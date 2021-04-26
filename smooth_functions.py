import torch

# A custom function so that we can use ReLU for the forward pass and the parametricSoftplus for the backward pass (as they did in the paper).

class SAT(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        return (1 / (1 + torch.exp(-10 * input))) * grad_input

# A custom class that is exatly ReLU, so that we can apply Parametric Softplus the same way.
class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

#custom function that implements Swish
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        return (torch.sigmoid(input)*(1+(1-torch.sigmoid(input))*input))*grad_input
