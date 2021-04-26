import torch
import torch.nn.functional as F
def adversarialLabelSmoothingLoss(yHat, y, alpha=0.02, device='cuda:0'):
    yHat = F.softmax(yHat, dim=1)
    smallestPred = torch.argmin(yHat, axis = 1)
    ySmall = torch.tensor(torch.eye(10)[smallestPred]) * alpha
    y = ((torch.tensor(torch.eye(10)[y]) * (1 - (1 * alpha))) + ySmall ).to(device)
    epsilon = 0.00000001
    #print(y[0, :])
    loss = -torch.mean(torch.sum(y * torch.log(yHat + epsilon), axis=1))

    return loss