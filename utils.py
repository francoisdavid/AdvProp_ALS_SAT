import matplotlib.pyplot as plt

import numpy as np
import torch

from attackers import *

#function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function that evaluates the model based on the test set loaded above. 
def evaluateModel(net, criterion, testloader, device='cuda:0'):
    losses = 0 
    counter = 0
    rightPred = 0

    for (test_idx, testBatch) in enumerate(testloader):
        x, y = testBatch
        x, y = x.to(device), y.to(device)
        pred = net(x, y, advSamples=False)
        loss = criterion(pred, y)
        
        n = y.size(0)
        losses += loss.sum().data.cpu().numpy() * n
        counter += n
        rightPred += torch.sum(torch.argmax(pred, axis = 1) == y )     
            
    return losses / float(counter), rightPred.item() / float(counter)


def testAdversarials(model, testloader, fgsm=True, pgd=False, iters=2, eps=0.1, alpha=2.0/255.0, device='cuda:0'):
  rightPred = 0 
  counter = 0
  for i, data in enumerate(testloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      if pgd == False:
        adversarialImages = fgsm_attack(model, inputs, labels, eps = eps)
      else:
        adversarialImages = pgd_attack(model, inputs, labels, iters = iters, eps = eps, alpha = alpha)
      outputs = model(adversarialImages, labels)
      counter += labels.size(0)
      rightPred += torch.sum(torch.argmax(outputs, axis = 1) == labels) 

  return rightPred.item() / float(counter)