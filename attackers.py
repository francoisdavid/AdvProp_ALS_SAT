# Modified from https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb 
from torch import nn
import torch
def pgd_attack(model, images, labels, eps=0.1, alpha=2/255, iters=2, device='cuda:0') :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data.to(device)
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images, labels)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images


# Modified from https://github.com/Harry24k/FGSM-pytorch/blob/master/FGSM.ipynb
def fgsm_attack(model, images, labels, eps=0.1, device='cuda:0') :
    loss = nn.CrossEntropyLoss()
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
            
    outputs = model(images, labels)
    
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images
