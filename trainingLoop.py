import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
from attackers import pgd_attack, fgsm_attack
from utils import evaluateModel, imshow, testAdversarials
from smooth_functions import SAT, ReLU, Swish
from label_smoothing import adversarialLabelSmoothingLoss
from models import *
import sys


def trainingLoop(net, epochs=100, earlyStopping=100, lr=0.1, adversarialTraining=False, advProp=False, als=False, eps=0.1, alpha = 2.0/255.0, activation = None):
  criterion = nn.CrossEntropyLoss()
  start_time = time.time()
  learningRate = lr

  # Altered version of what they did in https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
  optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9, weight_decay=5e-4)
  train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-20)
  # For the early stopping setup. Just to output accuracy of the best loss.
  bestLoss = np.inf
  countSinceImprovement = 0
  bestAccuracy = 0

  trainL = []
  trainA = []
  testA = []
  testL = []

  # Iterate over the epochs.
  for epoch in range(epochs):

      counter = 0
      rightPred = 0
      running_loss = 0.0
      countMini = 0
      gap = 0

      for i, data in enumerate(trainloader, 0):

          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)

          if adversarialTraining == True :
             # adversarialImages = fgsm_attack(net, inputs, labels)
             adversarialImages = pgd_attack(net, inputs, labels, eps, alpha)
             outputsPredictions = net(adversarialImages, labels, advSamples=True)
             optimizer.zero_grad()
             if als == True:
               lossAdvProp = adversarialLabelSmoothingLoss(outputsPredictions, labels)
             else:
                lossAdvProp = criterion(outputsPredictions, labels).to(device)
             lossAdvProp.backward()
             optimizer.step()

          optimizer.zero_grad()
          countMini +=1
          outputs = net(inputs, labels)
          if als == True:
            lossOriginal = adversarialLabelSmoothingLoss(outputs, labels)
          else:
            lossOriginal = criterion(outputs, labels).to(device)
          lossOriginal.backward()
          optimizer.step()
          
          if adversarialTraining == True :
             running_loss += (lossAdvProp.item() + lossOriginal.item())/ 2
          else:
             running_loss += lossOriginal.item()

          counter += labels.size(0)
          rightPred += torch.sum(torch.argmax(outputs, axis = 1) == labels)
          gap += 1
          if round(countMini % 49) == 48 or countMini == 195:
              testLoss, testAccuracy = evaluateModel(net, criterion, testloader)
              trainAccuracy = rightPred.item() / float(counter)
              for param_group in optimizer.param_groups:
                  learning = float(param_group['lr'])
              print("Epoch: %.0f -->  (%.0f/4) \tTrain L: %.4f Test L: %.4f \t Train A: %.4f Test A: %.4f \t LR: %.6f" % (epoch, float(countMini)/196 * 4, running_loss / (gap) , testLoss, trainAccuracy, testAccuracy, (float(learning)) ))
              
              testA.append(testAccuracy)
              testL.append(testLoss)
              trainA.append(trainAccuracy)
              trainL.append(running_loss / (gap))

              running_loss = 0.0
              gap = 0
              if testAccuracy < bestAccuracy:
                countSinceImprovement += 1
              else:
                bestLoss = testLoss
                bestAccuracy = testAccuracy
                countSinceImprovement = 0

      if countSinceImprovement > earlyStopping * 4:
            break
      if epochs - 20 > epoch:
        train_scheduler.step(epoch=epoch)
      print()
         


  print("Finished Training in --- %s seconds ---" % (time.time() - start_time))
  print("Test Loss: ", bestLoss , " \t Test Accuracy: ", bestAccuracy)

  return trainL, trainA, testL, testA
  
  
def runExperiment(exp, saving = True, plotting = True, test_attacks = True):
  parameters = exp["parameters"]
  if 'activation' in parameters.keys():
    model = resnet18(advProp = parameters["advProp"], activation = parameters["activation"])
  else:
    model = resnet18(advProp = parameters["advProp"])
  #model.cuda()
  print()
  trainLoss, trainAccur, testLoss, testAccur = trainingLoop(net=model, **parameters)
  if saving:
    PATH = exp["path"]
    npy_path = exp["npy_path"]
    torch.save(model.state_dict(), PATH)
    np.save('/content/gdrive/MyDrive/ift6756-project-shared-folder/saves/{}trainLoss.npy'.format(npy_path), trainLoss)
    np.save('/content/gdrive/MyDrive/ift6756-project-shared-folder/saves/{}trainAccur.npy'.format(npy_path), trainAccur)
    np.save('/content/gdrive/MyDrive/ift6756-project-shared-folder/saves/{}testLoss.npy'.format(npy_path), testLoss)
    np.save('/content/gdrive/MyDrive/ift6756-project-shared-folder/saves/{}testAccur.npy'.format(npy_path), testAccur)
  if plotting:
    plt.plot(testAccur)
    plt.plot(trainAccur)
    plt.legend(["Test Accuracy", "Train Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.show()
  if test_attacks:
    results = {
        "Testing Accuracy:": evaluateModel(model, nn.CrossEntropyLoss(), testloader)[1] * 100,
        "Adversarial Robustness FGSM 0.1: ": testAdversarials(model, testloader, eps=0.1) * 100,
        "Adversarial Robustness PGD 0.1: ": testAdversarials(model, testloader, pgd=True, eps=0.1, iters=2) * 100,
        "Adversarial Robustness FGSM 0.5: ": testAdversarials(model, testloader, eps=0.5) * 100,
        "Adversarial Robustness PGD 0.5: ": testAdversarials(model, testloader, pgd=True, eps=0.5, iters=2) * 100
    }
    print("Testing Accuracy:", evaluateModel(model, nn.CrossEntropyLoss(), testloader)[1] * 100, "%")
    print("Adversarial Robustness FGSM 0.1: ", testAdversarials(model, testloader, eps=0.1) * 100, " % accuracy")
    print("Adversarial Robustness PGD 0.1: ", testAdversarials(model, testloader, pgd=True, eps=0.1, iters=2) * 100, " % accuracy")
    print("Adversarial Robustness FGSM 0.5: ", testAdversarials(model, testloader, eps=0.5) * 100, " % accuracy")
    print("Adversarial Robustness PGD 0.5: ", testAdversarials(model, testloader, pgd=True, eps=0.5, iters=2) * 100, " % accuracy")

    return results

if __name__ == "__main__":
        
    default = np.array(["Traditional Training", 100, False, False, False, 0.1, 2.0/255, "/saves/traditionalTraining", "traditionalTraining_" ],  dtype=object)
    for i, arg in enumerate(sys.argv[1:]):
        default[i] = arg
    experiment={
      "description": default[0],
      "parameters":{
        'epochs': int(default[1]),
        'adversarialTraining': bool(default[2]),
        'advProp': bool(default[3]),
        'als': bool(default[4]),
        'eps': float(default[5]),
        'alpha': float(default[6])
      },
      "path": default[7],
      "npy_path": default[8]
    }
    print(experiment)
        
    # Make the dataloader for the dataset.
    # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
    CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(18),
        transforms.ToTensor(),
        # Normalize with the mean and standard deviation of the cifar100 dataset.
        transforms.Normalize(
            CIFAR10_TRAIN_MEAN,
            CIFAR10_TRAIN_STD),
    ])
    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize(
            CIFAR10_TRAIN_MEAN,
            CIFAR10_TRAIN_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=4)
                                             
    results = runExperiment(exp=experiment, saving = True, plotting = False)
    print(results)
                                             
                                             
                                             
                                             
                                             
                                             
    
