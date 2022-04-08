# coding: utf-8
import torch
from universal_224 import universal_adversarial_perturbation
from prepare_224 import fooling_rate, get_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

import matplotlib.pyplot as plt

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)

    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    #gradient = cv2.resize(gradient,(255,255))
    #cv2.imwrite(file_name, gradient)
    print(gradient.shape)
    plt.imshow(gradient)
    plt.savefig(file_name)
    return



#####success images
data = 'scratch/hx759/UAP/data_imagenette/imagenette2/'
batch_size = 128



# Data loading code
#traindir = os.path.join(data, 'train')
traindir = '/scratch/hx759/UAP/data_imagenette/imagenette2/train/'
valdir = '/scratch/hx759/UAP/data_imagenette/imagenette2/val/'
#valdir = os.path.join(data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))


trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True)

model = torchvision.models.resnet34(pretrained = True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()



########uap
noise_or = torch.load("/scratch/hx759/UAP/uap/imagenette/success_noise_target1_50.pth")
save_path = "./images/imagenette/success_noise_target1_50.png"

#import cv2

#noise = (noise_or[0,0,:,:] + noise_or[0,1,:,:] + noise_or[0,2,:,:])/3
#print(noise.shape)
from PIL import Image 
from pylab import *

eps = 10/255

noise_show = ((noise_or[0].cpu().detach().numpy() / eps) + 1) / 2
noise_show = np.transpose(noise_show, (1, 2, 0))

noise_show -= noise_show.min()
noise_show = noise_show/noise_show.max()*1

plt.imshow(noise_show)


plt.savefig(save_path)


#save_gradient_images(noise_or[0].cpu().detach().numpy(), save_path)
#save_gradient_images(noise, save_path)



'''
########pgd
noise_or = torch.load("/scratch/hx759/UAP/uap/imagenette/pgd_0.03_success_advimage_target0.pth")
save_path = "./images/imagenette/pgd_0.03_success_sum_advimage_target0_50.png"
max_images = 50

#import cv2

#noise = (noise_or[0,0,:,:] + noise_or[0,1,:,:] + noise_or[0,2,:,:])/3
#print(noise.shape)
from PIL import Image 
from pylab import *


noise_all = noise_or[:max_images]

print(len(noise_all))
print(len(noise_all[0]))


#torch.save(noise_all,"/scratch/hx759/UAP/uap/imagenette/pgd_success_advimage_target2.pth")
#noise_all = noise_or.cpu().detach().numpy()
noise = sum(noise_all)

#noise =noise /len(noise_all)

save_gradient_images(noise.cpu().detach().numpy(), save_path)

'''


