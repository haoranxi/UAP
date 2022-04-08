import torch
import torchvision
import os
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from universal_224 import universal_adversarial_perturbation,universal_adversarial_perturbation_target
from prepare_224 import fooling_rate, get_model


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

noise,number,good,bad = universal_adversarial_perturbation_target(trainloader, model, device,path = './dominant_label/imagenette/dominant_label_success_target2_train.pth', target_class = 2, max_iter_df = 100, max_images = 50)
print(number)
print(good)
print(bad)
torch.save(noise, './uap/imagenette/success_noise_target2_50.pth')
print("saving noise_load.pth")


noise_or = torch.load("/scratch/hx759/UAP/uap/imagenette/success_noise_target2_50.pth")
fr = fooling_rate(testloader,noise,model, device,path = './dominant_label/imagenette/dominant_label_success_target2_test.pth'