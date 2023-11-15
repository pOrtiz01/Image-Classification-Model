from itertools import count
import torchvision
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from model import encoder_front,classifier
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR,ExponentialLR
import argparse
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--e', type=int, default=20)
parser.add_argument('--b', type=int, default=8)
parser.add_argument('--l', type=str, default='encoder.pth')
parser.add_argument('--s', type=str, default='classifier.pth')
parser.add_argument('--p', type=str, default='loss.png')
parser.add_argument('--d',type=str,default='cpu')
args = parser.parse_args()

device = torch.device(args.d)
save_dir = Path(args.s)
save_dir.mkdir(exist_ok=True, parents=True)
front = encoder_front.front
vgg_encoder = encoder_front.encoder

vgg_encoder.load_state_dict(torch.load(args.l))

for param in vgg_encoder.parameters():
    param.requires_grad = False

vgg_encoder = nn.Sequential(*list(vgg_encoder.children()))
network = classifier(vgg_encoder, front)
network.train()
network.to(device)

transform = transforms.Compose([
    transforms.Resize((69,69)),
    transforms.RandomCrop((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

#train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True)


loss_func=nn.CrossEntropyLoss()
optimizer = SGD(network.front.parameters(), lr=args.lr)
scheduler = StepLR(optimizer,step_size=5, gamma=.1)
loss_train=[]

for i in tqdm(range(args.e)):
    loss = 0
    counter=0
    accuracy_count = 0
    for batch in train_loader:
        counter+=1
        images, labels = batch
        images=images.to(device)
        labels=labels.to(device)
        one_hot_labels = torch.nn.functional.one_hot(labels, 100)
        optimizer.zero_grad()
        output=network(images)
        for j in range(0, len(labels)):
          if (output[j].argmax().item() == int(labels[j].item())):
            accuracy_count += 1
        loss=loss_func(output,labels)
        loss.backward()
        optimizer.step()

    loss_train+=[loss.item()]
    accuracy = (accuracy_count/50000)*100
    print(f'Epoch {i + 1}/{args.e}, Loss: {loss.item()}, Accuracy:{accuracy}%')

    if (i + 1) % (args.e) == 0 or (i + 1) == args.e:
        state_dict = network.front.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        torch.save(state_dict, save_dir /
                   'cifar100_classifier_iter_{:d}.pth'.format(i + 1))

if args.p != None:
    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(np.arange(0,args.e),np.array(loss_train), label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.savefig(args.p)

