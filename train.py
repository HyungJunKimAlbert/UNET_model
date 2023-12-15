#%% Import Packages
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

# Customized function
from model import UNet
from dataset import ImageDataset, ToTensor, RandomFlip, Normalization
from utils import save, load


# Training Parameters
lr = 1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 100

data_dir = './dataset'
ckpt_dir = './checkpoint'
log_dir = './log'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Transform
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()
])

dataset_train = ImageDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

dataset_val = ImageDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# Network
net = UNet().to(device)
# Loss function
fn_loss = nn.BCEWithLogitsLoss().to(device)
# Optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)
# etc
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)
num_batch_train = int(np.ceil(num_data_train / BATCH_SIZE))
num_batch_val = int(np.ceil(num_data_val / BATCH_SIZE))
# etc function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x>0.5)
# Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


# Network Training
ST_EPOCH = 0
net, optim, ST_EPOCH = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(ST_EPOCH+1, NUM_EPOCHS+1):
    net.train()
    loss_arr = []

    for batch_idx, data in enumerate(loader_train, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)
        # backward pass
        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        # Loss
        loss_arr += [loss.item()]
        print(f'TRAIN: EPOCH [{epoch} / {NUM_EPOCHS}] | BATCH [{batch_idx} / {num_batch_train}] | LOSS [{loss}]')

        # To numpy
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))
        # Tensorboard
        writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch_idx, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch_idx, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)
            # Loss 
            loss = fn_loss(output, label) 
            loss_arr += [loss.item()]
            # To numpy
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            print(f"VALID: EPOCH [{epoch} / {NUM_EPOCHS}] | BATCH [{batch_idx} / {num_batch_val}] | LOSS [{loss}]")

            # Tensorboard
            writer_val.add_image('label', label, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_val * (epoch-1) + batch_idx, dataformats='NHWC')
    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    
    if epoch % 10 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)



# Tensorboard close
writer_train.close()
writer_val.close()



# #%% test
# transform = transforms.Compose([
#     Normalization(mean=0.5, std=0.5),
#     RandomFlip(), 
#     ToTensor()
# ])

# datatset_train = ImageDataset(data_dir='./dataset/train', transform=transform)

# #%% check sample 
# data = datatset_train.__getitem__(0)
# input = data['input']
# label = data['label']


# print('SHAPE')
# print(f"Input: {input.shape}, Label: {label.shape}")

# print('tpye')
# print(f"Input: {input.type()}, Label: {label.type()}")


# plt.subplot(121)
# plt.imshow(input.squeeze())

# plt.subplot(122)
# plt.imshow(label.squeeze())