# Import all the required modules
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


import albumentations as A
from albumentations.pytorch import ToTensorV2


from torch_lr_finder import LRFinder

from pytorch_grad_cam import GradCAM
from utils import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Model(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet18Model, self).__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes

        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]

        self.train_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )
        self.test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.nll_loss(logits, y)
      preds = torch.argmax(logits, dim=1)
      self.accuracy(preds, y)

      # Calling self.log will surface up scalars for you in TensorBoard
      self.log("val_loss", loss, prog_bar=True)
      self.log("val_acc", self.accuracy, prog_bar=True)
      return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
      LEARNING_RATE = 0.03
      WEIGHT_DECAY = 1e-4
      # # Loss Function
      # criterion = nn.CrossEntropyLoss()

      # optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


      # lr_finder2 = LRFinder(self, optimizer, criterion, device='cuda')
      # lr_finder2.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
      # lr_finder2.plot()
      # suggested_lr = lr_finder2.suggest_lr()
      # lr_finder2.reset()
      # EPOCHS = 20
      # STEPS_PER_EPOCH = 2000

      # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
      #                                           max_lr=suggested_lr,
      #                                           steps_per_epoch=STEPS_PER_EPOCH,
      #                                           epochs=EPOCHS,
      #                                           pct_start=int(0.3*EPOCHS)/EPOCHS if EPOCHS != 1 else 0.5,   # 30% of total number of Epochs
      #                                           div_factor=100,
      #                                           three_phase=False,
      #                                           final_div_factor=100,
      #                                           anneal_strategy="linear"
      #                                           )
      return torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
      # return scheduler

    def prepare_data(self):
        # download
        Cifar10SearchDataset(self.data_dir, train=True, download=True)
        Cifar10SearchDataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

      # Assign train/val datasets for use in dataloaders
      if stage == "fit" or stage is None:
          cifar_full = Cifar10SearchDataset(self.data_dir, train=True, transform=self.train_transforms)
          self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

      # Assign test dataset for use in dataloader(s)
      if stage == "test" or stage is None:
          self.cifar_test = Cifar10SearchDataset(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
