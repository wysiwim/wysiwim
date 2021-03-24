from __future__ import print_function, division

#import sys
#sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import expreport as report

DATASET_PORTION = 1.0

def count_labels(datasets):
    labels = set()
    for _, ds in datasets.items():
        labels.update(ds["func"].unique())
    return len(labels)

def create_label_mapping(datasets):
    labels = set()
    for _, ds in datasets.items():
        labels.update(ds["func"].unique())
    mapping = {}
    indx = 1
    for x in labels:
        mapping[x] = indx
        indx += 1
    return mapping

def load_img(img_path, img_id):
    return Image.open("%s/%s.png" % (img_path, img_id)).convert('RGB')


class PdDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ds, data_transform, labelmapping):
        self.ds = ds
        self.ds_size = len(self.ds.index)
        self.img_path = img_path
        self.data_transform = data_transform
        self.labelmapping = labelmapping

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        image = load_img(self.img_path, dat["id"])
        image = self.data_transform(image)
        raw_label = dat["func"]
        label = self.labelmapping[raw_label] if raw_label in self.labelmapping else 0
        return image, label

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, model_selection, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_P = 0.0
    best_R = 0.0
    best_F1 = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            y_test = []
            y_pred = []
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_test.extend(labels.data.to("cpu").numpy())
                y_pred.extend(preds.data.to("cpu").numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("code classification :: " + model_selection)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print("raw - %s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print("%s - A: %s; P: %s; R: %s, F1: %s" % (phase, epoch_acc.item(), precision, recall, f1))
            print()

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_P = precision
                best_R = recall
                best_F1 = f1
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    report.report.add_result(best_acc.item(), best_P, best_R, best_F1, "cc_"+model_selection)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def learn(data_path, image_path, model_selection, epochs=25):
    # Normalization for both phases
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    pddat = pd.read_csv(data_path)

    if DATASET_PORTION < 0.99:
        pddat = pddat.groupby('func', group_keys=False)\
            .apply(lambda x: x.sample(
                int(DATASET_PORTION* float(len(x))), random_state=42))
        print(pddat["func"].value_counts())
    if "train" in pddat.columns:
        ds = {
            "train": pddat[pddat["train"] == 1],
            "test":  pddat[pddat["train"] == 0]
        }
    else:
        train, test = train_test_split(pddat, test_size=0.2, stratify=pddat[["func"]], random_state=42)
        ds = {
            "train": train,
            "test": test
        }

    label_mapping = create_label_mapping(ds)
    label_count = len(label_mapping.keys()) + 1
    dataset_sizes = {x: len(ds[x]) for x in ['train', 'test']}


    pdds = {x: PdDataset(image_path, ds[x], data_transforms[x], label_mapping)
        for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(pdds[x], batch_size=4, shuffle=True, num_workers=4)
                for x in ['train', 'test']}

    if model_selection == "resnet50":
        model_ft = models.resnet50(pretrained=True)
    elif model_selection == "resnet18":
        model_ft = models.resnet18(pretrained=True)
    elif model_selection == "denset":
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

    if model_selection != "denset":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, label_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.05 every 6 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.05)


    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, model_selection, num_epochs=epochs)

