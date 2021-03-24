from __future__ import print_function, division

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pandas as pd
from sklearn import neighbors, svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import expreport as report

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


def retrain_model(dataset_in, model_in, image_path, epochs=25):
    def count_labels(datasets):
        labels = set()
        for _, ds in datasets.items():
            labels.update(ds["func"].unique())
        return len(labels)

    def create_label_mapping(datasets):
        labels = set()
        labels.update(ds["func"].unique())
        mapping = {}
        indx = 1
        for x in labels:
            mapping[x] = indx
            indx += 1
        return mapping

    def retrain_internal(model, criterion, optimizer, scheduler, dataloader, dataset_size, num_epochs=5):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            scheduler.step()
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    data_transform = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    ds = dataset_in #[dataset_in["train"] == 1]

    label_mapping = create_label_mapping(ds)
    label_count = len(label_mapping.keys()) + 1
    dataset_size = len(ds)

    pdd = PdDataset(image_path, ds, data_transform, label_mapping)

    dataloader = torch.utils.data.DataLoader(pdd, batch_size=4, shuffle=True, num_workers=4)

    num_ftrs = model_in.fc.in_features
    model_in.fc = nn.Linear(num_ftrs, label_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_in.to(device)
    # print(model_ft)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.05 every 6 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.05)

    return retrain_internal(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, dataset_size,
                           num_epochs=epochs)


class ImDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ds, data_transform):
        self.ds = ds
        self.ds_size = len(self.ds.index)
        self.img_path = img_path
        self.data_transform = data_transform

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        image = load_img(self.img_path, dat["id"])
        image = self.data_transform(image)
        return image, dat["id"]

class VecPairDataset(torch.utils.data.Dataset):
    def __init__(self, vectors, pairs_ds):
        self.ds = pairs_ds
        self.ds_size = len(self.ds.index)
        self.vectors = vectors

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        id1 = str(dat["id1"])
        id2 = str(dat["id2"])
        if id1 not in self.vectors:
            vec1 = next(iter(self.vectors.values()))
            print("vec1 missing!: ", id1)
        else:
            vec1 = self.vectors[id1]

        if id2 not in self.vectors:
            vec2 = next(iter(self.vectors.values()))
            print("vec2 missing!: ", id2)
        else:
            vec2 = self.vectors[id2]
        label = 0 if dat["type"] <= 0 else 1
        return vec1, vec2, label

class CCDBinClassifier(nn.Module):
    def __init__(self, vec_size):
        super(CCDBinClassifier, self).__init__()
        self.encode_dim = vec_size
        self.hidden_dim = vec_size
        self.num_layers = 1
        self.batch_size = 4
        self.label_size = 1
        self.gpu = True

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def init_hidden(self):
        if self.gpu is True:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x1, x2):
        x1 = x1.squeeze(-1).squeeze(-1)
        x2 = x2.squeeze(-1).squeeze(-1)

        y = torch.abs(torch.add(x1, -x2))

        x = self.fc(y)
        x = torch.sigmoid(x.squeeze(-1))
        return x

def compute_vectors(data_path, image_path, transform, retrained_model):
    vectors = {}

    raw_dataset = pd.read_csv(data_path)
    dataset_size = len(raw_dataset)
    dataset = ImDataset(image_path, raw_dataset, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    icnn = retrained_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    icnn = icnn.to(device)
    icnn.eval()  # Set model to evaluate mode
    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)

            outputs = icnn(inputs)
            outputs = outputs.to("cpu")
            ids = ids.to("cpu").numpy()
            for i in range(len(outputs)):
                vectors[str(ids[i])] = outputs[i]
    return vectors


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, retrain=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            trues = []
            predicts = []
            # Iterate over data.
            for inputsA, inputsB, labels in dataloaders[phase]:
                inputsA = inputsA.to(device)
                inputsB = inputsB.to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputsA, inputsB)
                    preds = (outputs.data > 0.50).float()
                    predicts.extend((outputs.data > 0.50).float().to("cpu").numpy())
                    trues.extend(labels.data.to("cpu").numpy())
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputsA.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts)
            print("raw - %s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            print("%s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            if (phase == "test") and (epoch == num_epochs-1):
                report.report.add_result(epoch_acc.item(), precision, recall, f1, "nn"+("" if not retrain else "_r"))
                print("CSV-RESULT: %s, %s, %s, %s, %s, %s" % ("bc"+("" if not retrain else "_r"), phase, epoch_acc.item(), precision, recall, f1))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

# Epochs is not used by classical algorithms
def learn(data_path, image_path, pairs_path, epochs=5, retrain=True):
    # Normalize data
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

    model = models.resnet50(pretrained=True)

    if retrain:
        raw_dataset = pd.read_csv(data_path)
        model = retrain_model(raw_dataset, model, image_path, epochs=epochs)

    model = nn.Sequential(*list(model.children())[:-1])  # remove last layer

    vectors = {
        "train": compute_vectors(data_path, image_path, data_transforms["train"], model),
        "test": compute_vectors(data_path, image_path, data_transforms["test"], model)
    }

    pddat = pd.read_csv(pairs_path)
    # Use train field if present, else do random splitting
    if "train" in pddat.columns:
        ds = {
            "train": pddat[pddat["train"] == 1],
            "test": pddat[pddat["train"] == 0]
        }
    else:
        train, test = train_test_split(pddat, test_size=0.2, stratify=pddat[["func"]])
        ds = {
            "train": train,
            "test": test
        }
    dataset_sizes = {x: len(ds[x]) for x in ['train', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    pdds = {x: VecPairDataset(vectors[x], ds[x])
        for x in ['train', 'test']}

    x_train = []
    y_train = []
    for idx, r in ds["train"].iterrows():
        id1 = str(r["id1"])
        id2 = str(r["id2"])
        if id1 not in vectors["train"]:
            continue
        if id2 not in vectors["train"]:
            continue
        x1 = vectors["train"][id1].numpy().flatten()
        x2 = vectors["train"][id2].numpy().flatten()
        x_train.append(np.absolute(x1-x2))
        y_train.append(0 if r["type"] <= 0 else 1)
    x_test = []
    y_test = []
    for idx, r in ds["test"].iterrows():
        id1 = str(r["id1"])
        id2 = str(r["id2"])
        if id1 not in vectors["test"]:
            continue
        if id2 not in vectors["test"]:
            continue
        x1 = vectors["train"][id1].numpy().flatten()
        x2 = vectors["train"][id2].numpy().flatten()
        x_test.append(np.absolute(x1-x2))
        y_test.append(0 if r["type"] <= 0 else 1)

    # run SVM
    kernels = {'linear', 'poly', 'rbf', 'sigmoid'}
    for kernel in kernels:
        try:
            clf = svm.SVC(kernel=kernel)
            clf.fit(x_train, y_train)
            confidence = clf.score(x_test, y_test)
            y_pred = clf.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print(kernel + "SVM :: ")
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print("raw - P: %s; R: %s, F1: %s" % (precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            print("P: %s; R: %s, F1: %s" % (precision, recall, f1))
            report.report.add_result(confidence, precision, recall, f1, kernel + "-svm"+("" if not retrain else "_r"))
            print("CSV-RESULT: %s, %s, %s, %s, %s" % (kernel + "svm"+("" if not retrain else "_r"),confidence, precision, recall, f1))
            print()
        except:
            print('SVM Error with kernel = ' + kernel)

    # run kNN
    algos = {'ball_tree', 'kd_tree', 'brute'}
    for algo in algos:
        try:
            clf = neighbors.KNeighborsClassifier(algorithm=algo)
            clf.fit(x_train, y_train)
            confidence = clf.score(x_test, y_test)
            y_pred = clf.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print(algo + "-kNN :: ")
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print("raw - P: %s; R: %s, F1: %s" % (precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            print("P: %s; R: %s, F1: %s" % (precision, recall, f1))
            report.report.add_result(confidence, precision, recall, f1, algo + "-knn"+("" if not retrain else "_r"))
            print("CSV-RESULT: %s, %s, %s, %s, %s" % ("knn"+("" if not retrain else "_r"), confidence, precision, recall, f1))
            print()
        except:
            print('KNN Error with algo = ' + algo)

    # run NN
    # dataloaders = {x: torch.utils.data.DataLoader(pdds[x], batch_size=4, shuffle=True, num_workers=4)
    #                for x in ['train', 'test']}
    # dataset_sizes = {x: len(ds[x].index) for x in ['train', 'test']}

    # model = CCDBinClassifier(2048)
    # model = model.to(device)

    # parameters = model.parameters()
    # optimizer = torch.optim.Adamax(parameters)
    # loss_function = torch.nn.BCELoss()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    # train_model(model, dataloaders, dataset_sizes, loss_function, optimizer, exp_lr_scheduler, num_epochs=epochs, retrain=retrain)
