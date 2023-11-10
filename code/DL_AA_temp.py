# %%
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataclasses import dataclass
import nibabel as nib
# %%
@dataclass
class Config:
    bs: int = 32
    nw: int = 8
    ssd: str = '../SampleSplits/'
    rep: int = 0
    scorename: str = 'age'
    predictor: str = 'smriPath'
    cr: str = 'clx'
def loadData(cfg, mode):
    # Batch Dataloader
    dset = MRIDataset(cfg, mode)
    dloader = DataLoader(dset, batch_size=cfg.bs,
                         shuffle=True, num_workers=cfg.nw, drop_last=True, pin_memory=True)
    return dloader
class MRIDataset(Dataset):
    def __init__(self, cfg, mode):
        self.df = readFrames(cfg.ssd, mode, cfg.rep)
        self.scorename = cfg.scorename
        self.predictor = cfg.predictor
        self.cr = cfg.cr
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        X, y = read_X_y_5D_idx(
            self.df, idx, self.predictor, self.scorename, self.cr)
        return [X, y]
def readFrames(ssd, mode, rep):
    # Read Data Frame
    df = pd.read_csv(ssd + mode + '_rep_' + str(rep) + '.csv')
    print('Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')
    return df
def read_X_y_5D_idx(df, idx, predictor, scorename, cr):
    X, y = [], []
    # Read image
    fN = df[predictor].iloc[idx]
    X = np.float32(nib.load(fN).get_fdata())
    X = (X - X.min()) / (X.max() - X.min())
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
    # Read label
    y = df[scorename].iloc[idx]
    if scorename == 'label':
        y -= 1
    if cr == 'reg':
        y = np.array(np.float32(y))
    elif cr == 'clx':
        y = np.array(y)
    return X, y
class MRINet(nn.Module):
    def __init__(self, num_classes=2):
        super(MRINet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# %%
def train(config, checkpoint_dir=None):
    cfg = Config(bs=int(config["batch_size"]), ssd='../SampleSplits/',
                 predictor='smriPath', scorename='dx', nw=4, cr='clx', rep=0)
    net = MRINet()
    device = "cpu"
    print(torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    trainloader = loadData(cfg, 'tr')
    valloader = loadData(cfg, 'va')
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")
def test_accuracy(net, device="cpu"):
    cfg = Config(bs=4, ssd='./SampleSplits/',
                 predictor='smriPath', scorename='dx', nw=4, cr='clx', rep=0)
    testloader = loadData(cfg, 'tr')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        train,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    # can use partial to wrap train to set arguments
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    best_trained_model = MRINet()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
# %%