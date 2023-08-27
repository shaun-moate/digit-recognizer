import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.load_params import load_params
from src.utils.data_loader import data_loader
from src.utils.evaluate import evaluate

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, labels, params):
        self.labels = torch.load(params.base.processed_data_dir + labels)
        self.data = torch.load(params.base.processed_data_dir + data) 

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        # Layer 1
        out = self.conv1(x)
        out = self.relu1(out)
        # Layer 2
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        # Layer 3
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool2(out)
        # Flatten
        out = out.view(out.size(0), -1)
        # Linear
        out = self.fc1(out)
        return F.log_softmax(out, dim = 1)

def load_data(params, device):
    print("Loading the processed training data...")
    train_data = Dataset("train_x_processed.pt", "train_y_processed.pt", params)
    train_loader = data_loader(train_data, params, device)
    print("Loading the processed validation data...")
    validation_data = Dataset("valid_x_processed.pt", "valid_y_processed.pt", params)
    validation_loader = data_loader(validation_data, params, device)
    return train_loader, validation_loader

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), params.train.model_path)
      torch.save(optimizer.state_dict(), params.train.optimizer_path)

if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print(" TRAINING - START ---------------------------------------------")
    print("---------------------------------------------------------------")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", dest="params", required=True)
    args = arg_parser.parse_args()
    params = load_params(args.params)

    torch.backends.cudnn.enabled = params.train.cudnn_enabled
    torch.manual_seed(params.base.random_seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, validation_loader = load_data(params, device)

    network = Net(params)
    network.to(params.base.device)

    if params.train.optimizer.type == "sgd":
      optimizer = optim.SGD(network.parameters(), lr=params.train.optimizer.learning_rate,
                            momentum=params.train.optimizer.momentum)
    elif params.train.optimizer.type == "adam" :
      optimizer = optim.Adam(network.parameters(), lr=params.train.optimizer.learning_rate)

    n_epochs = params.train.n_epochs
    train_losses = []
    train_counter = []
    results = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    log_interval = params.train.log_interval

    evaluate(network, validation_loader, params, results)
    for epoch in range(1, n_epochs + 1):
      train(epoch)
      evaluate(network, validation_loader, params, results)

    store_results = []
    for epoch, (loss, accuracy) in enumerate(results):
      store_results.append([epoch, loss, accuracy.item()])

    with open('reports/results.csv', 'w+', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["epoch", "loss", "accuracy"]) # headers
      writer.writerows(store_results) # Use writerows for nested list

    x = np.loadtxt("reports/results.csv",
                     delimiter=",", skiprows=1, usecols=(0), dtype=np.int32)
    y = np.loadtxt("reports/results.csv",
                     delimiter=",", skiprows=1, usecols=(2), dtype=np.float32)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, y)
    plt.savefig("reports/results.png")

    print("---------------------------------------------------------------")
    print(" TRAINING - COMPLETE ------------------------------------------")
    print("---------------------------------------------------------------")

