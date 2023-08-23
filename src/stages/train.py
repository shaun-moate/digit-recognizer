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
        self.conv1 = nn.Conv2d(1, params.train.struct.conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(params.train.struct.conv1, params.train.struct.conv2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, params.train.struct.linear1)
        self.fc2 = nn.Linear(params.train.struct.linear1, params.train.struct.linear2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

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

    learning_rate = params.train.learning_rate
    momentum = params.train.momentum
    network = Net(params)
    network.to(params.base.device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

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

