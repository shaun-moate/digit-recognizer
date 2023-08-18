import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from src.utils.load_params import load_params

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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

def load_data(params):
    print("Loading the processed training data...")
    train_data = Dataset("train_x_processed.pt", "train_y_processed.pt", params)
    print("Generating data and creating batches for training...")
    train_loader = torch.utils.data.DataLoader(
                      train_data,
                      batch_size=params.train.data.batch_size_train, 
                      shuffle=params.train.data.shuffle)

    print("Loading the processed validation data...")
    validation_data = Dataset("valid_x_processed.pt", "valid_y_processed.pt", params)
    print("Generating data and creating batches for validation...")
    validation_loader = torch.utils.data.DataLoader(
                      validation_data,
                      batch_size=params.train.data.batch_size_validation, 
                      shuffle=params.train.data.shuffle)
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
      torch.save(network.state_dict(), 'models/model.pth')
      torch.save(optimizer.state_dict(), 'models/optimizer.pth')

def validate():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in validation_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(validation_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(validation_loader.dataset),
    100. * correct / len(validation_loader.dataset)))


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

    train_loader, validation_loader = load_data(params)

    learning_rate = params.train.learning_rate
    momentum = params.train.momentum
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    n_epochs = params.train.n_epochs
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    log_interval = params.train.log_interval

    validate()
    for epoch in range(1, n_epochs + 1):
      train(epoch)
      validate()

    print("---------------------------------------------------------------")
    print(" TRAINING - COMPLETE ------------------------------------------")
    print("---------------------------------------------------------------")

