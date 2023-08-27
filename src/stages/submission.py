import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import argparse

from src.utils.load_params import load_params
from src.stages.train import Net

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
        self.data = torch.load("data/processed/" + data)

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        X = self.data[index]
        return X

def load_submission_data():
    print("Loading the processed submission data...")
    submission_data = Dataset("test_processed.pt")
    print("Generating data and creating batches for submission...")
    submission_loader = torch.utils.data.DataLoader(
                      submission_data)
    return submission_loader

def submission():
  with torch.no_grad():
    for data in submission_loader:
      output = network(data)
      pred = output.data.max(1, keepdim=True)[1]
      preds.append(pred.squeeze().tolist())
  print(f"Predictions: {len(preds)}")

if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print(" SUBMISSION - START ---------------------------------------------")
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

    submission_loader = load_submission_data()

    network = Net(params)
    network.load_state_dict(torch.load('models/model.pth'))

    preds = []
    preds_formatted = []

    submission()
    for image, pred in enumerate(np.array(preds).flatten("F")):
        preds_formatted.append([image+1, pred])

    with open('reports/submission.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "Label"]) # headers
        writer.writerows(preds_formatted) # Use writerows for nested list

    print("---------------------------------------------------------------")
    print(" SUBMISSION - COMPLETE ------------------------------------------")
    print("---------------------------------------------------------------")
