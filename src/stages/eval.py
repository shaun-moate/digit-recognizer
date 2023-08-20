import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json

from src.utils.load_params import load_params
from src.stages.train import Net, Dataset

def load_evaluation_data():
    print("Loading the processed validation data...")
    validation_data = Dataset("valid_x_processed.pt", "valid_y_processed.pt", params)
    print("Generating data and creating batches for validation...")
    evaluation_loader = torch.utils.data.DataLoader(
                      validation_data,
                      batch_size=100,
                      shuffle=True)
    return evaluation_loader

def evaluate(params):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in evaluation_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(evaluation_loader.dataset)
    output = {"Avg. Loss": round(test_loss, 4), "Accuracy": round((correct / len(evaluation_loader.dataset)).item(), 4)}
    output_file = open(params.eval.metrics_path, "w")
    json.dump(output, output_file, indent=6)

if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print(" EVALUATION - START -------------------------------------------")
    print("---------------------------------------------------------------")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", dest="params", required=True)
    args = arg_parser.parse_args()
    params = load_params(args.params)

    evaluation_loader = load_evaluation_data()
    network = Net(params)
    network.load_state_dict(torch.load(params.train.model_path))
    evaluate(params)
    print("---------------------------------------------------------------")
    print(" EVALUATION - COMPLETE ----------------------------------------")
    print("---------------------------------------------------------------")
