import torch
import torch.nn as nn
import argparse

from src.utils.load_params import load_params
from src.utils.data_loader import data_loader
from src.utils.evaluate import evaluate
from src.stages.train import Net, Dataset

def load_evaluation_data(device):
    print("Loading the processed evaluation data...")
    validation_data = Dataset("valid_x_processed.pt", "valid_y_processed.pt", params)
    evaluation_loader = data_loader(validation_data, params, device)
    return evaluation_loader

if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print(" EVALUATION - START -------------------------------------------")
    print("---------------------------------------------------------------")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", dest="params", required=True)
    args = arg_parser.parse_args()
    params = load_params(args.params)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    evaluation_loader = load_evaluation_data(device)
    network = Net(params)
    network.to(device)
    network.load_state_dict(torch.load(params.train.model_path))
    evaluate(network, evaluation_loader, params, store=True)
    print("---------------------------------------------------------------")
    print(" EVALUATION - COMPLETE ----------------------------------------")
    print("---------------------------------------------------------------")
