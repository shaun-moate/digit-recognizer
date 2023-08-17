import torch
import numpy as np
import argparse
from src.utils.load_params import load_params

def process_raw_data(params):
    test, train = load_data(params)
    test, train = tensorfy_data(test, train, params)
    save_processed_data(test, train, params)
    print("Processing Complete...")

def load_data(params):
    print("Loading text from CSVs...")
    test = np.loadtxt(params.base.raw_data_dir + "test.csv",
                     delimiter=",", skiprows=1, dtype=float)
    train = np.loadtxt(params.base.raw_data_dir + "train.csv",
                     delimiter=",", skiprows=1, dtype=float)
    return test, train

def tensorfy_data(test, train, params):
    print("Converting to Tensor...")
    test_tensor = torch.tensor(test)
    train_tensor = torch.tensor(train)
    print("Reshaping Tensors...")
    test_reshaped = test_tensor.resize_(len(test_tensor), params.train.struct.inputs_x, params.train.struct.inputs_y)
    train_reshaped = test_tensor.resize_(len(train_tensor), 1, params.train.struct.inputs_x, params.train.struct.inputs_y)
    return test_reshaped, train_reshaped

def save_processed_data(test, train, params):
    print("Saving Tensors to File...")
    torch.save(test, params.base.processed_data_dir + "test_processed.pt")
    torch.save(train, params.base.processed_data_dir + "train_processed.pt")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", dest="params", required=True)
    args = arg_parser.parse_args()

    params = load_params(args.params)
    process_raw_data(params)

