import torch
import numpy as np
import argparse
from src.utils.load_params import load_params


def load_data(params):
    print("Loading text from CSVs...")
    train = np.loadtxt(
        params.base.raw_data_dir + "train.csv",
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
    )
    test = np.loadtxt(
        params.base.raw_data_dir + "test.csv",
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
    )
    return train, test


def standardize(train, test):
    train_X = train[:, 1:]
    train_y = train[:, 0]
    mean_px = train_X.mean().astype(np.float32)
    std_px = train_X.std().astype(np.float32)
    return (train_X - mean_px) / (std_px), train_y, (test - mean_px) / (std_px)


def split_data(train, test, params):
    print("Splitting the data for train and validation...")
    s_size = train.shape[0]  # Training set size
    v_size = int(train.shape[0] * params.data_processing.split)  # Validation set size
    print("Standardizing data...")
    train_std, train_labels, test_std = standardize(train, test)
    return (
        train_std[: s_size - v_size],
        train_labels[: s_size - v_size],
        train_std[s_size - v_size :],
        train_labels[s_size - v_size :],
        test_std,
    )


def tensorfy_data(train_X, train_y, validation_X, validation_y, test_X, device):
    print("Converting to Tensor...")
    train_tensor = torch.tensor(train_X, device=device)
    train_labels = torch.tensor(train_y, device=device).type(torch.LongTensor)
    validation_tensor = torch.tensor(validation_X, device=device)
    validation_labels = torch.tensor(validation_y, device=device).type(torch.LongTensor)
    test_tensor = torch.tensor(test_X)
    print("Reshaping Tensors...")
    train_reshaped = train_tensor.resize_(len(train_tensor), 1, 28, 28)
    validation_reshaped = validation_tensor.resize_(len(validation_tensor), 1, 28, 28)
    test_reshaped = test_tensor.resize_(len(test_tensor), 1, 28, 28)
    return (
        train_reshaped,
        train_labels,
        validation_reshaped,
        validation_labels,
        test_reshaped,
    )


def save_processed_data(train_x, train_y, valid_x, valid_y, test, params):
    print("Saving Tensors to File...")
    torch.save(train_x, params.base.processed_data_dir + "train_x_processed.pt")
    torch.save(train_y, params.base.processed_data_dir + "train_y_processed.pt")
    torch.save(valid_x, params.base.processed_data_dir + "valid_x_processed.pt")
    torch.save(valid_y, params.base.processed_data_dir + "valid_y_processed.pt")
    torch.save(test, params.base.processed_data_dir + "test_processed.pt")


if __name__ == "__main__":
    print("---------------------------------------------------------------")
    print(" DATA PROCESSING - START --------------------------------------")
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

    np.random.seed(params.base.random_seed)
    train, test = load_data(params)
    train_X, train_y, validation_X, validation_y, test_X = split_data(
        train, test, params
    )
    train_X, train_y, validation_X, validation_y, test_X = tensorfy_data(
        train_X, train_y, validation_X, validation_y, test_X, device
    )
    save_processed_data(train_X, train_y, validation_X, validation_y, test_X, params)
    print("---------------------------------------------------------------")
    print(" DATA PROCESSING - COMPLETE -----------------------------------")
    print("---------------------------------------------------------------")
