import torch
from torch.utils.data.dataloader import default_collate

def data_loader(data, params, device):
    data_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=params.train.data.batch_size_train,
                        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
                        shuffle=params.train.data.shuffle)
    return data_loader
