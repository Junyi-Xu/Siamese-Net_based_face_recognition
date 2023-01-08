import torch
import os
from train import euclidean_distance, compute_accuracy
from network import Siamese
from config import get_config
from torch.utils.data import DataLoader
from dataset import CASIA
from tqdm import tqdm


def get_dataloader(conf):
    test_set = CASIA(conf.list_dir, 'test', 1500)
    test_loader = DataLoader(test_set, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    return test_loader


def test(model, test_loader):
    model.eval()
    distances_lst = []
    targets_lst = []
    for batch_idx, (inputs_1, inputs_2, targets) in enumerate(tqdm(test_loader, desc="Testing")):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs_1 = model(inputs_1)
            outputs_2 = model(inputs_2)
            distances = euclidean_distance((outputs_1, outputs_2)).squeeze()
            distances_lst += distances.tolist()
            targets_lst += targets.tolist()
    accuracy = compute_accuracy(distances_lst, targets_lst)
    return accuracy


def main(conf):
    test_loader = get_dataloader(conf)
    model = Siamese(in_channels=3)
    model.load_state_dict(torch.load(os.path.join(conf.output_dir, 'model.pth')))
    accuracy = test(model, test_loader)
    print(f'test_acc: {100.*accuracy:.5f}')


if __name__ == '__main__':
    conf = get_config()
    main(conf)