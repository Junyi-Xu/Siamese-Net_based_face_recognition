import torch
from torch.utils.data import DataLoader
from dataset import CASIA
from network import Siamese
import torch.optim as optim
from config import get_config
from tqdm import tqdm
import os


def get_dataloader(conf):
    train_set = CASIA(conf.list_dir, 'train', 60000)
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    val_set = CASIA(conf.list_dir, 'val', 1500)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    return train_loader, val_loader


def euclidean_distance(vects):
    x, y = vects
    return torch.sqrt(torch.sum(torch.square(x - y), dim=1, keepdim=True))


def contrastive_loss(dists, targets):
    zeros = torch.full(size=(dists.shape[0],), fill_value=0)
    margin = 1
    losses = targets * torch.square(dists) + (1 - targets) * torch.square(torch.maximum(margin - dists, zeros))
    return torch.mean(losses)


def compute_accuracy(dists, targets):
    match_lst = [int(dist < 0.5) == y for dist, y in zip(dists, targets)]
    return sum(match_lst) / len(dists)


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (inputs_1, inputs_2, targets) in enumerate(tqdm(train_loader, desc="Training")):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        # zero gradients
        optimizer.zero_grad()
        # make predictions for each image in the pair
        outputs_1 = model(inputs_1)
        outputs_2 = model(inputs_2)
        # calculate Euclidean distance
        distances = euclidean_distance((outputs_1, outputs_2)).squeeze()
        # compute loss and gradients
        losses = contrastive_loss(distances, targets)
        losses.backward()
        # adjust learning weights
        optimizer.step()
    return losses


def val(model, val_loader):
    model.eval()
    distances_lst = []
    losses = []
    targets_lst = []
    for batch_idx, (inputs_1, inputs_2, targets) in enumerate(tqdm(val_loader, desc="Validating")):
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs_1 = model(inputs_1)
            outputs_2 = model(inputs_2)
            distances = euclidean_distance((outputs_1, outputs_2)).squeeze()
            distances_lst += distances.tolist()
            losses.append(contrastive_loss(distances, targets).item())
            targets_lst += targets.tolist()
    accuracy = compute_accuracy(distances_lst, targets_lst)
    avg_loss = sum(losses) / len(losses)
    return avg_loss, accuracy


def main(conf):
    train_loader, val_loader = get_dataloader(conf)
    model = Siamese(in_channels=3)
    optimizer = optim.AdamW(model.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    for epoch in range(conf.epochs):
        print(f"epoch: [{epoch+1} | {conf.epochs}]")
        train_loss = train(model, train_loader, optimizer)
        val_loss, val_acc = val(model, val_loader)
        print(f'Epoch: {epoch+1}  train_loss: {train_loss:.5f}  val_loss: {100.*val_loss:.5f}  val_acc: {100.*val_acc:.5f}')
    torch.save(model.state_dict(), os.path.join(conf.output_dir, 'model.pth'))


if __name__ == '__main__':
    conf = get_config()
    main(conf)
