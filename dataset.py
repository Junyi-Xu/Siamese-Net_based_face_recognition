from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from tools import crop_face
from config import get_config


###################
# preparing dataset
###################

def get_image_path_list(input_dir):
    img_path_lst = []
    for root, dircs, files in sorted(os.walk(input_dir)):
        for file in sorted(files):
            if file.endswith('bmp'):
                img_path_lst.append(os.path.join(root, file))
    return img_path_lst


def read_images(input_dir, dataset_size=500, batch=5):
    img_path_lst = get_image_path_list(input_dir)
    imgs = np.empty(shape=(dataset_size * batch, 200, 200, 3), dtype=np.uint8)
    for i in tqdm(range(dataset_size), desc="Reading images"):
        for j in range(batch):
            img = cv2.imread(img_path_lst[i * batch + j])
            imgs[i * batch + j, :, :, :] = crop_face(img, [0, 0, img.width, img.height])
    return imgs


def create_pairs(imgs, batch, start, end, output_dir):
    """
    Slice dataset of size 500
    First 480 people for training set, 480-489 for evaluation set, last 10 people for test set
    Generate 625 positive and negative samples respectively for each person
    :param imgs: np.array, <>, array of all images in the dataset
    :param start: start index
    :param end: end index
    :param output_dir: output folder for generated image pairs
    """
    count_true = 0
    count_false = 0
    for x in tqdm(range(start, end)):
        same_person = imgs[x * batch:x * batch + batch, :, :, :]  # same person for every 25 images
        for img in same_person:
            for _img in same_person:
                new_img = np.concatenate([img, _img], 1)  # forming pairs
                new_img.imsave(os.path.join(output_dir, 'true', '%06d.jpg' % count_true))
                count_true += 1
            for _ in range(25):
                # select another person's image
                while True:
                    idx = np.random.randint(start * batch, end * batch)
                    if idx not in range(x * batch, x * batch + batch):
                        break
                new_img = np.concatenate([img, imgs[idx]], 1)
                new_img.imsave(os.path.join(output_dir, 'false', '%06d.jpg' % count_false))
                count_false += 1


def make_folder(dir):
    """
    Create true and false subdirectories for the given root directory
    :param dir: root directory
    """
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, 'true'), exist_ok=True)
    os.makedirs(os.path.join(dir, "false"), exist_ok=True)


def create_result_list(folder, output_dir, name):
    """
    Mark the matching of each image pairs in the given folder and create a list of results with
    random order
    :param name: train, val, or test
    :param folder: directory that contains matched and unmatched image pairs in two subdirectories
    :param output_dir: output folder for result list
    :return:
    """
    total = []
    for root, _, files in os.walk(os.path.join(folder, 'true')):
        for img in files:
            total.append('%s 1\n' % os.path.join(root, img))
    for root, _, files in os.walk(os.path.join(folder, 'false')):
        for img in files:
            total.append('%s 0\n' % os.path.join(root, img))
    random.shuffle(total)
    with open(os.path.join(output_dir, name + '.txt'), 'w', encoding='UTF-8') as f:
        for line in total:
            f.write(line)


class CASIA(Dataset):
    def __init__(self, list_dir, stage='train', data_size=10000):
        img_path_and_label = np.loadtxt(os.path.join(list_dir, stage+'.txt'), dtype=str)
        self.data_list = []
        transform = transforms.Compose([transforms.ToTensor()])
        for path, label in tqdm(img_path_and_label[:data_size], desc="Make {} dataset".format(stage)):
            img = transform(cv2.imread(path))
            img_pair = torch.split(img, split_size_or_sections=200, dim=2)
            self.data_list.append((img_pair, int(label)))

    def __getitem__(self, index):
        imgs, label = self.data_list[index]
        img1, img2 = imgs
        return img1, img2, label

    def __len__(self):
        return len(self.data_list)


def prepare_data(conf):
    imgs = read_images(conf.data_dir, 500, 5)
    # generate train dataset
    make_folder(conf.train_dir)
    create_pairs(imgs, 5, 0, 480, conf.train_dir)
    # generate validation dataset
    make_folder(conf.val_dir)
    create_pairs(imgs, 5, 480, 490, conf.val_dir)
    # generate test dataset
    make_folder(conf.test_dir)
    create_pairs(imgs, 5, 490, 500, conf.test_dir)

    # generate list of paths of pictures for each dataset
    os.makedirs(conf.list_dir, exist_ok=True)
    create_result_list(conf.train_dir, conf.list_dir, 'train')
    create_result_list(conf.val_dir, conf.list_dir, 'val')
    create_result_list(conf.test_dir, conf.list_dir, 'test')


if __name__ == '__main__':
    conf = get_config()
    prepare_data(conf)

