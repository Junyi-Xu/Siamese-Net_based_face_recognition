import numpy as np
import torch
import os
import cv2
from torchvision import transforms
from network import Siamese
from face_detector import FaceDetector
from train import euclidean_distance
from config import get_config
from tools import crop_face


def get_predict_model(conf):
    model = Siamese(in_channels=3)
    model.load_state_dict(torch.load(os.path.join(conf.output_dir, 'model.pth')))
    return model


def detect_face(img_path):
    detector = FaceDetector()
    img, boundings = detector.detect_face(img_path)
    img_torch_lst = []
    img_lst = []
    for i, box in enumerate(boundings):
        crop = crop_face(img, box)
        img_lst.append(crop)
    return img_lst


def load_database(database_dir):
    img_lst = []
    name_lst = []
    for file in os.listdir(database_dir):
        name_lst.append(file.split('.')[0])
        img = cv2.imread(os.path.join(database_dir, file))
        img = crop_face(img, [0, 0, img.shape[1], img.shape[0]])
        img_lst.append(img)
    return img_lst, name_lst


def predict(model, detected_face_lst, database_face_lst, database_name_lst, output_path):
    transform = transforms.Compose([transforms.ToTensor()])
    detected_torch = torch.stack(list(map(transform, detected_face_lst)), dim=0)
    database_torch_lst = list(map(transform, database_face_lst))
    matched_person_lst = []
    missed_person_lst = []
    num_face = len(detected_torch)
    output_faces = model(detected_torch)
    for i, face in enumerate(database_torch_lst):
        duplicate_face = torch.tile(database_torch_lst[i], (num_face, 1, 1, 1))
        outputs_database = model(duplicate_face)
        distances = euclidean_distance((output_faces, outputs_database)).squeeze()
        print(distances)
        matched_indices = torch.nonzero(distances < 0.5)
        name = database_name_lst[i]
        if torch.numel(matched_indices) == 1:
            idx = int(matched_indices[0][0])
            matched_face_pair = np.concatenate((database_face_lst[i], detected_face_lst[idx]), axis=1)
            cv2.imwrite(os.path.join(output_path, name+'.jpg'), matched_face_pair)
            matched_person_lst += [name]
        else:
            missed_person_lst += [name]
    return matched_person_lst, missed_person_lst


def main(conf):
    model = get_predict_model(conf)
    database_face_lst, database_name_lst = load_database(conf.database_dir)
    detected_face_lst = detect_face(conf.input_pic)
    os.makedirs(conf.output_pic, exist_ok=True)
    matched_person_lst, missed_person_lst = predict(model, detected_face_lst, database_face_lst, database_name_lst, conf.output_pic)
    print('People found: ', end='')
    for name in matched_person_lst:
        print(name, end=', ')
    print()
    print('People missing: ', end='')
    for name in missed_person_lst:
        print(name, end=', ')


if __name__ == '__main__':
    conf = get_config()
    main(conf)
