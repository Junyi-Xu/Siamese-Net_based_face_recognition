import torch
import os
import cv2
from network import Siamese
from face_detector import FaceDetector
from train import euclidean_distance
from config import get_config
from tools import crop_face, img_to_tensor, tensor_to_img


def get_predict_model(conf):
    model = Siamese(in_channels=3)
    model.load_state_dict(torch.load(os.path.join(conf.output_dir, 'model.pth')))
    return model


def detect_face(img_path):
    detector = FaceDetector()
    img, boundings = detector.detect_face(img_path)
    img_lst = []
    for i, box in enumerate(boundings):
        crop = crop_face(img, box)
        cv2.imwrite(os.path.join('out', '{}.jpg'.format(i)), crop)
        img_lst.append(img_to_tensor(crop))
    img_lst = torch.stack(img_lst, dim=0)
    return img_lst


def load_database(database_dir):
    img_lst = []
    for file in os.listdir(database_dir):
        img = cv2.imread(os.path.join(database_dir, file))
        img = crop_face(img, [0, 0, img.shape[1], img.shape[0]])
        img_lst.append(img_to_tensor(img))
    return img_lst


def predict(model, detected_face_lst, database_face_lst):
    results = []
    num_face = len(detected_face_lst)
    output_faces = model(detected_face_lst)
    for face in database_face_lst:
        duplicate_img = torch.tile(face, (num_face, 1, 1, 1))
        outputs_database = model(duplicate_img)
        distances = euclidean_distance((output_faces, outputs_database)).squeeze()
        matched_indices = torch.nonzero(distances < 0.5)[0]
        if len(matched_indices) == 1:
            idx = int(matched_indices[0])
            matched_face = detected_face_lst[idx, :, :, :]
            matched_face_pair = torch.cat((face, matched_face), dim=2)
            matched_face_pair = tensor_to_img(matched_face_pair)
            results += [True]
        else:
            results += [False]
    return results


def main(conf):
    model = get_predict_model(conf)
    database_img_lst = load_database(conf.database_dir)
    detected_face_lst = detect_face(conf.input_pic)
    print(predict(model, detected_face_lst, database_img_lst))


if __name__ == '__main__':
    conf = get_config()
    main(conf)
