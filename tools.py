import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


############################
# image processing functions
############################

def crop_face(img, box):
    left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h = bottom - top
    w = right - left
    margin = (h // 4, w // 4)
    crop = img[max(0, top - margin[0]):min(bottom + margin[0], img.shape[0]), max(0, left - margin[1]):min(right + margin[1], img.shape[1]), :]
    h, w, _ = crop.shape
    if crop.shape[1] > crop.shape[0]:
        crop = cv2.resize(crop, (200, int(h / w * 200)))
    else:
        crop = cv2.resize(crop, (int(w / h * 200), 200))
    # pad image with white margin to make it 200*200 pixels
    row_nums = 200 - crop.shape[0]
    line_nums = 200 - crop.shape[1]
    if row_nums % 2 == 0:
        crop = np.pad(crop, ((row_nums // 2, row_nums // 2), (0, 0), (0, 0)), 'constant')
    else:
        crop = np.pad(crop, ((row_nums // 2, row_nums // 2 + 1), (0, 0), (0, 0)), 'constant')
    if line_nums % 2 == 0:
        crop = np.pad(crop, ((0, 0), (line_nums // 2, line_nums // 2), (0, 0)), 'constant')
    else:
        crop = np.pad(crop, ((0, 0), (line_nums // 2, line_nums // 2 + 1), (0, 0)), 'constant')
    return crop


def crop_store_face(detector, input_dir='../data/all_face', output_dir='../data/crop'):
    for root, dirs, imgs in os.walk(input_dir):
        for img in tqdm(imgs):
            name = img
            img = cv2.imread(os.path.join(root, img))
            bounding = detector.detect_face(img)
            crop = detector.crop_face(img, bounding)
            cv2.imwrite(os.path.join(output_dir, name), crop)


def rotate(img, angle):
    """
    Rotate image by given angle
    :param img: image to be rotated
    :param angle: angle of rotation
    :return: image after being rotated by given angle
    """
    imgInfo = img.shape
    height, width, _ = imgInfo[0], imgInfo[1], imgInfo[2]
    matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), angle, 0.9)
    dst = cv2.warpAffine(img, matRotate, (width, height))
    return dst


def rotate_intensify(input_dir, output_dir='../data/all_face'):
    count = 0
    for root, dirs, imgs in tqdm(os.walk(input_dir)):
        for img in imgs:
            img = cv2.imread(os.path.join(root, img), 1)
            for angle in range(-20, 25, 10):
                dst = rotate(img, angle)
                cv2.imwrite(os.path.join(output_dir, '%05d.jpg' % count), dst)
                count += 1


def adjust_img(img, img_size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = img.resize((img_size, img_size), Image.LANCZOS)  # shrink image

    # HWC to CHW
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = np.array(image).astype('float32')
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)

    # normalization
    image /= 255
    image -= mean
    image /= std
    image = image[[0, 1, 2], :, :]
    image = np.expand_dims(image, axis=0).astype('float32')
    return image
