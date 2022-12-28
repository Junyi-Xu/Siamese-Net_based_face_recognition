import cv2
import os, random
import paddlehub as hub
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 图像旋转函数
def Rotate(img, angle):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5), angle, 0.9)
    dst = cv2.warpAffine(img,matRotate,(width,height))
    return dst

# 旋转增强
def rotateIntensify():
    data_dir = 'data/CASIA-FaceV5'
    person_list = os.listdir(data_dir)
    person_list.sort()
    count = 0
    for person in tqdm(person_list):
        if(person != '.DS_Store'):
            img_list = os.listdir(os.path.join(data_dir,person))
            img_list.sort()
            for img in img_list:
                img = os.path.join(data_dir, person, img)
                img = cv2.imread(img, 1)
                for angle in range(-20, 25, 10):
                    dst = Rotate(img, angle)
                    cv2.imwrite(os.path.join('data/all_face', '%05d.jpg' % count), dst)
                    count += 1

# 截取人脸图片并存储
def cropFace():
    face_detector = hub.Module(name="pyramidbox_lite_mobile")
    img_list = os.listdir('data/all_face')
    img_list.sort()
    imgs = []
    for img in img_list:
        imgs.append(os.path.join('data/all_face',img))
    for img in tqdm(imgs):
        name = img.split('/')[-1]
        img = cv2.imread(img)
        result = face_detector.face_detection(images=[img],use_gpu=True)
        locals().update(result[0]['data'][0])
        top = result[0]['data'][0]['top']
        bottom = result[0]['data'][0]['bottom']
        left = result[0]['data'][0]['left']
        right = result[0]['data'][0]['right']
        # 上下左右各扩大50个像素，截取较为完整的人脸
        crop = img[max(0, top - 50):min(bottom + 50, img.shape[0]), max(0, left - 50):min(right + 50, img.shape[1]), :]
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (200, int(h / w * 200))) if w > h else cv2.resize(crop, (int(w / h * 200), 200))
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
        cv2.imwrite(os.path.join('data/crop', name), crop)

def readImage(dirc):
    img_list = os.listdir(dirc)
    img_list.sort()
    imgPath = []
    for img in img_list:
        imgPath.append(os.path.join(dirc,img))
    imgList = []
    for path in imgPath:
        imgList.append(cv2.imread(path))
    return imgList,imgPath

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = img.resize((224,224), Image.LANCZOS)    # shrink image

    # HWC to CHW
    mean = np.array([0.485,0.456,0.406]).reshape(3, 1, 1)
    std = np.array([0.229,0.224,0.225]).reshape(3, 1, 1)
    image = np.array(image).astype('float32')
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)

    # 归一化
    image /= 255
    image -= mean
    image /= std
    image = image[[0, 1, 2], :, :]
    image = np.expand_dims(image, axis=0).astype('float32')
    return image

