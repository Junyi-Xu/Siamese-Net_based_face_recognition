import cv2
from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    def __init__(self, image_size=200):
        # create a face detection pipeline using MTCNN
        self.mtcnn = MTCNN(image_size=image_size,
                           margin=0,
                           min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7],
                           factor=0.709, post_process=True,
                           device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def detect_face(self, img_path):
        img = cv2.imread(img_path)
        boundings = self.mtcnn.detect(img)[0]
        return img, boundings

    def draw_bounding(self, img, boundings):
        image = img.copy()
        for box in boundings:
            left, top, right, bottom = box[:]
            image = cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        return image
