import cv2
import paddlehub as hub
import numpy as np
import image_processing,predict

face_detector = hub.Module(name="pyramidbox_lite_mobile")

cap = cv2.VideoCapture(0)

while True:
    sucess,img = cap.read()
    result = face_detector.face_detection(images=[img],use_gpu=False)

    for pic in result[0]['data']:
        left = pic['left']
        right = pic['right']
        top = pic['top']
        bottom = pic['bottom']
        h = bottom-top
        w = right-left
        crop = img[max(0, top - h//3):min(bottom + h//3, img.shape[0]), max(0, left - w//3):min(right + w//3, img.shape[1]), :]
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

        crop = image_processing.preprocessing(crop)
        predictor = predict.create_predictor("/Users/junyixu/Desktop/001-Processing/face_verification/__model__", "/Users/junyixu/Desktop/001-Processing/face_verification/__params__")
        output_data = predict.predicting(predictor, crop)