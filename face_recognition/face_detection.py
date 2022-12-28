import cv2,os
import paddlehub as hub
import numpy as np

def identifySeat(position):
    allSeats = [[(981,767),(1354,754),(1733,772),(2121,767),(2783,771),(2877,782),(3237,780)],
                [(837, 976),(1275, 937), (1698, 997),(2157,974),(2510,974),(2981,966),(3384,1005)],
                [(669,1205),(1142,1235),(1645,1211),(2157,1230),(2596,1278),(3106,1241),(3580,1271)],
                [(431,1631),(1000,1609),(1571,1622),(2166,1624),(2711,1642),(3289,1609),(3841,1682)]]

    # print(position)
    seats = []
    for personX,personY in position:
        for row in range(len(allSeats)):
            for col in range(len(allSeats[row])):
                x,y = allSeats[row][col]
                if abs(x-personX) < 100 and abs(y-personY) < 100:
                    seats.append((row,col))
    return seats


def storeCropImg(img,result,outputDirc):
    count = 0
    position = []
    for pic in result[0]['data']:
        left = pic['left']
        right = pic['right']
        top = pic['top']
        bottom = pic['bottom']
        h = bottom-top
        w = right-left
        position.append((left,top))
        crop = img[max(0, top - h//2):min(bottom + h//2, img.shape[0]), max(0, left - w//2):min(right + w//2, img.shape[1]), :]
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
        name = str(count)+".jpg"
        cv2.imwrite(os.path.join(outputDirc, name), crop)
        count+=1
    return position

def runFaceDetector(inputPath,outputDirc,show):
    # load Paddlehub human face detection module
    face_detector = hub.Module(name="pyramidbox_lite_mobile")

    # use module to do face detection
    img = cv2.imread(inputPath)
    result = face_detector.face_detection(images=[img], use_gpu=True, visualization=True,
                                          output_dir='data/detection_result')

    if show:
        path = result[0]['path'].split(".")[0] + ".jpg"
        resultImage = os.path.join("data/detection_result", path)
        resultImage = cv2.imread(resultImage)
        cv2.imshow("resultImage", resultImage)
        cv2.waitKey(1000)

    # store cropped face images and get coordinate of each face
    position = storeCropImg(img, result, outputDirc)

    seats = identifySeat(position)

    return seats