import image_processing, predict, face_detection
import numpy as np
import cv2


def prepareImg(sourceDirc,outDirc):
    sourceImgs,sourcePath = image_processing.readImage(sourceDirc)
    checkImgs,checkPath = image_processing.readImage(outDirc)
    return sourceImgs,checkImgs,sourcePath,checkPath

# pass image from data set, image to be checked, and the predictor
def doPredict(predictor,sourceImg,checkImg):
    img = np.concatenate([checkImg,sourceImg],1)
    img = image_processing.preprocessing(img)
    return predict.predicting(predictor,img)

def doPredictAndShow(predictor,sourceImg,checkImg):
    img = np.concatenate([checkImg,sourceImg ],1)
    cv2.imshow("image",img)
    cv2.waitKey(500)
    img = image_processing.preprocessing(img)
    return predict.predicting(predictor,img)

def doMatch(model_file,params_file,sourceDirc,outDirc,show):
    sourceImgs,checkImgs,sourcePath,checkPath = prepareImg(sourceDirc,outDirc)
    predictor = predict.create_predictor(model_file,params_file)
    matchDict = dict()
    for i in range(1,len(sourceImgs)):
        found = False;
        name = sourcePath[i].split('/')[-1].split('.')[0]
        print(name,end=" ")
        for j in range(1,len(checkImgs)):
            if show:
                output_data = doPredictAndShow(predictor,sourceImgs[i],checkImgs[j])
            else:
                output_data = doPredict(predictor,sourceImgs[i],checkImgs[j])
            if np.argmax(output_data[0]):
                found = True
                # print(j-1,end=" ")
                matchDict[j-1] = sourcePath[i]
                print("出席")
                break
        if not found:
            print("缺席")
    cv2.destroyAllWindows()
    return matchDict


def doRecognize(model_file,params_file,imagePath,sourceDirc,outDirc,show=False):
    seats = face_detection.runFaceDetector(imagePath,outDirc,show)
    matchDict = doMatch(model_file,params_file,sourceDirc,outDirc,show)
    return matchDict,seats
