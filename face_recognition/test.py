import numpy as np
import predict, image_processing
import cv2

"""
测试新图片步骤：
1. 将check_path的路径更换为新照片的路径
2. 运行test.py
3. 如果结果显示为"新照片满足要求"，则新照片没问题，可以放到source文件夹中；
   如果结果显示为"xxx和xxx混淆了"，那么新照片不满足要求，需更换照片重新测试
4. 测试完所有照片并把座位表PS完后，座位表照片需放在original_images文件夹中，
   并且更换draw.py和display.py中的image_path的路径为新座位表照片的路径
"""

model_file = "inference/face_verification/__model__"
params_file = "inference/face_verification/__params__"
check_path = "data/crop/00104.jpg"  # 需要测试的照片的路径,替换掉这行
source_dirc = "data/source"


def doPredict(predictor, sourceImg, checkImg):
    img = np.concatenate([checkImg, sourceImg], 1)
    cv2.imshow("image", img)
    img = image_processing.preprocessing(img)
    return predict.predicting(predictor, img)


checkImg = cv2.imread(check_path)
sourceImgList, sourceImgPath = image_processing.readImage(source_dirc)

# h, w = img1.shape[:2]
# img1 = cv2.resize(img1, (200, int(h / w * 200))) if w > h else cv2.resize(img1, (int(w / h * 200), 200))
# h, w = img1.shape[:2]
# img2 = cv2.resize(img2, (200, int(h / w * 200))) if w > h else cv2.resize(img2, (int(w / h * 200), 200))

predictor = predict.create_predictor(model_file, params_file)

allGood = True;

for sourceImg, sourcePath in zip(sourceImgList, sourceImgPath):
    if sourcePath.split('/')[-1] == ".DS_Store": continue
    img = np.concatenate([sourceImg, checkImg], 1)
    cv2.imshow("image", img)
    cv2.waitKey(500)
    img = image_processing.preprocessing(img)

    output_data = predict.predicting(predictor, img)

    if np.argmax(output_data[0]):
        checkName = check_path.split('/')[-1].split('.')[0]
        sourceName = sourcePath.split('/')[-1].split('.')[0]
        print(checkName, "和", sourceName, "混淆了")
        allGood = False

if allGood:
    print("新照片满足要求")