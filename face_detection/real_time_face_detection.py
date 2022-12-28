# 导入必要的库
import cv2
import paddlehub as hub


# 加载Paddlehub人脸检测模型
face_detector = hub.Module(name="pyramidbox_lite_mobile")

# 调用摄像头，参数为0时，即调用系统默认摄像头，如果有其他的摄像头可以调整参数为1，2等
cap=cv2.VideoCapture(0)

# 初始化掌控板
import time
from pinpong.board import Board, LED
Board("handpy").begin()
esp32 = LED()

while True:
    # 从摄像头读取图片
    sucess,img=cap.read()
    # 从图片中检测人脸位置，默认开启GPU推理，若无GPU环境，请将use_gpu设置为False
    result = face_detector.face_detection(images=[img],use_gpu=False)
    # 遍历结果并绘制矩形框
    if result[0]['data'] != []:
        for face in result[0]['data']:
            # 将Dict形式的key-value对转换成变量形式
            locals().update(face)
            print('bbox:',[left,top,right,bottom])  # 打印人脸位置的坐标
            cv2.rectangle(img, tuple([left,top]), tuple([right,bottom]), (255, 0, 0), 2)
            esp32.set_rgb_color(-1, 255, 0, 0)  # 如果识别到人脸，设置LED灯为红色
    else:
        esp32.rgb_disable(-1)  # 如果没有识别到人脸，关闭LED灯

    # 显示图像
    cv2.imshow("img",img)
    #保持画面的持续。

    k=cv2.waitKey(1)
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break

#关闭摄像头
cap.release()