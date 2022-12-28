import main


model_file = "inference/face_verification/__model__"
params_file = "inference/face_verification/__params__"
image_path = "data/original_images/seats.png"   # 更改为PS完的座位照的路径
source_dirc = "data/source"
out_dirc = "data/out"


matchDict, seats = main.doRecognize(model_file,params_file,image_path,source_dirc,out_dirc,True)
# print(matchDict)
# print(seats)