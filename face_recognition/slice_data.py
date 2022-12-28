# 切分数据集
# 前480个人为训练集，480-489为验证集，最后10人为测试集
# 两两配对每个人生成625张正样本，并随机生成625张负样本

import os, cv2
import numpy as np
from tqdm import tqdm
import random

imgs = []
img_list = os.listdir('data/crop')
img_list.sort()
for img in img_list:
    img = os.path.join('data/crop',img)
    img = cv2.imread(img)
    imgs.append(img)

imgs = np.array(imgs)

count_true = 0
count_false = 0
for x in tqdm(range(480)):
    same = imgs[x*25:x*25+25,:,:,:]
    for img in same:
        for _img in same:
            new_img = np.concatenate([img,_img],1)
            cv2.imwrite(os.path.join('data/train/true','%06d.jpg' % count_true),new_img)
            count_true+=1
        for _ in range(25):
            while True:
                index = np.random.randint(0,12000)
                if index not in range(x*25,x*25+25): break
            new_img = np.concatenate([img,imgs[index]],1)
            cv2.imwrite(os.path.join('data/train/false','%06d.jpg' % count_false),new_img)
            count_false += 1
for x in tqdm(range(480,490)):
    same = imgs[x*25:x*25+25,:,:,:]
    for img in same:
        for _img in same:
            new_img = np.concatenate([img,_img],1)
            cv2.imwrite(os.path.join('data/dev/true','%06d.jpg' % count_true),new_img)
            count_true += 1
        for _ in range(25):
            while True:
                index = np.random.randint(12000, 12250)
                if index not in range(x*25,x*25+25):
                    break
            new_img = np.concatenate([img,imgs[index]],1)
            cv2.imwrite(os.path.join('data/dev/false','%06d.jpg' % count_false),new_img)
            count_false += 1

for x in tqdm(range(490,500)):
    same = imgs[x*25:x*25+25,:,:,:]
    for img in same:
        for _img in same:
            new_img = np.concatenate([img,_img],1)
            cv2.imwrite(os.path.join('data/test/true','%06d.jpg' % count_true),new_img)
            count_true += 1
        for _ in range(25):
            while True:
                index = np.random.randint(12250, 12500)
                if index not in range(x*25,x*25+25):
                    break
            new_img = np.concatenate([img,imgs[index]],1)
            cv2.imwrite(os.path.join('data/test/false','%06d.jpg' % count_false),new_img)
            count_false += 1

total = []

true_list = os.listdir('data/train/true')
true_list.sort()
for item in true_list:
    total.append('%s 1\n' % os.path.join('train','true',item))

false_list = os.listdir('data/train/false')
false_list.sort()
for item in false_list:
    total.append('%s 0\n' % os.path.join('train','false',item))

random.shuffle(total)

with open('data/train.txt','w',encoding='UTF-8') as f:
    for line in total:
        f.write(line)

# 生成数据集列表
total = []

true_list = os.listdir('data/dev/true')
true_list.sort()
for item in true_list:
    total.append('%s 1\n' % os.path.join('dev','true',item))

false_list = os.listdir('data/dev/false')
false_list.sort()
for item in false_list:
    total.append('%s 0\n' % os.path.join('dev','false',item))

random.shuffle(total)

with open('data/dev.txt','w',encoding='UTF-8') as f:
    for line in total:
        f.write(line)

total = []

true_list = os.listdir('data/test/true')
true_list.sort()
for item in true_list:
    total.append('%s 1\n' % os.path.join('test','true',item))

false_list = os.listdir('data/test/false')
false_list.sort()
for item in false_list:
    total.append('%s 0\n' % os.path.join('test','false',item))

random.shuffle(total)

with open('data/test.txt','w',encoding='UTF-8') as f:
    for line in total:
        f.write(line)