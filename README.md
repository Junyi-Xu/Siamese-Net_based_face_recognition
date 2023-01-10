# Attendance Checking using Siamese Network
This project enables the use of one image to check the attendance of all people in the database.
The output will indicate who in the database is missing.
The algorithm is based on Siamese Network and MTCNN facial detection model.

## Requirements
- Python 3.8
- PyTorch
- facenet_pytorch
- Check the required python packages in `requirements.txt`
```
pip install -r requirements.txt
```


## Data Preparation and Preprocessing
This project uses the dataset [CASIA-FaceV5](http://biometrics.idealtest.org/findTotalDbByMode.do?mode=Face#/datasetDetail/9) for training, 
validation, and testing. The dataset image should be placed in `dataset/CASIA_FaceV5_Crop` or you can change the dataset path in `config.py`.

The dataset contains 500 people's face image sets, among which the first 480 sets will be used for training, sets from 481 to 490 will be used for 
validation, and the last 10 sets will be used for testing. Two images of same person and different people will be paired side-by-side and marked 
(0 if same person, 1 if different people).

Run `dataset.py` to slice the dataset.
```
python dataset.py
```
The resulting training, validation, and testing images will be stored in `dataset/train`, `dataset/val`, 
`dataset/test`. A list of all image paths and labels will be generated and stored in `dataset/list` as txt file for training, 
validation, and testing dataset respectively. Each contains a `true` folder that stores images of the same person and a `false` folder that stores 
images of different people.

```text
dataset
├── CASIA_FaceV5_Crop
    └── 000
        ├── 000_0.bmp
        ├── 000_1.bmp
        ├── 000_2.bmp
        ├── 000_3.bmp
        └── 000_4.bmp    
├── train
    ├── false
    └── true 
├── val
├── test
└── list
    ├── train.txt
    ├── test.txt
    └── val.txt
```

## Training and Testing
To train the Siamese network, run:
```
python train.py -b 10 -e 5 -lr 0.00001 -wd 5e-4
```

To test the Siamese network, run:
```
python test.py
```

The resulting model can achieve 85.8% testing accuracy and will be stored in `inference/model.ph`.



## Model application
To do attendance checking, run
```
python predict.py --input_pic picture/multiface.jpg, --database_dir database
```

You can change the picture used for face detection and recognition by changing the path fed as `--input_pic`.
You may also check the visualized results of face matching between detected face and database image in `output`.


## References
1. Tim Esler's facenet_pytorch repo: https://github.com/timesler/facenet-pytorch
2. Geol Choi's face-recognition-using-siamese-network repo: https://github.com/gchoi/face-recognition-using-siamese-network

