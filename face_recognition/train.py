import paddlehub as hub
import paddle,os
from paddlehub.dataset.base_cv_dataset import BaseCVDataset

os.environ["CPU_NUM"] = "4"


# 自定义数据集
class FaceDataset(BaseCVDataset):
   def __init__(self):
       # 数据集存放位置
       self.dataset_dir = "data/"
       super(FaceDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train_modify.txt",
           validate_list_file="dev_modify.txt",
           test_list_file="test_modify.txt",
           label_list=['0','1']
           )


dataset = FaceDataset()

paddle.enable_static()

# 使用mobilenet_v3_large_imagenet_ssld预训练模型进行finetune
module = hub.Module(name="mobilenet_v3_large_imagenet_ssld")

# 数据读取器
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)


# 优化器配置
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=1e-3,
    lr_scheduler="linear_decay",
    warmup_proportion=0.1,
    weight_decay=0.0001,
    optimizer_name="adam")

# 总体配置
config = hub.RunConfig(
    use_cuda=False,
    num_epoch=10,
    checkpoint_dir="mobilenet_v3",
    batch_size=100,
    eval_interval=100,
    strategy=strategy)

# 任务构建
input_dict, output_dict, program = module.context(trainable=True)

img = input_dict["image"]

feature_map = output_dict["feature_map"]

feed_list = [img.name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)

# 开启训练
task.finetune_and_eval()

# 加载best_model
task.init_if_load_best_model()


# 导出推理模型
task.save_inference_model(
    dirname='inference/face_verification',
    model_filename='__model__',
    params_filename='__params'
)