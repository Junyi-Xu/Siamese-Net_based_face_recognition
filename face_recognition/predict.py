import paddle.inference as paddle_infer
import numpy as np

def create_predictor(model_file,params_file):
    # 创建config
    config = paddle_infer.Config(model_file,params_file)

    # 根据config创建predictor
    predictor = paddle_infer.create_predictor(config)

    return predictor

def predicting(predictor,img):
    # 将设置模型输入Tensor
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 从CPU获取数据，设置到Tensor内部
    # fake_input = np.random.randn(100, 3, 318, 318).astype("float32")
    input_handle.reshape([100, 3, 224, 224])
    input_handle.copy_from_cpu(img)

    # 运行predictor
    predictor.run()

    # 获取输出Tensor
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    # print("Output data size is {}".format(output_data.size))
    # print("Output data shape is {}".format(output_data.shape))
    return output_data

