import argparse

parser = argparse.ArgumentParser(description='Siamese Network Training')
# Datasets
parser.add_argument('--data_dir', default="dataset/CASIA-FaceV5-Crop", type=str)
parser.add_argument('--test_dir', default="dataset/test", type=str)
parser.add_argument('--train_dir', default="dataset/train", type=str)
parser.add_argument('--validate_dir', default="dataset/val", type=str)
parser.add_argument('--list_dir', default="dataset/list", type=str)

# Param
parser.add_argument('-b','--batch-size', default=10, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=0.00001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-j', '--num_workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer-eps', default=1e-8, type=float)
parser.add_argument('--evaluate', action='store_true', help='evaluation mode')
parser.add_argument('--output_dir', default='inference', type=str, help='output folder of trained model')


# Application
parser.add_argument('--database_dir', default="database", type=str)
parser.add_argument('--input_pic', default="picture/multiface.jpg", type=str, help='path of input picture')

def get_config():
    return parser.parse_args()
