import torch

PARAMETERS_NAME = ['--num_iter', '--split_mode', '--num_epochs', '--batch_size', '--learning_rate', '--weights', '--k_groups', '--freeze_backbone']
DEFAULT_VALUES = [1, 'Hold-out', 30, 32, 0.0001, [0.7, 0.1], 5, True]
TYPES = ['int', 'string', 'int', 'int', 'float', 'list', 'int', 'bool']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224)
DATA_FOLDER = '../images/'

def printHelp():
    print("""
Usage: python src/main.py <MODEL> [options]

Models:
  RESNET        ResNet18 with transfer learning (pretrained on ImageNet)
  EFFICIENTNET  EfficientNet-B0 with transfer learning (pretrained on ImageNet)
Dataset format (place in images/ folder):
  Option A - Subfolders by count:  images/0/, images/1/, images/2/, ...
  Option B - CSV annotations file: images/annotations.csv  (columns: path, count)

Options:
  --num_iter=<n>           Number of full experiment iterations (default: 1)
  --split_mode=<mode>      Hold-out | K-fold cross-validation (default: Hold-out)
  --num_epochs=<n>         Training epochs (default: 30)
  --batch_size=<n>         Batch size (default: 32)
  --learning_rate=<lr>     Learning rate (default: 0.0001)
  --weights=<w1,w2>        Train/val proportions, e.g. 0.7,0.1 (default: 0.7,0.1)
  --k_groups=<k>           Number of folds for k-fold (default: 5)
  --freeze_backbone=<b>    Freeze backbone weights: True | False (default: True)
""")

def printError(error):
    if error:
        printHelp()
        exit(1)

def setParameter(args, param_name, default_value, type_str):
    for arg in args:
        if arg.startswith(param_name + '='):
            val = arg.split('=', 1)[1]
            if type_str == 'int':
                return int(val)
            elif type_str == 'float':
                return float(val)
            elif type_str == 'bool':
                return val.lower() in ('true', '1', 'yes')
            elif type_str == 'list':
                return list(map(float, val.split(',')))
            else:
                return str(val)
    return default_value

def getConfigs(args):
    return [setParameter(args, name, default, t)
            for name, default, t in zip(PARAMETERS_NAME, DEFAULT_VALUES, TYPES)]