import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)

if opt.data_type == 16:
    data['label'] = data['label'].half()
    data['inst'] = data['inst'].half()
elif opt.data_type == 8:
    data['label'] = data['label'].uint8()
    data['inst'] = data['inst'].uint8()

def main():
    generated = model.inference(data['label'], data['inst'], data['image'])



if __name__ == '__main__':
    main()