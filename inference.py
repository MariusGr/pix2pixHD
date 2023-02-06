import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath)
print(sys.path)

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
from PIL import Image
import util.util as util
import numpy as np
from PIL import Image

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# default inference options
opt.name = "label2city_1024p"
opt.no_instance = True
opt.label_nc = 0
opt.resize_or_crop = "crop"
opt.fineSize = 1024
opt.dataroot = "./pix2pixHD/datasets/cityscapes/"
opt.checkpoints_dir = "./pix2pixHD/checkpoints/"

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)

def scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def generate(image):
    image_tensor = dataset.dataset.image2tensor(image)

    generated = model.inference(
        image_tensor,
        image_tensor,
    )

    output = util.tensor2im(generated)
    output = Image.fromarray(np.uint8(output)).convert('RGB')
    return output

def main():
    onlyfiles = [os.path.join('./datasets/cityscapes/train_A/', f) for f in os.listdir('./datasets/cityscapes/train_A/')]
    image = Image.open(onlyfiles[0])
    output = generate(image)
    output.show()

if __name__ == '__main__':
    main()