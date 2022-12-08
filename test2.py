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
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow

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

def main():
    print(dataset)
    
    image = Image.open('./datasets/cityscapes/test_A/1_I.jpg')
    # image = scale_width(image, 2048)
    # image = crop(image, (0, 1080 - 1024), (2048, 1024))
    image_tensor = dataset.dataset.image2tensor(image)
    # image = np.asarray(image)
    # print(image)
    # # (1, 1, 1024, 2048)
    # # image = torch.tensor(image, dtype=torch.uint8)
    print(image)
    print(image.size)
    print("-"*100)
    print(image_tensor)
    print(image_tensor.size())

    generated = model.inference(
        image_tensor,
        image_tensor,
    )

    print(generated)
    print(generated.shape)
    output = util.tensor2im(generated)
    print(output)
    print(output.shape)
    output = Image.fromarray(np.uint8(output)).convert('RGB')

    imshow(np.asarray(output))
    # output.show()


if __name__ == '__main__':
    main()