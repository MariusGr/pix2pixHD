
import inference
import dir_helper
import os
from PIL import Image


def main():
    onlyfiles = [os.path.join('./datasets/cityscapes/train_A/', f) for f in os.listdir('./datasets/cityscapes/train_A/')]
    for file in onlyfiles:
        image = Image.open(file)
        output = inference.generate(image)

        dir_helper.create_dir("./generated")
        pathname, extension = os.path.splitext(file)
        name = pathname.split('/')[-1]

        output.save("generated/output_"+name+".jpg", "JPEG")
        #output.show()


if __name__ == '__main__':
    main()