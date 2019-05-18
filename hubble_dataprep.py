import numpy as np
import os
from shutil import copyfile
from PIL import Image


"""

This one takes simple the Hubble Legacy Field Cropped and breaks it into much smaller 256x256 blocks for use in the GAN



"""

img_size = 256
file_loc = ""
num_photos = 100000

image = Image.open(file_loc)
w, h = image.size


for i in range(num_photos):
    # Create random crops and save to directory
    x = np.random.randint(0, w-img_size-1)
    y = np.random.randint(0, h-img_size-1)
    newname = "{}.png".format(i)
    print("Cropping: {},{} -> {},{}".format(x,y, x+img_size, y+img_size))
    image.crop((x,y, x+img_size, y+img_size)) \
        .save(os.path.join("hubble/crops", newname), "PNG")
