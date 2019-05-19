import numpy as np
import os
from shutil import copyfile
from PIL import Image

"""

This one takes simple the Hubble Legacy Field Cropped and breaks it into much smaller 256x256 blocks for use in the GAN



"""

img_size = 64
file_loc = "/home/jacob/Development/AstroGAN/STSCI-H-p1917b-f-20791x19201.png"
#file_loc = "/home/jacob/Development/AstroGAN/heic1502a.tif"
num_photos = 10000
Image.MAX_IMAGE_PIXELS = 511520000

image = Image.open(file_loc)
w, h = image.size
image = image.resize((int(w/25),int(h/25)))
w, h = image.size


for i in range(num_photos):
    # Create random crops and save to directory
    x = np.random.randint(0, w-img_size-1)
    y = np.random.randint(0, h-img_size-1)
    newname = "{}.png".format(i)
    print("Cropping: {},{} -> {},{}".format(x,y, x+img_size, y+img_size))
    image.crop((x,y, x+img_size, y+img_size)) \
        .save(os.path.join("h/crops", newname), "PNG")
