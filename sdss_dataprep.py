import numpy as np
import os
from shutil import copyfile
from PIL import Image

datafile = np.genfromtxt("data/training_solutions_rev1.csv", dtype=float, delimiter=',', names=True)
print(datafile.dtype.names)


# Now have all the IDs, split into 512x512 images by type, as well as 424x424 by type

def move_by_type(solutions, image_dir):
    spiral_dir = os.path.join(image_dir, "spiral")
    elliptical_dir = os.path.join(image_dir, "elliptical")
    trash_dir = os.path.join(image_dir, "trash")
    for index, id in enumerate(solutions['GalaxyID']):
        file = os.path.join(image_dir, str(int(id)) + ".jpg")
        # now check which broad classification is best
        is_elliptical = solutions['Class11'][index]
        is_spiral = solutions['Class12'][index]
        is_disk_edge = solutions['Class41'][index]
        is_bar = solutions['Class31'][index]
        is_trash = solutions['Class13'][index]
        if is_spiral > 0.7 and is_bar > 0.7 and is_disk_edge > 0.8:
            copyfile(file, os.path.join("pure_spiral/sp", str(int(id)) + ".jpg"))
        #elif is_elliptical > 0.9:
        #    copyfile(file, os.path.join(elliptical_dir, str(int(id)) + ".jpg"))
        #elif is_trash > 0.3: # Trash biggest
        #    copyfile(file, os.path.join(trash_dir, str(int(id)) + ".jpg"))


def move_for_nvidia(image_dir, out_dir, res=(512,512)):
    """
    Copies all images in the given directory in a subdirectory of high-resolution images
    For use with Nvidia's ProGAN

    Assumes only images in the folder

    :param image_dir:
    :return:
    """
    subdir = out_dir
    for file in os.listdir(image_dir):
        im = Image.open(os.path.join(image_dir, file))
        out = im.resize(res)
        out.save(os.path.join(subdir, file), "JPEG")

move_by_type(datafile, "data/images_training_rev1")
#move_for_nvidia("data/images_training_rev1/spiral", out_dir="spiral_high_res")
#move_for_nvidia("data/images_training_rev1/elliptical", out_dir="elliptical_high_res")
#move_for_nvidia("data/images_training_rev1/trash", out_dir="trash_high_res")
#move_for_nvidia("/home/jacob/Development/nvidia_all", "high_res")