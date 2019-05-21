import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
#set_session(tf.Session(config=config))
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.utils import shuffle
import numpy as np
from keras.models import load_model


class ImageGenerator:
    def __init__(self, model_name, latent_space, save_dir):
        self.model_name = model_name
        self.latent_space = latent_space
        self.save_dir = save_dir

        self.generator = load_model(model_name)

    def gen_image(self, mean=0, sigma=1):
        noise = np.random.normal(mean, sigma, (1, self.latent_space))
        return self.generator.predict(noise)

    def gen_image_range(self, num_images=5, trunc_range=(2,0.04)):
        sigmas = np.linspace(trunc_range[0], trunc_range[1], num_images)
        images = []
        for sigma in sigmas:
            images.append(self.gen_image(mean=0, sigma=sigma))

        return images

#sdss_Spiralgen = ImageGenerator("/home/jacob/Development/AstroGAN/PureSpiral_422_generator_B64_Pix64_Latent100_D1024_UP4.h5", latent_space=100, save_dir="./")
sdss_Elipgen = ImageGenerator("/home/jacob/Development/AstroGAN/Elip_176_generator_B64_Pix64_Latent100_D1024_UP4.h5", latent_space=100, save_dir="./")

print(sdss_Elipgen.gen_image(0,1)[:,:,:].shape)
plt.imshow(0.5 + 0.5 * sdss_Elipgen.gen_image(0,2.)[:,:,:].reshape((64,64,3)))
plt.show()


hubble_128gen = ImageGenerator("/home/jacob/Development/AstroGAN/HubbleOne_539_generator_B32_Pix128_Latent100_D1024_UP5.h5", latent_space=100, save_dir="./")
plt.imshow(0.5 + 0.5 * hubble_128gen.gen_image(0,2.)[:,:,:].reshape((128,128,3)))
plt.show()