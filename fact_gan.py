"""

GAN for generating new images from FACT eventfiles

"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))

from factnn.utils.cross_validate import get_chunk_of_data, get_data_generators

directory = "/home/jacob/iact_events/"
gamma_dir = [directory + "gammaFeature/clump5/"]
proton_dir = [directory + "protonFeature/clump5/"]


from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from multiprocessing import Process, Manager



import numpy as np

def add_to_list(L, generator, index):
    L.append(generator[index])

class DCGAN():
    def __init__(self, width=424, height=424, channels=1, directory="", latent=100, dense=128, num_upscales=2, batch_size=32):
        # Input shape
        self.img_rows = width
        self.dense = dense
        self.img_cols = height
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent
        self.directory = directory
        self.batch_size = batch_size
        self.num_upscales = num_upscales

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(num_upscales=self.num_upscales)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        print(self.combined.summary())

    def datagen(self, directory):
        """
        Generates the images for the GAN from the SDSS ones

        :param directory:
        :return:
        """

        train_gen, _, _, _ = get_data_generators(directory=gamma_dir, max_elements=50000,
                                                 indicies=(30, 129, 1), rebin=self.img_rows,
                                                 batch_size=self.batch_size, as_channels=True,
                                                 model_type="Energy")
        return train_gen

    def build_generator(self, num_upscales=2):

        model = Sequential()

        model.add(Dense(self.dense * int(self.img_cols/(2**num_upscales)) * int(self.img_rows/(2**num_upscales)), activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((int(self.img_cols/(2**num_upscales)), int(self.img_rows/(2**num_upscales)), self.dense)))
        model.add(UpSampling2D())
        #model.add(Conv2D(1024, kernel_size=3, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        #model.add(UpSampling2D())
        #model.add(Conv2D(512, kernel_size=3, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        #model.add(UpSampling2D())
        #model.add(Conv2D(256, kernel_size=3, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation("relu"))
        #model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        #model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        #model.add(Conv2D(1024, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        # Add Minibatch discrimination here?
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def process_fact(self, save_que, image_gen):
        while True:
            index = np.random.randint(0, len(image_gen))
            imgs, _ = image_gen[index]
            save_que.put(imgs)


    def train(self, epochs, save_interval=50):

        # Load the dataset
        # Load it as a generator
        image_gen = self.datagen(self.directory)

        ran_num = np.random.randint(0,1000)
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        num_samples = len(image_gen)
        total = 1
        processes = []

        n_processes = 12

        manager = Manager()
        save_que = manager.Queue(50)

        for _ in range(n_processes):
            processes.append(Process(target=self.process_fact, args=(save_que,image_gen)))

        for p in processes:
            p.start()

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select random images from batch_size
            if epoch*12 >= total*num_samples - 1:
                image_gen.on_epoch_end()

            i = 10
            while i > 0:
                imgs = save_que.get()
                # Rescaled between 1 and -1 per batch
                imgs = 2*(imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))-1

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Adversarial ground truths
                valid = np.ones((imgs.shape[0], 1))
                fake = np.zeros((imgs.shape[0], 1))

                # Split into combined real and fake of batch_size
                #all_imgs = np.concatenate((imgs, gen_imgs), axis=0)
                #all_labels = np.concatenate((valid, fake), axis=0)

                #all_imgs, all_labels = shuffle(all_imgs, all_labels)

                #imgs = all_imgs[:self.batch_size]
                #valid = all_labels[:self.batch_size]
                #gen_imgs = all_imgs[self.batch_size:]
                #fake = all_labels[self.batch_size:]
                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                i -= 1

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

            if epoch % 100 == 0:
                self.discriminator.save("FACT165_{}_discriminator_B{}_Pix{}_Latent{}_D{}_UP{}.h5".format(ran_num, self.batch_size, self.img_rows, self.latent_dim, self.dense, self.num_upscales))
                self.generator.save("FACT165_{}_generator_B{}_Pix{}_Latent{}_D{}_UP{}.h5".format(ran_num, self.batch_size, self.img_rows, self.latent_dim, self.dense, self.num_upscales))
                self.combined.save("FACT165_{}_combined_B{}_Pix{}_Latent{}_D{}_UP{}.h5".format(ran_num, self.batch_size, self.img_rows, self.latent_dim, self.dense, self.num_upscales))

        for p in processes:
            p.join()
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape((self.img_rows,self.img_rows)), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/small1615_fact_%d.png" % epoch, dpi=300)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN(width=32, batch_size=64, height=32, latent=100, dense=1024, num_upscales=2, directory="four/")
    dcgan.train(epochs=50000, save_interval=20)
