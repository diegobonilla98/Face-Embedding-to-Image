from increase_memory_alloc import K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Add, Reshape, GlobalAveragePooling2D, PReLU, Flatten, LeakyReLU, ReLU, BatchNormalization, Dropout, Input, Activation, GaussianNoise, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
from SubpixelConv2d import SubpixelConv2D

import cv2
import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader


class FaceGenerator:
    def __init__(self, is_test=False):
        self.batch_size = 32
        self.image_shape = (128, 128, 3)
        self.embedding_shape = (2048, )
        self.data_loader = DataLoader(self.image_shape, self.batch_size)

        self.embedding_dim = 2048
        self.redux_emb = 512

        self.gen_lr = 0.0002
        self.dis_lr = 0.0002

        self.init = RandomNormal(stddev=0.02)

        self.gen_opt = Adam(learning_rate=self.gen_lr, beta_1=0.5)
        self.dis_opt = Adam(learning_rate=self.dis_lr, beta_1=0.5)

        if is_test:
            self.generator = self.create_generator()
            return

        self.test_num = 3
        self.sample_X, self.sample_y = self.data_loader.load_batch(self.test_num)

        self.generator = self.create_generator()
        self.generator.compile(self.gen_opt, loss='mae')  # , loss_weights=[10.])
        self.generator.summary()
        plot_model(self.generator, 'generator.png', show_shapes=True)

        self.discriminator = self.create_discriminator()
        self.set_trainable(self.discriminator, True)
        self.set_trainable(self.generator, False)
        self.discriminator.compile(optimizer=self.dis_opt, loss='binary_crossentropy', metrics=['acc'], loss_weights=[0.5])
        self.discriminator.summary()
        plot_model(self.discriminator, 'discriminator.png', show_shapes=True)

        self.set_trainable(self.discriminator, False)
        self.set_trainable(self.generator, True)

        input_embedding = Input(shape=(self.embedding_dim,))

        generated_bis = self.generator(input_embedding)
        dis = self.discriminator([generated_bis, input_embedding])
        self.adversarial = Model(input_embedding, [generated_bis, dis], name='Adversarial')
        self.adversarial.compile(optimizer=self.gen_opt, loss=['mae', 'binary_crossentropy'], loss_weights=[100., 1.])
        self.adversarial.summary()
        plot_model(self.adversarial, 'adversarial.png', show_shapes=True)

    @staticmethod
    def set_trainable(model, state):
        model.trainable = state
        for layer in model.layers:
            layer.trainable = state

    def create_generator(self):
        embedding_input = Input(shape=(self.embedding_dim,))

        embedding_x = Dense(self.redux_emb, kernel_initializer=self.init)(embedding_input)
        embedding_x = LeakyReLU(alpha=0.2)(embedding_x)

        def deconvolutional(num_filters, stride, padding, kernel_size, input_tensor):
            u = Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding, strides=1,
                       kernel_initializer=self.init)(input_tensor)
            u = LeakyReLU(alpha=0.2)(u)
            u = BatchNormalization(momentum=0.9)(u)
            u = UpSampling2D()(u)
            u = Dropout(0.3)(u)
            return u

        reshaped = (8, 8, 512)
        x = Dense(np.prod(reshaped), kernel_initializer=self.init)(embedding_x)
        x = Reshape(reshaped)(x)

        x = deconvolutional(num_filters=512, stride=2, padding='same', kernel_size=4, input_tensor=x)
        x = deconvolutional(num_filters=256, stride=2, padding='same', kernel_size=4, input_tensor=x)
        x = deconvolutional(num_filters=128, stride=2, padding='same', kernel_size=4, input_tensor=x)
        x = deconvolutional(num_filters=64, stride=2, padding='same', kernel_size=4, input_tensor=x)

        output = Conv2D(filters=3, strides=1, padding='same', kernel_size=7, kernel_initializer=self.init)(x)
        output = Activation('tanh')(output)

        return Model(embedding_input, output, name='Generator')

    def create_discriminator(self):
        image_input = Input(shape=self.image_shape)
        embedding_input = Input(shape=(self.embedding_dim,))

        def convolutional(num_filters, stride, padding, kernel_size, input_tensor, use_bias=False):
            u = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=stride, padding=padding,
                       kernel_initializer=self.init, use_bias=use_bias)(input_tensor)
            u = BatchNormalization(momentum=0.9)(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u

        x = GaussianNoise(0.1)(image_input)
        x = convolutional(num_filters=64, stride=2, padding='valid', kernel_size=4, input_tensor=x)
        x = convolutional(num_filters=128, stride=2, padding='valid', kernel_size=4, input_tensor=x)
        x = convolutional(num_filters=256, stride=2, padding='valid', kernel_size=4, input_tensor=x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.embedding_dim)(x)
        x = Concatenate()([x, embedding_input])
        x = ReLU()(x)
        x = Dense(self.embedding_dim * 2, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        return Model([image_input, embedding_input], output, name='Discriminator')

    def plot_results(self, epoch):
        fake = self.generator.predict(self.sample_X)
        res = (np.vstack([np.hstack([fake[0, :, :, :], self.sample_y[0, :, :, :]]),
                          np.hstack([fake[1, :, :, :], self.sample_y[1, :, :, :]]),
                          np.hstack([fake[2, :, :, :], self.sample_y[2, :, :, :]])]) + 1) / 2
        plt.clf()
        plt.imshow(res)
        plt.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def train(self, num_epochs):
        assert self.batch_size % 2 == 0
        dis_losses = []
        dis_t_losses = []
        dis_f_losses = []
        gen_losses = []
        what_loss = 0
        for epoch in range(num_epochs):
            X, y = self.data_loader.load_batch(self.batch_size)

            true = np.ones((X.shape[0], ), 'float32')
            fake = np.zeros((X.shape[0], ), 'float32')
            true = true * 0.7 + np.random.random((X.shape[0], )) * 0.5
            fake = fake + np.random.random((X.shape[0], )) * 0.3

            self.set_trainable(self.discriminator, True)
            self.set_trainable(self.generator, False)

            if np.random.rand() < 0.05:
                true, fake = fake, true
            true_d_loss = self.discriminator.train_on_batch([y, X], true)
            gen = self.generator.predict(X)
            if what_loss == 0:
                fake_d_loss = self.discriminator.train_on_batch([y, X[::-1, :]], fake)
            elif what_loss == 1:
                fake_d_loss = self.discriminator.train_on_batch([gen, X[::-1, :]], fake)
            else:
                what_loss = -1
                fake_d_loss = self.discriminator.train_on_batch([gen, X], fake)
            d_loss = (fake_d_loss[0] + true_d_loss[0]) / 2
            what_loss += 1

            true = np.ones((X.shape[0],), 'float32')

            self.set_trainable(self.discriminator, False)
            self.set_trainable(self.generator, True)
            gen_loss = sum(self.adversarial.train_on_batch(X, [y, true]))

            print(f"Epoch {epoch}/{num_epochs}:\t[Adv_loss: {gen_loss}]\t[D_loss: {d_loss}, true: {true_d_loss}, fake: {fake_d_loss}]")
            gen_losses.append(gen_loss)
            dis_losses.append(d_loss)
            dis_t_losses.append(true_d_loss[0])
            dis_f_losses.append(fake_d_loss[0])

            if epoch % 10 == 0:
                self.plot_results(epoch)
            if epoch % 100 == 0:
                self.generator.save_weights(f'./RESULTS/weights/gen_epoch_{epoch}.h5')
            if epoch % 2 == 0:
                plt.clf()
                plt.plot(gen_losses, label="Gen Loss", alpha=0.8)
                plt.plot(dis_losses, label="Total Dis Loss", alpha=0.2)
                plt.plot(dis_t_losses, label="True Dis", alpha=0.2)
                plt.plot(dis_f_losses, label="Fake Dis", alpha=0.2)
                plt.legend()
                plt.savefig(f'./RESULTS/metrics.png')
                plt.close()


if __name__ == '__main__':
    gan = FaceGenerator()
    gan.train(30_000)
