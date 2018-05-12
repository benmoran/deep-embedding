import numpy as np

import time

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Conv2D, Conv2DTranspose, Add
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras import callbacks
from keras.objectives import binary_crossentropy
from concrete_util import (kl_normal, kl_discrete, sampling_normal,
                  sampling_concrete, plot_digit_grid, EPSILON)


class SSVAE():
    """
    Class to handle building and training VAE models.
    """
    def __init__(self, input_shape=(121, 145, 3), extra_shape=4, latent_cont_dim=2,
                 latent_disc_dim=2, hidden_dim=128, filters=(64, 64, 64)):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        extra_shape : Number of extra observed covariates to pass to encoder/decoder

        latent_cont_dim : int
            Dimension of latent distribution.

        latent_disc_dim : int
            Dimension of discrete latent distribution.

        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        """
        self.opt = None
        self.model = None
        self.input_shape = input_shape
        self.extra_shape = extra_shape
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
        self.hidden_dim = hidden_dim
        self.filters = filters

    def fit(self, x_train, w_train, y_train, num_epochs=1, batch_size=100, val_split=.1,
            learning_rate=1e-3, reset_model=True):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        if x_train.shape[0] % batch_size != 0:
            raise(RuntimeError("Training data shape {} is not divisible by batch size {}".format(x_train.shape[0], self.batch_size)))

        # Update parameters
        K.set_value(self.opt.lr, learning_rate)
        self.model.compile(optimizer=self.opt,
                           loss={'generated': self._recon_loss,
                                 'alpha': self._kl_disc_loss,
                                 'z_enc': self._kl_normal_loss,
                                 'probs': self._discrim_loss,},
                           loss_weights={'generated':1.0,
                                         'alpha':1.0, # discrete KL
                                         'z_enc': 1.0, # continuous KL
                                         'probs': 10.0, # discrim loss
                                         },
                           metrics={'probs': 'categorical_accuracy'}) # TODO more metrics

        y_1hot = np.hstack([1-y_train, y_train]).astype('f')

        tb_callback = callbacks.TensorBoard(log_dir='./logs/{}'.format(time.time()),
                                            histogram_freq=1,
                                            batch_size=self.batch_size,
                                            write_graph=False,
                                            write_grads=False,
                                            write_images=False,
                                            embeddings_freq=0,
                                            embeddings_layer_names=None,
                                            embeddings_metadata=None)
        self.model.fit([x_train, w_train],
                       y={'generated': x_train,
                                'alpha': y_train, # not used
                                'z_enc': y_train, # not used
                                'probs': y_1hot},
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       validation_split=val_split,
                       callbacks=[tb_callback],
                       )

    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        image_in = Input(batch_shape=(self.batch_size,) + self.input_shape)
        extra_in = Input(batch_shape=(self.batch_size, self.extra_shape))
        inputs = [image_in, extra_in]

        # Instantiate encoder layers
        Q_0 = Conv2D(self.input_shape[2], (3, 3), padding='same',
                     activation='relu')
        Q_1 = Conv2D(self.filters[0], (3, 3), padding='same', strides=(2, 2),
                     activation='relu')
        Q_2 = Conv2D(self.filters[1], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_3 = Conv2D(self.filters[2], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_6 = Dense(self.hidden_dim, activation='relu')


        Q_z_mean = Dense(self.latent_cont_dim, name='z_mean')
        Q_z_log_var = Dense(self.latent_cont_dim, name='z_log_var')

        # Set up encoder
        x = Q_0(image_in)
        x = Q_1(x)
        x = Q_2(x)
        x = Q_3(x)
        flat = Q_4(x)
        extra_hidden = Q_5(extra_in)
        merged = Concatenate()([flat, extra_hidden])
        hidden = Q_6(merged)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        # Parameters for concrete latent distribution
        if self.latent_disc_dim:
            Q_c = Dense(self.latent_disc_dim, activation='softmax', name='alpha')
            alpha = Q_c(hidden)

        # Sample from latent distributions
        if self.latent_disc_dim:
            z = Lambda(self._sampling_normal)([z_mean, z_log_var])
            c = Lambda(self._sampling_concrete)(alpha)
            encoding = Concatenate()([z, c])
        else:
            encoding = Lambda(self._sampling_normal)([z_mean, z_log_var])

        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (self.input_shape[0] // 2, self.input_shape[1] // 2, self.filters[2])
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_1 = Dense(np.prod(out_shape), activation='relu')
        G_2 = Reshape(out_shape)
        G_3 = Conv2DTranspose(self.filters[2], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_4 = Conv2DTranspose(self.filters[1], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_5 = Conv2DTranspose(self.filters[0], (3, 3), padding='valid',
                              strides=(2, 2), activation='relu')
        G_6 = Conv2D(self.input_shape[2], (3, 3), padding='same',
                     strides=(1, 1), activation='sigmoid', name='generated')

        # Apply generator layers
        concat = Concatenate()([encoding, extra_in])
        x = G_0(concat)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        generated = G_6(x)

        z_enc = Concatenate(name='z_enc')([z_mean, z_log_var])
        probs = Lambda(lambda x: x+0, name='probs')(alpha) # This is just a copy for now so we can use two losses
        self.model = Model(inputs=inputs, outputs=[generated, alpha, probs, z_enc])


        # Set up generator
        latent_in = Input(batch_shape=(self.batch_size, self.latent_dim))
        inputs_G = [latent_in, extra_in]
        # We concatenate the observed extra covariates to the latent dim for the encoder.
        concat = Concatenate()([latent_in, extra_in])
        x = G_0(concat)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        if self.latent_disc_dim:
            self.alpha = alpha

        # Compile models
        self.opt = RMSprop()
        # TODO: why compiling here as well as in fit?
        self.model.compile(optimizer=self.opt,
                           loss={'generated': self._recon_loss,
                                 'alpha': self._kl_disc_loss,
                                 'z_enc': self._kl_normal_loss,
                                 'probs': self._discrim_loss,},
                           metrics={'probs': 'categorical_accuracy'}) # TODO more metrics

        #self.model.compile(optimizer=self.opt, loss='mse')#self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')

        print("Completed model setup.")

    def generate(self, latent_sample, extra):
        """
        Generating examples from samples from the latent distribution.
        """
        # Model requires batch_size batches, so tile if this is not the case
        if latent_sample.shape[0] != self.batch_size:
            latent_sample = np.tile(latent_sample, self.batch_size).reshape(
                              (self.batch_size, self.latent_dim))
            extra = np.tile(extra, self.batch_size).reshape(
                              (self.batch_size, self.extra_shape))
        return self.generator.predict([latent_sample, extra], batch_size=self.batch_size)

    # def plot(self, std_dev=2.):
    #     """
    #     Method to plot generated digits on a grid.
    #     """
    #     return plot_digit_grid(self, std_dev=std_dev)

    def _recon_loss(self, x, x_generated):
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        return reconstruction_loss

    def _kl_normal_loss(self, _prior, z_enc):
        # Ignores prior and uses N(0,1)
        z_mean = z_enc[:,:self.latent_dim]
        z_log_var = z_enc[:,self.latent_dim:]
        kl_normal_loss = kl_normal(z_mean, z_log_var)
        return kl_normal_loss

    def _kl_disc_loss(self, _prior, alpha):
        # Ignores prior and uses Uniform
        if self.latent_disc_dim:
            return kl_discrete(alpha)
        return 0.0

    def _discrim_loss(self, labels, probs):
        return K.categorical_crossentropy(labels, probs)

    # def _vae_loss(self, x, x_generated):
    #     """
    #     Variational Auto Encoder loss.
    #     """
    #     x = K.flatten(x)
    #     x_generated = K.flatten(x_generated)
    #     reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
    #                               binary_crossentropy(x, x_generated)
    #     kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
    #     if self.latent_disc_dim:
    #         kl_disc_loss = kl_discrete(self.alpha)
    #     else:
    #         kl_disc_loss = 0
    #     # TODO Preserve references to individual loss components
    #     # TODO Add semi-sup loss

    #     return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (self.batch_size, self.latent_cont_dim))

    def _sampling_concrete(self, args):
        """
        Sampling from a concrete distribution
        """
        return sampling_concrete(args, (self.batch_size, self.latent_disc_dim))
