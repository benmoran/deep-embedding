import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

import pandas as pd

from keras import backend as K
from keras.datasets import mnist
from plotly import tools

EPSILON = 1e-8

import h5py

def save_large_dataset(file_name, variable):
    h5f = h5py.File(file_name + '.h5', 'w')
    h5f.create_dataset('variable', data=variable, compression='lzf')
    h5f.close()

def load_large_dataset(file_name):
    h5f = h5py.File(file_name + '.h5','r')
    variable = h5f['variable'][:]
    h5f.close()
    return variable


def get_processed_covariates(filename):
    df = pd.read_hdf(filename)
    # pd.read_hdf("/data/code/deep-embedding/concrete/covariates.h5")
    # PAC_ID  Label   Age  Gender          TIV  Site
    
    mu, std = df.Age.mean(), df.Age.std()
    df['age'] = (df.Age - mu)/std
    
    mu, std = df.TIV.mean(), df.TIV.std()
    df['tiv'] = (df.TIV - mu)/std
    for level in sorted(df.Site.unique()):
        df['site{}'.format(level)] = (df.Site == level)
    for level in sorted(df.Gender.unique()):
        df['gender{}'.format(level)] = (df.Gender == level)
    return df.loc[:, 'age':].astype('f').values
    

def get_processed_mnist():
    """
    Get normalized MNIST datasets with correct shapes (Tensorflow style).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    return (x_train, y_train), (x_test, y_test)


def get_processed_pac(batch_size=100, split_frac=0.9):
    """
    Load Photon PAC data
    """
    # imageslices.h5 is X[:,:,:,55:58,0] from Pawel's images.h5
    X = load_large_dataset('imageslices')
    Y = load_large_dataset('labels')
    W = get_processed_covariates('covariates.h5')
    np.random.seed(9999) #seed fixed for reproducibility
    mask = np.random.choice(len(X),
                            batch_size *
                            int((len(X) * split_frac)
                                // batch_size), False)
    #mask = np.random.rand(len(X)) < 0.9  #array of boolean variables

    training_set = X[mask]
    training_labels = Y[mask]
    training_extra = W[mask]    

    validation_set = X[~mask]
    validation_labels = Y[~mask]
    validation_extra = W[mask]
    
    return (training_set, training_labels, training_extra), \
        (validation_set, validation_labels, validation_extra)

def get_one_hot_vector(idx, dim=10):
    """
    Returns a 1-hot vector of dimension dim with the 1 at index idx

    Parameters
    ----------
    idx : int
        Index where one hot vector is 1

    dim : int
        Dimension of one hot vector
    """
    one_hot = np.zeros(dim)
    one_hot[idx] = 1.
    return one_hot


def plot_digit_grid(model, fig_size=10, digit_size=28, std_dev=2.,
                    filename='vae'):
    """
    Plot a grid of generated digits. Each column corresponds to a different
    setting of the discrete variable, each row to a random setting of the other
    latent variables.

    Parameters
    ----------
    model : VAE model

    fig_size : int

    digit_size : int

    std_dev : float

    filename : string
    """
    figure = np.zeros((digit_size * fig_size, digit_size * fig_size))
    grid_x = np.linspace(-std_dev, std_dev, fig_size)
    grid_y = np.linspace(-std_dev, std_dev, fig_size)

    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            # Sample from latent distribution
            if model.latent_disc_dim:
                z_sample = std_dev * np.random.rand(model.latent_cont_dim)
                c_sample = get_one_hot_vector(j % model.latent_disc_dim, model.latent_disc_dim)
                latent_sample = np.hstack((z_sample, c_sample))
            else:
                latent_sample = std_dev * np.random.rand(model.latent_dim)
                latent_sample[0] = xi
                latent_sample[1] = yi
            # Generate a digit and plot it
            generated = model.generate(latent_sample)
            digit = generated[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    trace = go.Heatmap(
                x=grid_x,
                y=grid_y,
                z=figure,
                colorscale='Viridis'
            )

    layout = go.Layout(
        yaxis=dict(
            autorange='reversed'
        )
    )

    fig = go.Figure(data=[trace], layout=layout)

    py.plot(fig, filename=get_timestamp_filename(filename), auto_open=False)


def get_timestamp_filename(filename):
    """
    Returns a string of the form "filename_<date>.html"
    """
    date = time.strftime("%H-%M_%d-%m-%Y")
    return filename + "_" + date + ".html"


def kl_normal(z_mean, z_log_var):
    """
    KL divergence between N(0,1) and N(z_mean, exp(z_log_var)) where covariance
    matrix is diagonal.

    Parameters
    ----------
    z_mean : Tensor

    z_log_var : Tensor

    dim : int
        Dimension of tensor
    """
    # Sum over columns, so this now has size (batch_size,)
    kl_per_example = .5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
    return K.mean(kl_per_example)


def kl_discrete(dist):
    """
    KL divergence between a uniform distribution over num_cat categories and
    dist.

    Parameters
    ----------
    dist : Tensor - shape (None, num_categories)

    num_cat : int
    """
    num_categories = tuple(dist.get_shape().as_list())[1]
    dist_sum = K.sum(dist, axis=1)  # Sum over columns, this now has size (batch_size,)
    dist_neg_entropy = K.sum(dist * K.log(dist + EPSILON), axis=1)
    return np.log(num_categories) + K.mean(dist_neg_entropy - dist_sum)


def sampling_concrete(alpha, out_shape, temperature=0.67):
    """
    Sample from a concrete distribution with parameters alpha.

    Parameters
    ----------
    alpha : Tensor
        Parameters
    """
    uniform = K.random_uniform(shape=out_shape)
    gumbel = - K.log(- K.log(uniform + EPSILON) + EPSILON)
    logit = (K.log(alpha + EPSILON) + gumbel) / temperature
    return K.softmax(logit)


def sampling_normal(z_mean, z_log_var, out_shape):
    """
    Sampling from a normal distribution with mean z_mean and variance z_log_var
    """
    epsilon = K.random_normal(shape=out_shape, mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
