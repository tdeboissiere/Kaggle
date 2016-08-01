import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import AveragePooling2D

import theano.tensor.nnet.abstract_conv as absconv

import h5py


def CNN(nb_classes, img_dim, pretr_weights_file=None, model_name=None):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, name="convolution2d_1", input_shape=(3, 224, 224), border_mode="same", activation='relu'))
    model.add(Convolution2D(32, 3, 3, name="convolution2d_2", border_mode="same", activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_1"))

    model.add(Convolution2D(64, 3, 3, name="convolution2d_3", border_mode="same", activation='relu'))
    model.add(Convolution2D(64, 3, 3, name="convolution2d_4", border_mode="same", activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_2"))

    model.add(Convolution2D(128, 3, 3, name="convolution2d_5", border_mode="same", activation='relu'))
    model.add(Convolution2D(128, 3, 3, name="convolution2d_6", border_mode="same", activation='relu'))
    model.add(Convolution2D(128, 3, 3, name="convolution2d_7", border_mode="same", activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name="maxpooling2d_3"))

    model.add(Flatten(name="flatten_1"))
    model.add(Dense(1024, activation='relu', name="dense_1"))
    model.add(Dropout(0.5, name="dropout_1"))
    model.add(Dense(1024, activation='relu', name="dense_2"))
    model.add(Dropout(0.5, name="dropout_2"))
    model.add(Dense(nb_classes, activation='softmax', name="dense_3"))

    if model_name:
        model.name = model_name
    else:
        model.name = "CNN"

    if pretr_weights_file:
        model.load_weights(pretr_weights_file)
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(nb_classes, activation='softmax', name="dense_4"))

    return model


def VGG(nb_classes, img_dim, pretr_weights_file=None, model_name=None):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=img_dim, name="zeropadding2d_1"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_1"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_2"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_1"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_3"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_3"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_4"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_4"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_2"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_5"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_5"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_6"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_6"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_7"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_7"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_3"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_8"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_8"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_9"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_9"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_10"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_10"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_4"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_11"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_11"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_12"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_12"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_13"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_13"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_5"))

    model.add(Flatten(name="flatten_1"))
    model.add(Dense(4096, activation='relu', name="dense_1"))
    model.add(Dropout(0.5, name="dropout_1"))
    model.add(Dense(4096, activation='relu', name="dense_2"))
    model.add(Dropout(0.5, name="dropout_2"))
    model.add(Dense(1000, activation='softmax', name="dense_3"))

    if model_name:
        model.name = model_name
    else:
        model.name = "VGG"

    if pretr_weights_file:
        model.load_weights(pretr_weights_file)
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(nb_classes, activation='softmax', name="dense_4"))

    # Freeze layers until specified number
    # for k in range(freeze_until):
    #     model.layers[k].trainable = True

    return model


def VGG19(nb_classes, img_dim, pretr_weights_file=None, model_name=None):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=img_dim, name="zeropadding2d_1"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_1"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_2"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_1"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_3"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_3"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_4"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_4"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_2"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_5"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_5"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_6"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_6"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_7"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_7"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_8"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_8"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_3"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_9"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_9"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_10"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_10"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_11"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_11"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_12"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_12"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_4"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_13"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_13"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_14"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_14"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_15"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_15"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_16"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_16"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_5"))

    model.add(Flatten(name="flatten_1"))
    model.add(Dense(4096, activation='relu', name="dense_1"))
    model.add(Dropout(0.5, name="dropout_1"))
    model.add(Dense(4096, activation='relu', name="dense_2"))
    model.add(Dropout(0.5, name="dropout_2"))
    model.add(Dense(1000, activation='softmax', name="dense_3"))

    if model_name:
        model.name = model_name
    else:
        model.name = "VGG19"

    if pretr_weights_file:
        model.load_weights(pretr_weights_file)
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(nb_classes, activation='softmax', name="dense_4"))

    # Freeze layers until specified number
    # for k in range(freeze_until):
    #     model.layers[k].trainable = True

    return model


def VGGCAM(nb_classes, img_dim, pretr_weights_file=None, model_name=None):
    """
    Build VGGCAM network

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=img_dim, name="zeropadding2d_1"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_1"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_2"))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="convolution2d_2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_1"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_3"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_3"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_4"))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="convolution2d_4"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_2"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_5"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_5"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_6"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_6"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_7"))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="convolution2d_7"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_3"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_8"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_8"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_9"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_9"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_10"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_10"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="maxpooling2d_4"))

    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_11"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_11"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_12"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_12"))
    model.add(ZeroPadding2D((1, 1), name="zeropadding2d_13"))
    model.add(Convolution2D(512, 3, 3, activation='relu', name="convolution2d_13"))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode="same", name="convolution2d_14"))
    model.add(AveragePooling2D((14, 14), name="average_pooling2d_1"))
    model.add(Flatten(name="flatten_1"))
    # Add the W layer
    model.add(Dense(10, activation='softmax', name="dense_1"))

    if model_name:
        model.name = model_name
    else:
        model.name = "VGGCAM"

    if pretr_weights_file:

        with h5py.File(pretr_weights_file) as hw:
            for k in range(hw.attrs['nb_layers']):
                g = hw['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
                if model.layers[k].name == "convolution2d_13":
                    break
    return model


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):

    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 224 * 224))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])


def load(model_name, nb_classes, img_dim, pretr_weights_file=None):

    if model_name == "VGG":
        model = VGG(nb_classes, img_dim, pretr_weights_file=pretr_weights_file, model_name=None)
    elif model_name == "VGG19":
        model = VGG19(nb_classes, img_dim, pretr_weights_file=pretr_weights_file, model_name=None)
    elif model_name == "VGGCAM":
        model = VGGCAM(nb_classes, img_dim, pretr_weights_file=pretr_weights_file, model_name=None)
    elif model_name == "CNN":
        model = CNN(nb_classes, img_dim, pretr_weights_file=pretr_weights_file, model_name=None)
    return model
