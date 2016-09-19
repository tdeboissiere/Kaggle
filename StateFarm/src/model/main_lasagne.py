
import numpy as np
import pickle
import h5py

import sys
sys.path.append("../utils")
import general_utils
import batch_utils

import lasagne as lasagne
import theano.tensor as T
import theano
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

from keras.utils import generic_utils, np_utils

from dotenv import load_dotenv, find_dotenv

import os
import time
import glob
# import sys
import pandas as pd


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
        names[0],
        ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                  flip_filters=False, nonlinearity=None) if use_bias
        else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                       flip_filters=False, nonlinearity=None)
    ))

    net.append((
        names[1],
        BatchNormLayer(net[-1][1])
    ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern)),
        int(lasagne.layers.get_output_shape(incoming_layer)[1] * ratio_n_filter), 1, int(1.0 / ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1] * upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern)),
            int(lasagne.layers.get_output_shape(incoming_layer)[1] * 4 * ratio_n_filter), 1, int(1.0 / ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_model(input_var, usage="train", pretr_weights_file=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 224, 224), input_var=input_var)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)

    if usage == "train":

        net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
        # Pretrained weights
        with open("../../data/external/resnet50.pkl", "rb") as f:
            net_train = pickle.load(f)
        lasagne.layers.set_all_param_values(net["prob"], net_train['values'])

        # Rewire last layers
        net.pop("fc1000", None)
        net.pop("prob", None)
        try:
            net["fc1000"]
        except:
            print("Ok rewired")
        net['fc10'] = DenseLayer(net['pool5'], num_units=10, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc10'], nonlinearity=softmax)

    elif usage == "inference":
        net['fc10'] = DenseLayer(net['pool5'], num_units=10, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc10'], nonlinearity=softmax)

        # Pretrained weights
        with open(pretr_weights_file, "rb") as f:
            net_train = pickle.load(f)
        lasagne.layers.set_all_param_values(net["prob"], net_train['values'])

    return net["prob"]


def cross_validate_inmemory(**kwargs):
    """
    StateFarm competition:
    Training set has 26 unique drivers. We do 26 fold CV where
    a driver is alternatively singled out to be the validation set

    Load the whole train data in memory for faster operations

    args: model (keras model)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    nb_classes = kwargs["nb_classes"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    prob = kwargs["prob"]
    data_file = kwargs["data_file"]
    semi_super_file = kwargs["semi_super_file"]
    list_folds = kwargs["list_folds"]
    weak_labels = kwargs["weak_labels"]
    experiment = kwargs["experiment"]

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    mean_values = np.load("../../data/external/resnet_mean_values.npy")

    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, "ResNet")
    # Create if it does not exist
    general_utils.create_dir(model_dir)
    # Automatically determine experiment name
    list_exp = glob.glob(model_dir + "/*")
    # Create the experiment dir and weights dir
    if experiment:
        exp_dir = exp_dir = os.path.join(model_dir, experiment)
    else:
        exp_dir = os.path.join(model_dir, "Experiment_%s" % len(list_exp))
    general_utils.create_dir(exp_dir)

    # Batch generator
    DataAug = batch_utils.AugDataGenerator(data_file,
                                           batch_size=batch_size,
                                           prob=prob,
                                           dset="train",
                                           maxproc=4,
                                           num_cached=60,
                                           random_augm=False,
                                           hdf5_file_semi=semi_super_file)
    DataAug.add_transform("h_flip")
    DataAug.add_transform("random_rot", angle=40)
    DataAug.add_transform("random_tr", tr_x=40, tr_y=40)
    DataAug.add_transform("random_blur", kernel_size=5)
    DataAug.add_transform("random_crop", min_crop_size=140, max_crop_size=160)

    epoch_size = n_batch_per_epoch * batch_size

    general_utils.pretty_print("Load all data...")

    with h5py.File(data_file, "r") as hf:
        X = hf["train_data"][:, :, :, :]
        y = hf["train_label"][:].astype(np.int32)

        try:
            for fold in list_folds:

                min_valid_loss = 100

                # Save losses
                list_train_loss = []
                list_valid_loss = []

                # Load valid data in memory for fast error evaluation
                idx_valid = hf["valid_fold%s" % fold][:]
                idx_train = hf["train_fold%s" % fold][:]
                X_valid = X[idx_valid]
                y_valid = y[idx_valid]

                # Normalise
                X_valid = X_valid[:, ::-1, :, :]
                X_valid = (X_valid - mean_values).astype(np.float32)

                # Define model
                input_var = T.tensor4('inputs')
                target_var = T.matrix('targets')

                network = build_model(input_var, usage="train")

                prediction = lasagne.layers.get_output(network)
                loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
                loss = loss.mean()

                params = lasagne.layers.get_all_params(network, trainable=True)

                updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=5E-4, momentum=0.9)
                train_fn = theano.function([input_var, target_var], loss, updates=updates)

                test_prediction = lasagne.layers.get_output(network, deterministic=True)
                test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
                test_loss = test_loss.mean()

                val_fn = theano.function([input_var, target_var], test_loss)

                # Loop over epochs
                for e in range(nb_epoch):
                    # Initialize progbar and batch counter
                    progbar = generic_utils.Progbar(epoch_size)
                    batch_counter = 1
                    l_train_loss = []
                    l_valid_loss = []
                    start = time.time()

                    for X_train, y_train in DataAug.gen_batch_inmemory(X, y, idx_train=idx_train):
                        if True:
                            general_utils.plot_batch(X_train, np.argmax(y_train, 1), batch_size)

                        # Normalise
                        X_train - X_train[:, ::-1, :, :]
                        X_train = (X_train - mean_values).astype(np.float32)
                        # Train
                        train_loss = train_fn(X_train, y_train.astype(np.float32))

                        l_train_loss.append(train_loss)
                        batch_counter += 1
                        progbar.add(batch_size, values=[("train loss", train_loss)])
                        if batch_counter >= n_batch_per_epoch:
                            break
                    print("")
                    print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

                    # Split image list into num_chunks chunks
                    chunk_size = batch_size
                    num_imgs = X_valid.shape[0]
                    num_chunks = num_imgs / chunk_size
                    list_chunks = np.array_split(np.arange(num_imgs), num_chunks)

                    # Loop over chunks
                    for chunk_idx in list_chunks:
                        X_b, y_b = X_valid[chunk_idx].astype(np.float32), y_valid[chunk_idx]
                        y_b = np_utils.to_categorical(y_b, nb_classes=nb_classes).astype(np.float32)
                        valid_loss = val_fn(X_b, y_b)
                        l_valid_loss.append(valid_loss)

                    train_loss = float(np.mean(l_train_loss))  # use float to make it json saveable
                    valid_loss = float(np.mean(l_valid_loss))  # use float to make it json saveable
                    print("Train loss:", train_loss, "valid loss:", valid_loss)
                    list_train_loss.append(train_loss)
                    list_valid_loss.append(valid_loss)

                    # Record experimental data in a dict
                    d_log = {}
                    d_log["fold"] = fold
                    d_log["nb_classes"] = nb_classes
                    d_log["batch_size"] = batch_size
                    d_log["n_batch_per_epoch"] = n_batch_per_epoch
                    d_log["nb_epoch"] = nb_epoch
                    d_log["epoch_size"] = epoch_size
                    d_log["prob"] = prob
                    d_log["augmentator_config"] = DataAug.get_config()
                    d_log["train_loss"] = list_train_loss
                    d_log["valid_loss"] = list_valid_loss

                    json_file = os.path.join(exp_dir, 'experiment_log_fold%s.json' % fold)
                    general_utils.save_exp_log(json_file, d_log)

                    # Only save the best epoch
                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        trained_weights_path = os.path.join(exp_dir, 'resnet_weights_fold%s.pickle' % fold)
                        model = {
                            'values': lasagne.layers.get_all_param_values(network),
                            'mean_image': mean_values
                        }
                        pickle.dump(model, open(trained_weights_path, 'wb'), protocol=-1)

        except KeyboardInterrupt:
            pass


def make_submission_CV(**kwargs):
    """
    StateFarm competition:
    Training a simple model on the whole training set. Save it for future use

    args: model (keras model)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    data_file = kwargs["data_file"]
    tr_weights_path = kwargs["tr_weights_path"]
    list_folds = kwargs["list_folds"]

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Compile model.
    for fold in list_folds:
        pretr_weights_file = os.path.join(tr_weights_path, 'resnet_weights_fold%s.pickle' % fold)
        assert os.path.isfile(pretr_weights_file)

        # Get mean values from the pretr_weights_file
        with open(pretr_weights_file, "rb") as f:
            net_train = pickle.load(f)
        mean_values = net_train["mean_image"]

        # Define model
        input_var = T.tensor4('inputs')

        network = build_model(input_var, usage="inference", pretr_weights_file=pretr_weights_file)

        prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_function = theano.function([input_var], prediction)

        with h5py.File(data_file, "r") as hf:
            X_test = hf["test_data"]
            id_test = hf["test_id"][:]
            y_test_pred = np.zeros((X_test.shape[0], 10))

            # Split image list into num_chunks chunks
            chunk_size = 32
            num_imgs = X_test.shape[0]
            num_chunks = num_imgs / chunk_size
            list_chunks = np.array_split(np.arange(num_imgs), num_chunks)

            for index, chunk_idx in enumerate(list_chunks):

                sys.stdout.write("\rFold %s Processing img %s/%s" %
                                 (fold, index + 1, X_test.shape[0]))
                sys.stdout.flush()

                X_test_batch = X_test[chunk_idx.tolist()]

                # To GBR and normalize
                X_test_batch = X_test_batch[:, ::-1, :, :]
                X_test_batch = (X_test_batch - mean_values)

                y_test_pred[chunk_idx] = predict_function(X_test_batch.astype(np.float32))

            print("")
            # Create a list of columns
            list_col = ["img", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
            # Get id_test
            id_test_fold = id_test.copy().reshape(-1, 1)
            # Save predictions of the fold
            subm_data = np.hstack([id_test_fold, y_test_pred])
            df_subm = pd.DataFrame(subm_data, columns=list_col)
            # Get directory corresponding to the weights
            subm_dir = os.path.basename(tr_weights_path)
            subm_name = 'resnet_fold%s.csv' % fold
            subm_file = os.path.join(subm_dir, subm_name)
            df_subm.to_csv(subm_file, index=False)


if __name__ == "__main__":

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))
    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))
    external_dir = os.path.expanduser(os.environ.get("EXTERNAL_DIR"))

    data_file = "fullframe_VGG_train.h5"
    data_file = os.path.join(data_dir, data_file)

    semi_super_file = "fullframe_VGG_test.h5"
    semi_super_file = os.path.join(data_dir, semi_super_file)
    # semi_super_file = None

    experiment = "Experiment_0"

    tr_weights_file = ""
    tr_weights_file = os.path.join(external_dir, tr_weights_file)

    # Set default params
    d_params = {"data_file": data_file,
                "semi_super_file": semi_super_file,
                "tr_weights_file": tr_weights_file,
                "objective": "categorical_crossentropy",
                "weak_labels": True,
                "img_dim": (3, 224, 224),
                "nb_classes": 10,
                "batch_size": 32,
                "n_batch_per_epoch": 100,
                "nb_epoch": 20,
                "prob": 0.9,
                "list_folds": range(8),
                "experiment": experiment
                }

    # Launch training
    cross_validate_inmemory(**d_params)

    # Launch submission
    data_file = "fullframe_VGG_test.h5"
    data_file = os.path.join(data_dir, data_file)

    tr_weights_path = "../../models/ResNet/%s" % experiment

    # Set default params
    d_params = {"data_file": data_file,
                "tr_weights_path": tr_weights_path,
                "list_folds": range(8)
                }

    # Launch submission
    make_submission_CV(**d_params)
