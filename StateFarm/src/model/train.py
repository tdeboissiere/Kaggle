from keras.layers.core import Dense
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import model_from_json
import models
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import Callback

from dotenv import load_dotenv, find_dotenv


from sklearn.metrics import log_loss

import os
import sys
import glob
import h5py
import numpy as np
import datetime
import pandas as pd
import time
# Utils
sys.path.append("../utils")
import batch_utils
import general_utils


def normalisation(X, normalisation_style):
    # Convert to GBR then normalise
    if normalisation_style == "VGG":
        X = X[:, ::-1, :, :].astype(np.float32)
        X = X.transpose(0, 2, 3, 1)
        mean_pixel = [103.939, 116.779, 123.68]
        for c in range(3):
            X[:, :, :, c] = X[:, :, :, c] - mean_pixel[c]
        X = X.transpose(0, 3, 1, 2)
    elif normalisation_style == "standard":
        X = X.astype(np.float32) / 255.

    return X


def cross_validate_inmemory(model_name, **kwargs):
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
    do_plot = kwargs["do_plot"]
    data_file = kwargs["data_file"]
    semi_super_file = kwargs["semi_super_file"]
    pretr_weights_file = kwargs["pretr_weights_file"]
    normalisation_style = kwargs["normalisation_style"]
    weak_labels = kwargs["weak_labels"]
    objective = kwargs["objective"]
    experiment = kwargs["experiment"]
    start_fold = kwargs["start_fold"]

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, model_name)
    # Create if it does not exist
    general_utils.create_dir(model_dir)
    # Automatically determine experiment name
    list_exp = glob.glob(model_dir + "/*")
    # Create the experiment dir and weights dir
    if experiment:
        exp_dir = os.path.join(model_dir, experiment)
    else:
        exp_dir = os.path.join(model_dir, "Experiment_%s" % len(list_exp))
    general_utils.create_dir(exp_dir)

    # Compile model.
    # opt = RMSprop(lr=5E-6, rho=0.9, epsilon=1e-06)
    opt = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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
    # DataAug.add_transform("v_flip")
    # DataAug.add_transform("fixed_rot", angle=40)
    DataAug.add_transform("random_rot", angle=40)
    # DataAug.add_transform("fixed_tr", tr_x=40, tr_y=40)
    DataAug.add_transform("random_tr", tr_x=40, tr_y=40)
    # DataAug.add_transform("fixed_blur", kernel_size=5)
    DataAug.add_transform("random_blur", kernel_size=5)
    # DataAug.add_transform("fixed_erode", kernel_size=4)
    DataAug.add_transform("random_erode", kernel_size=3)
    # DataAug.add_transform("fixed_dilate", kernel_size=4)
    DataAug.add_transform("random_dilate", kernel_size=3)
    # DataAug.add_transform("fixed_crop", pos_x=10, pos_y=10, crop_size_x=200, crop_size_y=200)
    DataAug.add_transform("random_crop", min_crop_size=140, max_crop_size=160)
    # DataAug.add_transform("hist_equal")
    # DataAug.add_transform("random_occlusion", occ_size_x=100, occ_size_y=100)

    epoch_size = n_batch_per_epoch * batch_size

    general_utils.pretty_print("Load all data...")

    with h5py.File(data_file, "r") as hf:
        X = hf["train_data"][:, :, :, :]
        y = hf["train_label"][:].astype(np.uint8)
        y = np_utils.to_categorical(y, nb_classes=nb_classes)  # Format for keras

        try:
            for fold in range(start_fold, 8):
                # for fold in np.random.permutation(26):

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
                X_valid = normalisation(X_valid, normalisation_style)

                # Compile model
                general_utils.pretty_print("Compiling...")
                model = models.load(model_name,
                                    nb_classes,
                                    X_valid.shape[-3:],
                                    pretr_weights_file=pretr_weights_file)
                model.compile(optimizer=opt, loss=objective)

                # Save architecture
                json_string = model.to_json()
                with open(os.path.join(data_dir, '%s_archi.json' % model.name), 'w') as f:
                    f.write(json_string)

                for e in range(nb_epoch):
                    # Initialize progbar and batch counter
                    progbar = generic_utils.Progbar(epoch_size)
                    batch_counter = 1
                    l_train_loss = []
                    start = time.time()

                    for X_train, y_train in DataAug.gen_batch_inmemory(X, y, idx_train=idx_train):
                        if do_plot:
                            general_utils.plot_batch(X_train, np.argmax(y_train, 1), batch_size)

                        # Normalise
                        X_train = normalisation(X_train, normalisation_style)

                        train_loss = model.train_on_batch(X_train, y_train)
                        l_train_loss.append(train_loss)
                        batch_counter += 1
                        progbar.add(batch_size, values=[("train loss", train_loss)])
                        if batch_counter >= n_batch_per_epoch:
                            break
                    print("")
                    print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
                    y_valid_pred = model.predict(X_valid, verbose=0, batch_size=16)
                    train_loss = float(np.mean(l_train_loss))  # use float to make it json saveable
                    valid_loss = log_loss(y_valid, y_valid_pred)
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
                    d_log["optimizer"] = opt.get_config()
                    d_log["augmentator_config"] = DataAug.get_config()
                    d_log["train_loss"] = list_train_loss
                    d_log["valid_loss"] = list_valid_loss

                    json_file = os.path.join(exp_dir, 'experiment_log_fold%s.json' % fold)
                    general_utils.save_exp_log(json_file, d_log)

                    # Only save the best epoch
                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        trained_weights_path = os.path.join(exp_dir, '%s_weights_fold%s.h5' % (model.name, fold))
                        model.save_weights(trained_weights_path, overwrite=True)

        except KeyboardInterrupt:
            pass


def make_submission_CV(model_name, **kwargs):
    """
    StateFarm competition:
    Training a simple model on the whole training set. Save it for future use

    args: model (keras model)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    data_file = kwargs["data_file"]
    tr_weights_path = kwargs["tr_weights_path"]
    model_archi_file = kwargs["model_archi_file"]
    normalisation_style = kwargs["normalisation_style"]
    start_fold = kwargs["start_fold"]

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Compile model.
    model = model_from_json(open(model_archi_file).read())
    model.name = model_name
    list_y_test_pred = []
    for fold in range(start_fold, 6):
        weights_file = os.path.join(tr_weights_path, "%s_weights_fold%s.h5" % (model.name, fold))
        model.load_weights(weights_file)
        model.compile(optimizer="sgd", loss='categorical_crossentropy')

        with h5py.File(data_file, "r") as hf:
            X_test = hf["test_data"]
            id_test = hf["test_id"][:]
            list_pred = []

            chunk_size = 10000
            num_imgs = X_test.shape[0]

            # Split image list into num_chunks chunks
            num_chunks = num_imgs / chunk_size
            list_chunks = np.array_split(np.arange(num_imgs), num_chunks)

            # Loop over chunks
            for index, chunk_idx in enumerate(list_chunks):
                sys.stdout.write("\rFold %s Processing chunk %s/%s" %
                                 (fold, index + 1, len(list_chunks)))
                sys.stdout.flush()

                X_test_batch = X_test[chunk_idx, :, :, :]
                X_test_batch = normalisation(X_test_batch, normalisation_style)

                y_test_pred = model.predict(X_test_batch, batch_size=16, verbose=0)
                list_pred.append(y_test_pred)
            print("")

            # Combine all chunks
            y_test_pred = np.vstack(list_pred)
            # Get id_test
            id_test_fold = id_test.copy().reshape(-1, 1)
            # Save predictions of the fold
            subm_data = np.hstack([id_test_fold, y_test_pred])
            df_subm = pd.DataFrame(subm_data, columns=["img", "c0", "c1", "c2", "c3",
                                                       "c4", "c5", "c6", "c7", "c8", "c9"])
            # Get directory corresponding to the weights
            subm_dir = os.path.dirname(tr_weights_path)
            subm_name = model.name + '_fold%s.csv' % fold
            subm_file = os.path.join(subm_dir, subm_name)
            df_subm.to_csv(subm_file, index=False)

        list_y_test_pred.append(y_test_pred)

    y_test_pred = np.mean(list_y_test_pred, 0)

    id_test = id_test.reshape(-1, 1)
    subm_data = np.hstack([id_test, y_test_pred])
    df_subm = pd.DataFrame(subm_data, columns=["img", "c0", "c1", "c2", "c3",
                                               "c4", "c5", "c6", "c7", "c8", "c9"])

    now = datetime.datetime.now()
    # Get directory corresponding to the weights
    subm_dir = os.path.dirname(tr_weights_path)
    subm_name = model.name + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + ".csv"
    subm_file = os.path.join(subm_dir, subm_name)
    df_subm.to_csv(subm_file, index=False)

