import sys
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
try:
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import cm
except:
    pass
import json
import glob


def pretty_print(string):
    """
    Simple utility to highlight printing

    args: string (str) string to print
    """

    print("")
    print(string)
    print("")


def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def plot_batch(X, y, batch_size):
    """
    Plot the images in X, add a label (in y)

    details: build a gridspec of the size of the batch
             (valid batch_sizes are multiple of 2 from 8 to 64)
    """

    d_class = {0: "safe driving",
               1: "texting - right",
               2: "talking on the phone - right",
               3: "texting - left",
               4: "talking on the phone - left",
               5: "operating the radio",
               6: "drinking",
               7: "reaching behind",
               8: "hair and makeup",
               9: "talking to passenger"}

    assert X.shape[0] >= batch_size, "batch size greater than X.shape[0]"

    if batch_size == 8:
        gs = gridspec.GridSpec(2, 4)
    elif batch_size == 16:
        gs = gridspec.GridSpec(4, 4)
    elif batch_size == 32:
        gs = gridspec.GridSpec(4, 8)
    elif batch_size == 64:
        gs = gridspec.GridSpec(8, 8)
    else:
        print("Batch too big")
        return
    fig = plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        ax = plt.subplot(gs[i])
        img = X[i, :, :, :]
        img_shape = img.shape
        min_s = min(img_shape)
        if img_shape.index(min_s) == 0:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_xlabel(d_class[int(y[i])], fontsize=8)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    gs.tight_layout(fig)
    plt.show()
    raw_input()
    plt.clf()
    plt.close()


def save_exp_log(file_name, d_log):
    """
    Utility to save the experiment log as json file

    details: Save the experiment log (a dict d_log) in file_name

    args: file_name (str) the file in which to save the log
          d_log (dict) python dict holding experiment log
    """

    with open(file_name, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)
