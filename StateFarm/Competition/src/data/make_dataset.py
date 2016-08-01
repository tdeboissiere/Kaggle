import os
import sys
import h5py
import cv2
import pandas as pd
import numpy as np
import parmap
import glob
from dotenv import find_dotenv, load_dotenv


def resize_VGG(img):
    """
    Resize img to a 224 x 224 image
    """

    return cv2.resize(img,
                      (224, 224),
                      interpolation=cv2.INTER_AREA)


def format_image(img_path, tf):
    # ::-1 for BGR to RGB
    img = cv2.imread(img_path)[:, :, ::-1]
    img = tf(img)
    # Go from height width chan to chan height width
    img = img.transpose(2, 0, 1)
    return [img]


def img_to_HDF5(tf_type, pretrained_file=None):
    """
    Gather the data in a single HDF5 file.
    Resize/ crop the data before dumping it,
    as specified by tf_type

    args: tf_type (str) the transformation to apply to
                         the image before dumping it
          pretrained_file (str) test submission file with predictions from a previous model
    """

    # Get the env var
    raw_dir = os.environ.get("RAW_DIR")
    data_dir = os.environ.get("DATA_DIR")
    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "test")

    # Get the image transformation function
    if tf_type == "resize_VGG":
        tf = resize_VGG
        out_file_name = "fullframe_VGG"
        c, h, w = 3, 224, 224

    # Read the driver file
    driver_file = os.path.join(raw_dir, "driver_imgs_list.csv")
    df = pd.read_csv(driver_file)
    # Merge class and img to get path name
    df["img_path"] = train_dir + "/" + df.classname + "/" + df.img
    # Convert class to int
    df["class_int"] = df.classname.apply(lambda x: int(x[1:]))

    # Randomize dtaframe for heterogeneous folds
    np.random.seed(10)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)

    # Put train data in HDF5
    hdf5_file = os.path.join(data_dir, "%s_train.h5" % out_file_name)
    with h5py.File(hdf5_file, "w") as hf:

        train_data = hf.create_dataset("train_data",
                                       (0, c, h, w),
                                       maxshape=(None, c, h, w),
                                       dtype=np.uint8)
        train_label = hf.create_dataset("train_label",
                                        (0,),
                                        maxshape=(None, )
                                        )
        hf.create_dataset("train_imgname", data=df.img.values.astype(str))

        # Get datasets with the index of the train/valid split
        list_driver = np.sort(df.subject.unique())
        # Split in chunks of 3 to 4 drivers
        num_drivers = len(list_driver)
        num_chunks = num_drivers / 3
        list_chunks = np.array_split(np.arange(num_drivers), num_chunks)

        for idx, chunk in enumerate(list_chunks):
            idx_train = df[~np.in1d(df.subject.values, list_driver[chunk])].index.values
            idx_valid = df[np.in1d(df.subject.values, list_driver[chunk])].index.values
            hf.create_dataset("train_fold%s" % idx, data=idx_train)
            hf.create_dataset("valid_fold%s" % idx, data=idx_valid)

        # Split image list into num_chunks chunks
        num_imgs = len(df)
        chunk_size = 1000
        num_chunks = num_imgs / chunk_size
        list_chunks = np.array_split(np.arange(num_imgs), num_chunks)

        # Loop over dataframe
        for index, chunk_idx in enumerate(list_chunks):
            sys.stdout.write("\rProcessing train chunk %s/%s" %
                             (index + 1, len(list_chunks)))
            sys.stdout.flush()

            list_img_path = df.img_path.values[chunk_idx]
            output = parmap.map(format_image, list_img_path, tf, parallel=True)

            arr_img = np.vstack(output)
            arr_label = df.class_int.values[chunk_idx]

            # Resize HDF5 datasetset
            train_data.resize(train_data.shape[0] + len(chunk_idx), axis=0)
            train_label.resize(train_label.shape[0] + len(chunk_idx), axis=0)
            # Add to HDF5 train_data
            train_data[-len(chunk_idx):, :, :, :] = arr_img.astype(np.uint8)
            train_label[-len(chunk_idx):] = arr_label

    print
    hdf5_file = os.path.join(data_dir, "%s_test.h5" % out_file_name)
    with h5py.File(hdf5_file, "w") as hf:
        # Read a submission file
        if pretrained_file:
            df_test = pd.read_csv(pretrained_file)
        else:
            list_test_img = [os.path.basename(imgt) for imgt in glob.glob(os.path.join(raw_dir, "test/*"))]
            df_test = pd.DataFrame(np.array(list_test_img), columns=["img"])
            for c in map(lambda x: "c" + str(x), range(10)):
                df_test[c] = np.zeros(len(df_test))

        df_test["img_path"] = test_dir + "/" + df_test["img"]
        # Shuffle the data
        df_test = df_test.iloc[np.random.permutation(len(df_test))]
        df_test = df_test.reset_index(drop=True)
        # Create an array with the list of images
        arr_img_test = df_test.img_path.values
        arr_img_label = df_test[map(lambda x: "c" + str(x), range(10))].values
        list_test_id = [os.path.basename(imgt) for imgt in arr_img_test]
        # Create dataset with image names
        hf.create_dataset("test_id", data=np.array(list_test_id))
        # Create dataset with pseudo labels
        hf.create_dataset("test_label", data=arr_img_label)
        # Finally, an array for the data
        test_data = hf.create_dataset("test_data",
                                      (0, c, h, w),
                                      maxshape=(None, c, h, w),
                                      dtype=np.uint8)
        hf.create_dataset("test_imgname", data=np.array(arr_img_test).astype(str))

        # Split image list into num_chunks chunks
        num_imgs = len(arr_img_test)
        chunk_size = 1000
        num_chunks = num_imgs / chunk_size
        list_chunks = np.array_split(np.arange(num_imgs), num_chunks)

        # Loop over dataframe
        for index, chunk_idx in enumerate(list_chunks):
            sys.stdout.write("\rProcessing test chunk %s/%s" %
                             (index + 1, len(list_chunks)))
            sys.stdout.flush()

            list_img_path = arr_img_test[chunk_idx]
            output = parmap.map(format_image, list_img_path, tf, parallel=True)

            arr_img = np.vstack(output)

            # Resize HDF5 datasetset
            test_data.resize(test_data.shape[0] + len(chunk_idx), axis=0)
            # Add to HDF5 test_data
            test_data[-len(chunk_idx):, :, :, :] = arr_img.astype(np.uint8)

        print


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    # Check the env variables exist
    raw_msg = "Set your raw data absolute path in the .env file at project root"
    data_msg = "Set your processed data absolute path in the .env file at project root"
    assert "RAW_DIR" in os.environ, raw_msg
    assert "DATA_DIR" in os.environ, data_msg

    tf_type = "resize_VGG"
    pretrained_file = None  # Previous submission with soft target for test images

    img_to_HDF5(tf_type, pretrained_file)
