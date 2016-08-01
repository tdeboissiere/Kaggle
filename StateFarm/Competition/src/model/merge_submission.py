from dotenv import load_dotenv, find_dotenv
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
import os
import glob
import numpy as np
import json
import pandas as pd
import datetime


def merge(merge_type):

    list_path = []

    list_path += glob.glob("../../models/VGGCAM/Experiment_3/*.csv")

    list_df = [pd.read_csv(p) for p in list_path]

    # Sort by images
    list_df_sort = []
    for df in list_df:
        df = df.sort_values("img")
        df = df.reset_index(drop=True)
        list_df_sort.append(df)
    list_df = list_df_sort

    # Check the sort
    list_img = list_df[0].img.values
    for i, df in enumerate(list_df[1:]):
        print list_path[i], np.array_equal(list_img, df.img.values)

    list_c = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    if merge_type == "geom":
        arr_img = list_df[0]["img"].values
        for df in list_df[1:]:
            assert np.array_equal(arr_img, df.img.values)
        # Get the geom mean
        from scipy.stats.mstats import gmean
        df_geom = list_df[0].copy()
        for col in list_c:
            df_geom[col] = gmean(np.vstack([df[col].values for df in list_df]).T, axis=1)
        sumc = df_geom[list_c].sum(1)
        for col in list_c:
            df_geom[col] /= sumc
        # Check against a working df
        df2 = pd.read_csv('../../reports/submissions/geom_resnet_E5-7-8-9.csv')
        plt.scatter(df.c0, df2.c0)
        plt.show()
        print df_geom.head()
        df_geom.to_csv(
            "../../reports/submissions/geom_VGGCAM_E3.csv", index=False)

    elif merge_type == "voting":
        list_argmax = []
        for df in list_df:
            list_argmax.append(np.argmax(df[list_c].values, 1))
        arr_argmax = np.vstack(list_argmax).T
        list_prob = []
        for i in range(arr_argmax.shape[0]):
            arr = [len(np.where(arr_argmax[i, :] == k)[0]) / float(arr_argmax.shape[1]) for k in range(10)]
            list_prob.append(arr)
        arr_prob = np.vstack(list_prob)
        print arr_prob.shape
        df_vote = list_df[0]["img"].copy().to_frame()
        for i, c in enumerate(list_c):
            df_vote[c] = arr_prob[:, i]
        df2 = pd.read_csv('../../reports/submissions/geom_resnet_E5-7-8-9.csv')
        plt.scatter(df_vote.c0, df2.c0)
        plt.show()
        df_vote.to_csv(
            "../../reports/submissions/vote_good_submissions.csv", index=False)
        raw_input()

    else:
        df_mean = pd.concat(list_df).groupby("img").mean()
        print df_mean.head()
        df_mean.reset_index(level='img', inplace=True)
        print df_mean.head()
        # Check against a working df
        df2 = pd.read_csv('../../reports/submissions/geom_resnet_E5-7-8-9.csv')
        plt.scatter(df_mean.c0, df2.c0)
        plt.show()
        df_mean.to_csv(
            "../../reports/submissions/avg_good_submissions.csv", index=False)

if __name__ == '__main__':

    # reprocess_python3_csv()
    merge_resnet(merge_type="harmo")
    # merge_single_out(merge_type="soft")
    # merge_augmented_inference(merge_type="mean")
    # merge_a_la_carte(merge_type="geom")
    # merge_VGG19(merge_type="geom")

    # VGG1_path = "../../models/ResNet/Experiment_5/"
    # get_list_files(VGG1_path, model_name="resnet")
