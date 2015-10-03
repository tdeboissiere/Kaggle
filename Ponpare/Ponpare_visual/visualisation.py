import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
import scipy.sparse as sps
import os, sys 
import cPickle as pickle
import script_utils as script_utils
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import matplotlib.cm as cm

def plot_PCA():

    #Load data
    test = pd.read_csv("./Data/coupon_vector_test.csv")
    train = pd.read_csv("./Data/coupon_vector_train.csv")
    uchar = pd.read_csv("./Data/uchar_train.csv")
    ucharv = pd.read_csv("./Data/ucharv_train.csv")

    col_pca = [el for el in uchar.columns.values if el not in ["USER_ID_hash", "REG_DATE", "SEX_ID", "AGE", "WITHDRAW_DATE", "PREF_NAME"]]
    # col_pca = [el for el in uchar.columns.values if "area" in el]
    # col_pca = [el for el in uchar.columns.values if el in ["DISCOUNT_PRICE", "DISPPERIOD", "USABLE_DATE_sum"]]
    X = uchar[col_pca].values
    Xv = ucharv[col_pca].values
    Xt = test[col_pca].values
    Xtr = train[col_pca].values

    pca = PCA(n_components=2)
    pca.fit(X)
    X_uchar = pca.transform(X.copy())
    X_ucharv = pca.transform(Xv.copy())
    X_test = pca.transform(Xt.copy())
    X_train = pca.transform(Xtr.copy())

    X_uchar = uchar[["DISCOUNT_PRICE", "GENRE_NAME_Nail and eye salon"]].values
    X_ucharv = ucharv[["DISCOUNT_PRICE", "GENRE_NAME_Nail and eye salon"]].values
    X_test = test[["DISCOUNT_PRICE", "GENRE_NAME_Nail and eye salon"]].values
    X_train = train[["DISCOUNT_PRICE", "GENRE_NAME_Nail and eye salon"]].values

    # Percentage of variance explained for each components
    print("explained variance ratio (first two components): %s"
          % str(pca.explained_variance_ratio_))

    print X_uchar[:10]

    plt.figure(1)
    plt.scatter(X_uchar[:,0][(uchar["SEX_ID"] == "f").values], X_uchar[:,1][(uchar["SEX_ID"] == "f").values], c="b", label = "female", alpha = 0.5)
    plt.scatter(X_uchar[:,0][(uchar["SEX_ID"] == "m").values], X_uchar[:,1][(uchar["SEX_ID"] == "m").values], c="r", label = "male", alpha = 0.5)
    plt.legend()
    plt.title("PCA of uchar")


    plt.figure(2)
    plt.scatter(X_ucharv[:,0][(ucharv["SEX_ID"] == "f").values], X_ucharv[:,1][(ucharv["SEX_ID"] == "f").values], c="b", label = "female", alpha = 0.5)
    plt.scatter(X_uchar[:,0][(ucharv["SEX_ID"] == "m").values], X_ucharv[:,1][(ucharv["SEX_ID"] == "m").values], c="r", label = "male", alpha = 0.5)
    plt.legend()
    plt.title("PCA of ucharv")

    plt.figure(3)
    plt.scatter(X_train[:,0], X_train[:,1], c="r", alpha = 0.5, label = "train")
    plt.scatter(X_test[:,0], X_test[:,1], c="b", alpha = 0.5, label = "test")
    plt.title("PCA of test/train")

    plt.figure(4)
    plt.scatter(X_uchar[:,0], X_uchar[:,1], c=uchar["AGE"].values, label = "uchar", alpha = 0.5, cmap = cm.Reds)
    plt.title("Age with colormap")
    plt.show()
    raw_input()

def biplot() :

    #read in all the input data
    cpdtr = pd.read_csv("./Data/coupon_detail_train.csv")
    cpltr = pd.read_csv("./Data/coupon_list_train.csv")
    cplte = pd.read_csv("./Data/coupon_list_test.csv")
    ulist = pd.read_csv("./Data/user_list.csv")

    # Merge detail with user
    m = pd.merge(cpdtr, ulist, left_on = "USER_ID_hash", right_on = "USER_ID_hash")
    m = pd.merge(m, cpltr, left_on = "COUPON_ID_hash", right_on = "COUPON_ID_hash")

    import seaborn as sns

    sns.violinplot(x="AGE", y="CATALOG_PRICE", hue="SEX_ID", data=m)

    # plt.figure(1)
    # plt.scatter(m["CATALOG_PRICE"][(m["SEX_ID"] == "m").values], m["DISCOUNT_PRICE"][(m["SEX_ID"] == "m").values], c="r", label = "male", alpha = 0.5)
    # plt.scatter(m["CATALOG_PRICE"][(m["SEX_ID"] == "f").values], m["DISCOUNT_PRICE"][(m["SEX_ID"] == "f").values], c="b", label = "female", alpha = 0.5)
    # plt.legend()
    # plt.title("Nope")

    # plt.figure(2)
    # plt.scatter(m["AGE"][(m["SEX_ID"] == "f").values], m["CATALOG_PRICE"][(m["SEX_ID"] == "f").values], c="b", label = "female", alpha = 0.5)
    # plt.scatter(m["AGE"][(m["SEX_ID"] == "m").values], m["CATALOG_PRICE"][(m["SEX_ID"] == "m").values], c="r", label = "male", alpha = 0.5)
    # plt.legend()
    # plt.title("Nope")

    plt.show()
    raw_input()


if __name__ == "__main__" :
    
    # plot_PCA()
    biplot()