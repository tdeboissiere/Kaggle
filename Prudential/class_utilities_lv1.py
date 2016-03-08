import numpy as np
##############
#Set the seed
##############
np.random.seed(1000)
print "-"*20
print "SEED set to 1000" 
print "-"*20

from sklearn.base import BaseEstimator
import sklearn.cross_validation as cv
import pandas as pd
import os
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.utils import np_utils
import preprocessing_lv1 as prep
from scipy.spatial.distance import cosine


class Level1Model():
    """Class to train  1st layer models 
    
    Args:
        d_model = (dict) holds the regressors and their name
        folds = (int) number of CV folds to use
    """

    def __init__(self, d_model):
        self.d_model = d_model
        #Create folder to store regression output
        if not os.path.exists("./Data/Level1_model_files/Train/"): 
            os.makedirs("./Data/Level1_model_files/Train/")
        if not os.path.exists("./Data/Level1_model_files/Test/"): 
            os.makedirs("./Data/Level1_model_files/Test/")

    def _get_oos_preds(self, X, y, Id, model_name):
        """
        Use cross validation to 
        get Out of Sample predictions for the model specified by model_name

        args :
                X  : input data of shape (n_samples, n_features)
                y  : target data of shape (n_samples)
                Id : row identification of shape (n_samples)
                model_name (str) : name of the model to train
        """
                
        #Initialize output
        y_pred_oos, list_col = None, None
        if "reg" in model_name :
            y_pred_oos = np.zeros(X.shape[0])
            list_col = ["Id", model_name, "Response"]

        elif "bin" in model_name :
            pos_class = int(model_name[-1])
            # Turn into a binary classification problem
            y[y==pos_class]=20
            y[y!=20] =0
            y[y==20] =1 
            y_pred_oos = np.zeros(X.shape[0])
            list_col = ["Id", "xgb_class_%s" % pos_class,"Response"]

        else :
            y_pred_oos = np.zeros((X.shape[0], len(set(y))))
            list_col = ["Id"]+ [model_name + "_" + str(i+1) for i in range(8)] + ["Response"]

        #Loop over CV folds
        n_folds = self.d_model[model_name]["n_folds"]
        kf = cv.KFold(y.size, n_folds=n_folds)
        for icv, (train_indices, oos_indices) in enumerate(kf):
            print "CV fold:", str(icv+1) + "/" + str(n_folds)
            X_train, y_train = X[train_indices], y[train_indices]
            X_oos = X[oos_indices]

            # XGB regressor
            if "xgb_reg" in model_name :
                xgmat = xgb.DMatrix(X_train, label=y_train)
                param = self.d_model[model_name]["param"]
                param['objective'] = "reg:linear"
                num_round = self.d_model[model_name]["num_round"]
                plst = param.items()
                #Train
                bst = xgb.train(plst, xgmat, num_round)
                # Construct matrix for test set
                xgmat_oos = xgb.DMatrix(X_oos)
                y_pred_oos[oos_indices] = bst.predict(xgmat_oos)

            # XGB classifier
            elif "xgb_class" in model_name :
                xgmat = xgb.DMatrix(X_train, label=y_train)
                param = self.d_model[model_name]["param"]
                param['num_class'] = 8
                param['objective'] = 'multi:softprob'
                num_round = self.d_model[model_name]["num_round"]
                plst = param.items()
                #Train
                bst = xgb.train(plst, xgmat, num_round)
                # Construct matrix for test set
                xgmat_oos = xgb.DMatrix(X_oos)
                y_pred_oos[oos_indices] = bst.predict(xgmat_oos)

            # XGB binary classifier (OVA)
            elif "xgb_bin" in model_name :

                xgmat = xgb.DMatrix(X_train, label=y_train)
                param = self.d_model[model_name]["param"]
                num_round = self.d_model[model_name]["num_round"]
                plst = param.items()
                #Train
                bst = xgb.train(plst, xgmat, num_round)
                # Construct matrix for test set
                xgmat_oos = xgb.DMatrix(X_oos)
                y_pred_oos[oos_indices] = bst.predict(xgmat_oos)

            elif "keras_reg1" in model_name :

                print("Building model...")
                model = Sequential()
                model.add(Dense(64, input_dim=X_train.shape[1], init="glorot_uniform"))  
                model.add(PReLU())
                model.add(BatchNormalization())
                model.add(Dropout(0.5))
                model.add(Dense(64, init='uniform'))
                model.add(PReLU())
                model.add(BatchNormalization())
                model.add(Dropout(0.5))
                model.add(Dense(output_dim=1, init="glorot_uniform"))
                model.add(Activation("linear"))
                model.compile(loss='rmse', optimizer="rmsprop")
                print("Training model...")
                model.fit(X_train, y_train, nb_epoch=30, batch_size=128, verbose=1)
                
                y_pred_oos[oos_indices] = model.predict(X_oos, verbose=0)              

            # Similarity model
            elif "cosine" in model_name:
                # Construct mean vectors
                d_mean = {}
                for i in range(8):
                  d_mean[i] = np.median(X_train[y_train == i+1], axis=0)

                y_pred_cosine = []
                # Get cosine similarity distance
                for i in range(X_oos.shape[0]):
                  vec = np.array([cosine(X_oos[i,:], d_mean[k]) for k in range(8)])
                  y_pred_cosine.append(vec)

                y_pred_oos[oos_indices] = np.array(y_pred_cosine)

            # Linear regression
            elif "linreg" in model_name :
                clf = LinearRegression(n_jobs = -1)
                clf.fit(X_train, y_train)
                y_pred_oos[oos_indices] = clf.predict(X_oos)

            elif "logistic" in model_name :
                clf = LogisticRegression(class_weight="balanced",n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred_oos[oos_indices] = clf.predict_proba(X_oos)

            elif "knnreg" in model_name :
                neighbors = self.d_model[model_name]["n_neighbors"]
                clf = KNeighborsRegressor(n_neighbors=neighbors, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred_oos[oos_indices] = clf.predict(X_oos)

            elif "knnclass" in model_name:
                neighbors = self.d_model[model_name]["n_neighbors"]
                clf = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred_oos[oos_indices] = clf.predict_proba(X_oos)

        return y_pred_oos, list_col

    def save_oos_pred(self) :
        """
        For all the models specified in self.d_model,
        get OOS predictions to get LV1 train features

        Save them to a csv file
        """

        #Loop over estimators
        for model_name in self.d_model.keys() :
            #Only train if desired
            if self.d_model[model_name]["train"]:
                # Get data
                feat_choice = self.d_model[model_name]["feat_choice"]
                X, y, Id = prep.prepare_lv1_data(feat_choice, "train")
                y_keep = y.copy() # keep a copy of y to deal with classifier which may modify y
                print "Compute OOS pred for model: ", model_name
                # Compute OOS preds
                y_pred_oos, list_col = self._get_oos_preds(X, y, Id, model_name)
                #Save predictions to csv file with pandas
                Id = np.reshape(Id, (Id.shape[0],1))
                y_keep = np.reshape(y_keep, (y_keep.shape[0],1))
                y_pred_oos = np.reshape(y_pred_oos, (y_pred_oos.shape[0], len(list_col)-2))
                data = np.hstack((Id, y_pred_oos, y_keep))
                df = pd.DataFrame(data, columns=list_col)
                df[["Id", "Response"]] = df[["Id", "Response"]].astype(int)
                df = df.sort_values("Id")
                # Specify output file name
                out_name = "./Data/Level1_model_files/Train/%s_%s_train.csv" % (model_name, feat_choice)
                # Special case for KNN : specify n_neighbors in out_name
                if "knn" in model_name :
                    out_name = "./Data/Level1_model_files/Train/%s_%s_%s_train.csv" % \
                    (model_name, feat_choice, self.d_model[model_name]["n_neighbors"])
                df.to_csv(out_name, index = False, float_format='%.3f')

    def _get_test_preds(self, X_train, y_train, Id_train, X_test, Id_test, model_name):
        """
        Train model specified by model_name on training data 
        And apply to test data to get test data level2 features 

        args :
                X  : input data of shape (n_samples, n_features) for the train and test samples
                y  : target data of shape (n_samples) only train sample
                Id : row identification of shape (n_samples) for the train and test_sample
                model_name (str) : name of the model to train
        """

        #Initialize output
        y_pred_test, list_col = None, None
        if "reg" in model_name :
            list_col = ["Id", model_name]

        elif "bin" in model_name :
            pos_class = int(model_name[-1])
            # Turn into a binary classification problem
            y_train[y_train==pos_class]=20
            y_train[y_train!=20] =0
            y_train[y_train==20] =1 
            list_col = ["Id", "xgb_class_%s" % pos_class]

        else :
            list_col = ["Id"]+ [model_name + "_" + str(i+1) for i in range(8)] 

        # XGB regressor
        if "xgb_reg" in model_name :
            xgmat = xgb.DMatrix(X_train, label=y_train)
            param = self.d_model[model_name]["param"]
            param['objective'] = "reg:linear"
            num_round = self.d_model[model_name]["num_round"]
            plst = param.items()
            #Train
            bst = xgb.train(plst, xgmat, num_round)
            # Construct matrix for test set
            xgmat_test = xgb.DMatrix(X_test)
            y_pred_test = bst.predict(xgmat_test)

        # XGB binary classifier (OVA)
        elif "xgb_bin" in model_name :

            xgmat = xgb.DMatrix(X_train, label=y_train)
            param = self.d_model[model_name]["param"]
            num_round = self.d_model[model_name]["num_round"]
            plst = param.items()
            #Train
            bst = xgb.train(plst, xgmat, num_round)
            # Construct matrix for test set
            xgmat_test = xgb.DMatrix(X_test)
            y_pred_test = bst.predict(xgmat_test)

        # Keras softmax
        elif "keras_reg1" in model_name :

            print("Building model...")
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], init="glorot_uniform"))  
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(64, init='uniform'))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(output_dim=1, init="glorot_uniform"))
            model.add(Activation("linear"))
            model.compile(loss='rmse', optimizer="rmsprop")
            print("Training model...")
            model.fit(X_train, y_train, nb_epoch=30, batch_size=128, verbose=0)
                
            y_pred_test = model.predict(X_test, verbose=0)   

        # Similarity model
        elif "cosine" in model_name:
            # Construct mean vectors
            d_mean = {}
            for i in range(8):
                d_mean[i] = np.median(X_train[y_train == i+1], axis=0)

            y_pred_cosine = []
            # Get cosine similarity distance
            for i in range(X_test.shape[0]):
                vec = np.array([cosine(X_test[i,:], d_mean[k]) for k in range(8)])
                y_pred_cosine.append(vec)

            y_pred_test = np.array(y_pred_cosine)

        # Linear regression
        elif "linreg" in model_name :
            clf = LinearRegression(n_jobs = -1)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)


        elif "logistic" in model_name :
            clf = LogisticRegression(class_weight="balanced",n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict_proba(X_test)

        elif "knnreg" in model_name :
            neighbors = self.d_model[model_name]["n_neighbors"]
            clf = KNeighborsRegressor(n_neighbors=neighbors, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)

        return y_pred_test, list_col

    def save_test_pred(self) :
        """
        For all the models specified in self.d_model,
        get  LV1 test features

        Save them to a csv file
        """

        if not os.path.exists("./Data/Level1_model_files/Test/"): 
            os.makedirs("./Data/Level1_model_files/Test/")

        #Loop over estimators
        for model_name in self.d_model.keys() :
            #Only train if desired
            if self.d_model[model_name]["train"]:
                # Get data
                feat_choice = self.d_model[model_name]["feat_choice"]
                X_train, y_train, Id_train = prep.prepare_lv1_data(feat_choice, "train")
                X_test,dummy, Id_test = prep.prepare_lv1_data(feat_choice, "test")
                print "Compute test pred for model: ", model_name
                # Compute test preds
                y_pred_test, list_col = self._get_test_preds(X_train, y_train, Id_train, X_test, Id_test, model_name)

                #Save predictions to csv file with pandas
                Id_test = np.reshape(Id_test, (Id_test.shape[0],1))
                y_pred_test = np.reshape(y_pred_test, (y_pred_test.shape[0], len(list_col)-1))
                data = np.hstack((Id_test, y_pred_test))
                df = pd.DataFrame(data, columns=list_col)
                df["Id"] = df["Id"].astype(int)
                df = df.sort_values("Id")

                # Specify output file name
                out_name = "./Data/Level1_model_files/Test/%s_%s_test.csv" % (model_name, feat_choice)
                # Special case for KNN : specify n_neighbors in out_name
                if "knn" in model_name :
                    out_name = "./Data/Level1_model_files/Test/%s_%s_%s_test.csv" % \
                    (model_name, feat_choice, self.d_model[model_name]["n_neighbors"])
                df.to_csv(out_name, index = False, float_format='%.3f')