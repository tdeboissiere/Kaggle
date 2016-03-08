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
from sklearn import svm

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import preprocessing_lv2 as prep
from scipy.spatial.distance import cosine


import matplotlib.pylab as plt
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa


def eval_wrapper(yhat, y):
    """
    Evaluation metric for the competition : quad weighted kappa
    """  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)
        
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    """
    Apply offset to the predictions to improve performance
    """
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

class Level2Model():
    """Class to train  2nd layer models 
    
    Args:
        folds = (int) number of CV folds to use
    """

    def __init__(self, d_model):
        self.d_model = d_model
        #Create folder to store output
        if not os.path.exists("./Data/Level2_model_files/"): 
            os.makedirs("./Data/Level2_model_files/")

    def _score_reg(self, y_pred_cv, y_cv):
        """
        Score function with the quad weighted kappa metric
        """
        return eval_wrapper(y_pred_cv, y_cv)

    def _stack_preds(self, X, y, Id, model_name):
        """
        Estimate the score of the stacked model specified by 
        model_name

        args :
                X  : input data of shape (n_samples, n_features)
                y  : target data of shape (n_samples)
                Id : row identification of shape (n_samples)
                model_name (str) : name of the model to train
        """
        #Loop over CV folds
        n_folds = self.d_model[model_name]["n_folds"]
        kf = cv.KFold(y.size, n_folds=n_folds)
        list_score = []
        num_classes = 8
        for icv, (train_indices, cv_indices) in enumerate(kf):
            print "CV fold:", str(icv+1) + "/" + str(n_folds)
            X_train, y_train = X[train_indices], y[train_indices]
            X_cv, y_cv = X[cv_indices], y[cv_indices]

            # XGB regressor for stacking
            if "xgb_reg" in model_name :
                xgtrain = xgb.DMatrix(X_train, label=y_train)
                param = self.d_model[model_name]["param"]
                param['objective'] = "reg:linear"
                num_round = self.d_model[model_name]["num_round"]
                plst = param.items()
                #Train
                bst = xgb.train(plst, xgtrain, num_round)
                # Construct matrix for test set
                xgcv = xgb.DMatrix(X_cv)
                y_pred_cv = bst.predict(xgcv)
                # Score test sample
                score = self._score_reg(y_pred_cv, y_cv)
                print "Uncalibrated Kappa :", score

                # train offsets 
                print "Train standard offset"
                offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
                data = np.vstack((bst.predict(xgtrain), bst.predict(xgtrain), y_train))
                for j in range(num_classes):
                    data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 
                for j in range(num_classes):
                    train_offset = lambda x: -apply_offset(data, x, j)
                    offsets[j] = fmin_powell(train_offset, offsets[j], disp = False)  

                # apply offsets to test
                data = np.vstack((y_pred_cv, y_pred_cv, y_cv))
                for j in range(num_classes):
                    data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 

                final_y_xgb_cv = np.round(np.clip(data[1], 1, 8)).astype(int)
                print "Calibrated Kappa: ", eval_wrapper(final_y_xgb_cv, y_cv)
                list_score.append(eval_wrapper(final_y_xgb_cv, y_cv))

            # Linear regression for stacking
            elif "linreg" in model_name :
                clf = LinearRegression(n_jobs = -1)
                clf.fit(X_train, y_train)
                y_pred_cv = clf.predict(X_cv)
                # Score test sample
                score = self._score_reg(y_pred_cv, y_cv)
                print "Kappa :", score

                # train offsets 
                print "Train standard offset"
                offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
                data = np.vstack((clf.predict(X_train), clf.predict(X_train), y_train))
                for j in range(num_classes):
                    data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 
                for j in range(num_classes):
                    train_offset = lambda x: -apply_offset(data, x, j)
                    offsets[j] = fmin_powell(train_offset, offsets[j], disp = False)  

                # apply offsets to test
                data = np.vstack((y_pred_cv, y_pred_cv, y_cv))
                for j in range(num_classes):
                    data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 

                final_y_lin_cv = np.round(np.clip(data[1], 1, 8)).astype(int)
                print "Calibrated Kappa: ", eval_wrapper(final_y_lin_cv, y_cv)
                list_score.append(eval_wrapper(final_y_lin_cv, y_cv))

        print np.mean(list_score),"+/-",np.std(list_score)


    def ensemble(self):
        """
        For all the models specified in self.d_model,
        train them and estimate the CV score
        """
        # Loop over possible ensemblers
        for model_name in self.d_model.keys() :
            # Only train if desired
            if self.d_model[model_name]["train"]:
                # Get data
                feat_choice = self.d_model[model_name]["feat_choice"]
                X, y, Id = prep.prepare_lv2_data(feat_choice, "train")
                self._stack_preds(X, y, Id, model_name)

    def _stack_preds_submission(self, X_train, y_train, X_test, model_name):
        """
        Train stacked model on training data and apply to the test data

        args :
                X  : input data of shape (n_samples, n_features)
                y  : target data of shape (n_samples)
                model_name (str) : name of the model to train
        """
        num_classes = 8

        # XGB regressor
        if "xgb_reg" in model_name :
            xgtrain = xgb.DMatrix(X_train, label=y_train)
            param = self.d_model[model_name]["param"]
            param['objective'] = "reg:linear"
            num_round = self.d_model[model_name]["num_round"]
            plst = param.items()
            #Train
            bst = xgb.train(plst, xgtrain, num_round)
            # Construct matrix for test set
            xgtest = xgb.DMatrix(X_test)
            y_pred_test = bst.predict(xgtest)

            # train offsets 
            print "Train standard offset"
            offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
            data = np.vstack((bst.predict(xgtrain), bst.predict(xgtrain), y_train))
            for j in range(num_classes):
                data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 
            for j in range(num_classes):
                train_offset = lambda x: -apply_offset(data, x, j)
                offsets[j] = fmin_powell(train_offset, offsets[j], disp = False)  

            # apply offsets to test
            data = np.vstack((y_pred_test, y_pred_test))
            for j in range(num_classes):
                data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 

            final_y_test = np.round(np.clip(data[1], 1, 8)).astype(int)
            return final_y_test

        # Linear regression
        elif "linreg" in model_name :

            clf = LinearRegression(n_jobs = -1)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            # Score test sample

            # train offsets 
            print "Train standard offset"
            offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
            data = np.vstack((clf.predict(X_train), clf.predict(X_train), y_train))
            for j in range(num_classes):
                data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 
            for j in range(num_classes):
                train_offset = lambda x: -apply_offset(data, x, j)
                offsets[j] = fmin_powell(train_offset, offsets[j], disp = False)  

            # apply offsets to test
            data = np.vstack((y_pred_test, y_pred_test))
            for j in range(num_classes):
                data[1, data[0].astype(int) ==j] = data[0, data[0].astype(int)==j] + offsets[j] 

            final_y_test = np.round(np.clip(data[1], 1, 8)).astype(int)
            return final_y_test


    def ensemble_submission(self):
        """
        For all the models specified in self.d_model,
        train them and apply them on the test data to 
        create a submission file
        """

        # Loop over possible ensemblers
        for model_name in self.d_model.keys() :
            # Only train if desired
            if self.d_model[model_name]["train"]:
                # Get data
                feat_choice = self.d_model[model_name]["feat_choice"]
                X_train, y_train, Id_train = prep.prepare_lv2_data(feat_choice, "train")
                X_test, dummy, Id_test = prep.prepare_lv2_data(feat_choice, "test")
                # X, y, Id = X[:1000,:], y[:1000], Id[:1000]
                y_pred_test = self._stack_preds_submission(X_train, y_train, X_test, model_name)

                #Save predictions to csv file with pandas
                Id_test = np.reshape(Id_test, (Id_test.shape[0],1))
                y_pred_test = np.reshape(y_pred_test, (y_pred_test.shape[0], 1))
                data = np.hstack((Id_test, y_pred_test))
                df = pd.DataFrame(data, columns=["Id", "Response"])
                df["Id"] = df["Id"].astype(int)
                df = df.sort_values("Id")

                out_name = "./Data/Level2_model_files/submission_%s_stack_with_keras.csv" % model_name
                df.to_csv(out_name, index = False)