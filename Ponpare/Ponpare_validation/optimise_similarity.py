
import pandas as pd
import numpy as np
import scipy.sparse as sps
import cPickle as pickle
import sys
sys.path.append("../Ponpare_utilities/") 
import mean_average_precision as mapr
from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp
from scipy.spatial import distance
import xgboost as xgb


def prepare_similarity(week_ID, var_choice, metric) :
    """ Similarity method to recommend coupons to users.
    Rationale :
    We have built a vector representing each user based on the user's visit history and purchase.
    We then compare this vector with the test coupon vector using a similarity metric.
    We then sort the coupons in the test list given their similarity score.

    args : 
    week_ID (str) validation week
    var_choice (str) choice of feature engineering
    metric (str) metric to compute similarity (cf. scipy.spatial.distance.cdist)

    returns: test_sparse, uchar_sparse, list_user, list_coupon, d_user_pred
    test_sparse = sparse matrix of test coupon features 
    uchar_sparse = sparse matrix of user features
    list_coupon = list of test coupons 
    list_user = list of user ID 
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon

    """

    #Get the list of user dataframe
    ulist = pd.read_csv("../Data/Validation/" + week_ID + "/user_list_validation_" + week_ID +".csv")
    ulist = ulist[["USER_ID_hash", "SEX_ID"]]

    #Binarize SEX_ID 
    ulist["SEX_ID_f"] = 0 
    ulist.loc[ulist["SEX_ID"] == "f", "SEX_ID_f"] = 1
    ulist["SEX_ID_m"] = 0 
    ulist.loc[ulist["SEX_ID"] == "m", "SEX_ID_m"] = 1
    ulist = ulist.drop(["SEX_ID"], 1)

    # Load test coupon with the features of interest (see preprocessing.py for details)
    test = pd.read_csv("../Data/Validation/" + week_ID + "/test_var_" + str(var_choice) + "_validation_" + week_ID +".csv")
    #Load dataframe of user characteristic (obtained by aggregating the features of coupons viewed/purchased) (see preprocessing.py for details)
    uchar = pd.read_csv("../Data/Validation/" + week_ID + "/uchar_var_" + str(var_choice) + "_train_validation_" + week_ID +".csv")

    #######################################################
    # Specificity of my user/coupon similarity model :
    # Add a sex feature to both uchar and test
    # For uchar, SEX_ID_f = 1 if female, else 0
    #            SEX_ID_m = 1 if male, else 0
    # For test, SEX_ID_xx is the probability of the coupon
    # being purchased by a person of sex SEX_ID_xx.
    # This probability is predicted with an xgboost model
    # which we load and apply to the test data
    ########################################################

    # Load xgb models to predict age and sex of purchaser from coupon data
    bst_sex = xgb.Booster(model_file = "../Data/Validation/%s/xgb_model/xgb_sex_pred.model" % week_ID)

    # Load the dataframe /w test coupon information and make predictions /w xgboost model
    test_for_pred = pd.read_csv("../Data/Validation/%s/coupon_list_test_LE_validation_%s.csv" % (week_ID, week_ID) )
    df_test_for_pred = pd.DataFrame(test_for_pred["COUPON_ID_hash"])
    pred_cols =  ["GENRE_NAME", "large_area_name", "small_area_name", "CATALOG_PRICE",
                 "DISPPERIOD", "DISPFROM_day", "DISPEND_day", "PRICE_RATE", "DISCOUNT_PRICE"]
    X_test_for_pred = test_for_pred[pred_cols].values
    # Make predictions /w xgboost model
    xgmat_test_for_pred = xgb.DMatrix(X_test_for_pred)
    df_test_for_pred["SEX_ID"] = bst_sex.predict(xgmat_test_for_pred)
    # Add the SEX_ID feature to test
    test = pd.merge(test, df_test_for_pred, left_on = "COUPON_ID_hash", right_on = "COUPON_ID_hash")

    # We want SEX_ID_f and SEX_ID_m instead of SEX_ID. The lines below make the necessary modifications
    # Add features SEX_ID_f and SEX_ID_m to test and uchar
    uchar = pd.merge(uchar, ulist, left_on = "USER_ID_hash", right_on = "USER_ID_hash")
    test["SEX_ID_f"] = 1-test["SEX_ID"]
    test["SEX_ID_m"] =  test["SEX_ID"]
    test = test.drop(["SEX_ID"], 1)

    # Get sparse uchar and test matrix
    uchar_sparse = sps.csr_matrix(uchar.iloc[:,1:].values.astype(float))
    test_sparse = sps.csr_matrix(test.iloc[:,1:].values.astype(float).T)

    # Define the list of user and the list of coupon 
    list_user = uchar["USER_ID_hash"].values 
    list_coupon = test["COUPON_ID_hash"].values

    #Store predictions in a dict
    d_user_pred = {}
    for user in list_user :
      d_user_pred[user] = []

    return test_sparse, uchar_sparse, list_user, list_coupon, d_user_pred

def optimise_similarity():
    """ Optimise similarity weights
    Use hyperopt to minimize a desired target (example: minimize -median)
    """

    d_uchar         = {}
    d_test          = {}
    d_user_full     = {}
    d_coupon        = {}
    d_user_pred     = {}
    d_user_purchase = {}

    metric = "correlation"
    var_choice = "1"
    list_week = ["week51", "week52"]
    for week_ID in list_week :
        test_sparse, uchar_sparse,list_user, list_coupon, user_pred = prepare_similarity(week_ID,var_choice, metric)
        d_uchar[week_ID]     = uchar_sparse
        d_test[week_ID]      = test_sparse
        d_user_full[week_ID] = list_user
        d_coupon[week_ID]    = list_coupon
        d_user_pred[week_ID] = user_pred
        #Get actual purchase dict
        with open("../Data/Validation/" + week_ID + "/dict_purchase_validation_" + week_ID + ".pickle", "r") as fp:
            d_user_purchase[week_ID] = pickle.load(fp)

        # Take care of users which registered during validation test week
        for key in d_user_purchase[week_ID].keys() :
            try :
                d_user_pred[week_ID][key]
            except KeyError :
                d_user_pred[week_ID][key] = []

    print "Loading OK"

    def objective_function(x_int):
        objective_function.n_iterations += 1

        list_score = []

        # Parameter to optimise
        gnr, disc, disp, large, small, val, us_sum, sex = x_int

        #Build sparse matrix of weights
        Wm = sps.block_diag((gnr*np.eye(13), disc*np.eye(1), disp*np.eye(1), large*np.eye(9), small*np.eye(55), val*np.eye(2), us_sum*np.eye(1), sex*np.eye(2)))
        Wm_sparse = sps.csr_matrix(Wm)

        for week_ID in list_week :

            WmT = Wm_sparse.dot(d_test[week_ID])
            score = 1./distance.cdist(uchar_sparse.todense(), WmT.T.todense(), metric) 

            #Store predictions in a dict
            d_user_pred[week_ID] = {}

            # Compute score  
            for i, user in enumerate(d_user_full[week_ID]) :
                list_pred = np.ravel(score[i,:])
                list_index_top10 = list_pred.argsort()[-10:][::-1]
                d_user_pred[week_ID][user] = d_coupon[week_ID][list_index_top10]

            for key in d_user_purchase[week_ID].keys() :
                try :
                    d_user_pred[week_ID][key]
                except KeyError :
                    d_user_pred[week_ID][key] = []

            list_user = d_user_purchase[week_ID].keys()
            list_actual = [d_user_purchase[week_ID][key] for key in list_user]
            list_pred = [d_user_pred[week_ID][key] for key in list_user] 

            list_score.append(mapr.mapk(list_actual, list_pred))

        list_score = np.array(list_score)

        print objective_function.n_iterations, \
            "gnr, disc, disp, large, small, val, us_sum, sex =", gnr, disc, disp, large, small, val, us_sum, sex, \
            "\nMean of MAP = ", np.mean(list_score), \
            "\n Std of MAP = ", np.std(list_score)
        return -np.min(list_score)

    objective_function.n_iterations = 0
    best = hyperopt_fmin(objective_function,
        space=(hp.uniform('gnr', 0, 5), 
               hp.uniform('disc', 0, 5),
               hp.uniform('disp', 0, 5),
               hp.uniform('large', 0, 5),
               hp.uniform('small', 0, 5),
               hp.uniform('val', 0, 5), 
               hp.uniform('us_sum', 0, 5),
               hp.uniform('sex', 0, 5)),
        algo=tpe.suggest,
        max_evals=10)

    print best
    objective_function(tuple([best[key] for key in ["gnr", "disc", "disp", "large", "small", "val", "us_sum", "sex"]]))

if __name__ == '__main__':

    optimise_similarity()