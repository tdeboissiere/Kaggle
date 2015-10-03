
import pandas as pd
import numpy as np
import scipy.sparse as sps
import sys, os
sys.path.append("../Ponpare_utilities/") 
import mean_average_precision as mapr
import cPickle as pickle
import script_utils as script_utils
from scipy.spatial import distance
import xgboost as xgb
from sklearn import preprocessing

def predict_sex(week_ID):
    """ This script to fit an xgboost classifier 
    to predict the sex of the likely purchaser of a coupon
    given the coupon characteristics 
    args : week_ID (str) the validation week
    """

    print "Fitting xgboost model to infer purchaser sex from coupon features"

    # Load coupon test vector
    train = pd.read_csv("../Data/Validation/%s/coupon_train_aggregated_LE_validation_%s.csv" % (week_ID, week_ID))
    list_cols =  ["GENRE_NAME", "large_area_name", "small_area_name", "CATALOG_PRICE",
                    "DISPPERIOD", "DISPFROM_day", "DISPEND_day", "PRICE_RATE", "DISCOUNT_PRICE", "SEX_ID"]
    train = train[list_cols]

    list_cols = [col for col in train.columns.values if col != "SEX_ID"]
    X_train = train[list_cols].values
    y_train = train["SEX_ID"].values

    d_feature = {}
    for i in range(len(list_cols)) :
        d_feature["f" + str(i)] = list_cols[i]

    #Label encode y
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    # Launch xgb
    num_round              = 100
    param                  = {}
    param['objective']     = 'binary:logistic'
    param['eval_metric']   = 'auc'
    param['silent']        = 1
    param['bst:eta']       = 0.1
    param['bst:max_depth'] = 9
    xgmat                  = xgb.DMatrix(X_train, label=y_train)
    plst                   = param.items()+[('eval_metric', 'auc')]
    watchlist              = [] #[(xgb.DMatrix(X_cv, label=y_train[cv_indices]), 'eval'), (xgmat, "train")]
    bst                    = xgb.train(plst, xgmat, num_round, watchlist)
        
    #Save xgboost model
    if not os.path.exists("../Data/Validation/%s/xgb_model/" % week_ID):
        os.makedirs("../Data/Validation/%s/xgb_model/" % week_ID) 
    bst.save_model("../Data/Validation/%s/xgb_model/xgb_sex_pred.model" % week_ID)


def get_similarity_distance(week_ID, var_choice, metric) :
    """ Similarity method to recommend coupons to users.
    Rationale :
    We have built a vector representing each user based on the user's visit history and purchase.
    We then compare this vector with the test coupon vector using a similarity metric.
    We then sort the coupons in the test list given their similarity score.

    args : 
    week_ID (str) validation week
    var_choice (str) choice of feature engineering
    metric (str) metric to compute similarity (cf. scipy.spatial.distance.cdist)

    returns: d_user_pred, list_user, list_coupon
    list_coupon = list of test coupons 
    list_user = list of user ID 
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon

    """

    print "Compute similarity distance"

    #Save xgboost model
    if not os.path.isfile("../Data/Validation/%s/xgb_model/xgb_sex_pred.model" % week_ID):
        predict_sex(week_ID)

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


    #Weight each group of features. The weights are optimised by feedback from LB and validation sets.
    cm1 , cm2, cm3, cm4, cm5, cm6, cm7, cm8 = 2, 1.25, 0, 1, 0.625, 4.5, 0, 0.5
    #Build sparse matrix of weights
    Wm = sps.block_diag((cm1*np.eye(13), cm2*np.eye(1), cm3*np.eye(1), cm4*np.eye(9), cm5*np.eye(55), cm6*np.eye(2), cm7*np.eye(1), cm8*np.eye(2)))
    Wm_sparse = sps.csr_matrix(Wm)
    #Build the matrix of users
    uchar_sparse = sps.csr_matrix(uchar.iloc[:,1:].values.astype(float))
    # Build the matrix of test coupons
    test_sparse = sps.csr_matrix(test.iloc[:,1:].values.astype(float).T)
    # Give weight to the features by multiplying Wm and the matrix of test coupons
    WmT = Wm_sparse.dot(test_sparse)
    #Finally compute the score using scipy.spatial.distance.cdist and a choice of metric
    #Use the inverse of the score so that highest score corresponds likeliest purchase
    score = 1./distance.cdist(uchar_sparse.todense(), WmT.T.todense(), metric)

    # Define the list of user and the list of coupon 
    list_user = uchar["USER_ID_hash"].values 
    list_coupon = test["COUPON_ID_hash"].values

    #Store predictions in a dict
    d_user_pred = {}
    for user in list_user :
      d_user_pred[user] = []

    # Compute score for users 
    for i, user in enumerate(list_user) :
        list_pred = score[i,:]
        d_user_pred[user] = np.ravel(list_pred)

    # Pickle the predictions for future use
    d_pred = {"list_coupon" : list_coupon.tolist(), "d_user_pred" : d_user_pred}
    with open("../Data/Validation/%s/d_pred_similarity_%s.pickle" %(week_ID, week_ID), "w") as f:
        pickle.dump(d_pred, f, protocol = pickle.HIGHEST_PROTOCOL) 

    return d_user_pred, list_user, list_coupon


def score_similarity_predictions():
    """ Score cosine similarity predictions
    """

    list_score = []

    # Loop over validation weeks
    for week_ID in ["week51", "week52"] :
        script_utils.print_utility("Training until " + week_ID)
        #Get predictions
        d_user_pred, list_user_full, list_coupon = get_similarity_distance(week_ID, "1", "cosine")
        #Format predictions
        for index, user in enumerate(list_user_full) :
            list_pred = d_user_pred[user]
            top_k       = np.argsort(-list_pred)[:10]
            d_user_pred[user] = list_coupon[top_k]

        #Get actual purchase
        d_user_purchase = {}
        with open("../Data/Validation/" + week_ID + "/dict_purchase_validation_" + week_ID + ".pickle", "r") as fp:
            d_user_purchase = pickle.load(fp)

        # Take care of users which registered during validation test week
        for key in d_user_purchase.keys() :
            try :
                d_user_pred[key]
            except KeyError :
                d_user_pred[key] = []

        list_user = np.array(d_user_purchase.keys())
        permut = np.random.permutation(len(list_user))

        list_actual = [d_user_purchase[key] for key in list_user[permut][:int(len(permut))]]
        list_pred = [d_user_pred[key] for key in list_user[permut][:int(len(permut))]] 

        list_score.append(mapr.mapk(list_actual, list_pred))

    list_score = np.array(list_score)
    print list_score 
    print str(np.mean(list_score)) + " +/- " + str(np.std(list_score))
    return np.mean(list_score)

if __name__ == '__main__':

    score_similarity_predictions()