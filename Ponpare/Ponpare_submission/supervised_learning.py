import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
import sys , os
from datetime import date
import xgboost as xgb
import calendar
import cPickle as pickle
from pysofia.compat import RankSVM
from sklearn.preprocessing import MinMaxScaler

def prepare_test_df(ulist, cplte, list_col_test):
    """ Create the test data set by taking the cartesian product 
    of the user list and the list of test coupons.
    Store to a .csv file
    This is to be able to apply the supervised learning method easily.

    N.B This step takes a LOT of memory

    args : ulist (df) dataframe of the list of users 
           cplte (df) dataframe of the list of test coupons 
           list_col_test (list) list of columns (= features) to keep
    """

    # Create new col and merge on it to obtain the cartesian product
    ulist["key"] = 1
    cplte["key"] = 1
    df = pd.merge(ulist, cplte, on = "key")
    del df["key"]
    df["INTEREST"] = -1.
    # Select the features of interest
    df = df[list_col_test]
    #NA imputation
    df = df.fillna(-1)

    #Save to .csv
    df.to_csv("../Data/Data_translated/test_supervised_learning.csv", index = False)
    return

def prepare_data():
    """Format data to feed to a classifier

    The target is the "interest" of the user for a given coupon.
    The features are the user characteristics + coupon characteristics 

    The target is the user "INTEREST" defined as follows for the train set :
    target = 1 if user has purchased the coupon 
    target = sigmoid(# views) if user has visited the coupon 
    target = 0 otherwise

    returns : X_train, y_train, d_info
    the feature matrix, target vector and a dictionnary
    to store relevant information for classification
    """

    def get_unix_time(row):
        """Convert to unix time. Neglect time of the day
        """
        row = row.split(" ")
        row = row[0].split("-")
        y,m,d = int(row[0]), int(row[1]), int(row[2])
        return calendar.timegm(date(y,m,d).timetuple())

    def get_day_of_week(row):
        """Convert to unix time. Neglect time of the day
        """
        row = row.split(" ")
        row = row[0].split("-")
        y,m,d = int(row[0]), int(row[1]), int(row[2])

        return date(y,m,d).weekday()

    def sigmoid(val):
        """Modified sigmoid function
        """
        return 0.5/(1+np.exp(-0.1*val)) #0.027 for week 49

    cpdtr = pd.read_csv("../Data/Data_translated/coupon_detail_train_translated.csv")
    cpvtr = pd.read_csv("../Data/Data_translated/coupon_visit_train_translated.csv")
    cpltr = pd.read_csv("../Data/Data_translated/coupon_list_train_translated.csv")
    cplte = pd.read_csv("../Data/Data_translated/coupon_list_test_translated.csv")
    ulist = pd.read_csv("../Data/Data_translated/user_list_translated.csv")

    #Select coupons viewed but not purchased
    cpvtr_nop = cpvtr[cpvtr["PURCHASE_FLG"] == 0].copy()
    #Groupby and count # of view for cpvtr_nop
    cpvtr_nop = cpvtr_nop[["VIEW_COUPON_ID_hash", "USER_ID_hash"]]
    # Define the user INTEREST. Default = 0
    cpvtr_nop["INTEREST"] = 0
    g = cpvtr_nop.groupby(["VIEW_COUPON_ID_hash", "USER_ID_hash"])
    # Apply count function => INTEREST now contains the number of time the coupons were viewed by each user
    cpvtr_nop = g.count()
    # Reset index or else, COUP and USER become indices
    cpvtr_nop.reset_index(inplace = True)

    # For coupons the user has viewed more han once, apply sigmoid on the number of views
    cpvtr_nop.loc[cpvtr_nop["INTEREST"]>0, "INTEREST"] = cpvtr_nop["INTEREST"][cpvtr_nop["INTEREST"]>0].apply(sigmoid)  

    cpvtr = cpvtr_nop
    cpvtr.columns = ["COUPON_ID_hash", "USER_ID_hash", "INTEREST"]

    # Feature engineering
    cplte["DISPFROM_day"] = cplte["DISPFROM"].apply(get_day_of_week)
    cpltr["DISPFROM_day"] = cpltr["DISPFROM"].apply(get_day_of_week)
    cplte["DISPEND_day"] = cplte["DISPEND"].apply(get_day_of_week)
    cpltr["DISPEND_day"] = cpltr["DISPEND"].apply(get_day_of_week)

    #Binarize sex id this way
    ulist.loc[ulist["SEX_ID"] == "f", "SEX_ID"] = 0
    ulist.loc[ulist["SEX_ID"] == "m", "SEX_ID"] = 1

    #Format REG_DATE
    ulist["REG_DATE_UNIX"] = ulist["REG_DATE"].apply(get_unix_time)

    # List of  features to keep
    list_col = ["COUPON_ID_hash","USER_ID_hash", "GENRE_NAME", "large_area_name", "small_area_name", "PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
    "DISPFROM_day", "DISPEND_day", "DISPPERIOD", "VALIDFROM", "VALIDEND",
       "VALIDPERIOD", "USABLE_DATE_MON", "USABLE_DATE_TUE",
       "USABLE_DATE_WED", "USABLE_DATE_THU", "USABLE_DATE_FRI",
       "USABLE_DATE_SAT", "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY",
       "USABLE_DATE_BEFORE_HOLIDAY", "AGE", "SEX_ID", "REG_DATE_UNIX", "INTEREST"]

    #making of the train set
    train = pd.merge(cpdtr, cpltr)
    train = pd.merge(ulist, train, left_on = "USER_ID_hash", right_on = "USER_ID_hash")
    #Interest set to 1 for purchased items
    train["INTEREST"] = 1.0
    train = train[list_col]

    #making of the trainv set
    trainv = pd.merge(cpvtr, cpltr, left_on = "COUPON_ID_hash", right_on = "COUPON_ID_hash")
    trainv = pd.merge(trainv, ulist, left_on = "USER_ID_hash", right_on = "USER_ID_hash")
    trainv = trainv[list_col]

    #Create a dedicated .csv file for the test set (= cartesian product of users and test coupon list)
    if not os.path.isfile("../Data/Data_translated/test_supervised_learning.csv" ) :
        prepare_test_df(ulist, cplte, list_col)

    #NA imputation
    train = train.fillna(-1)
    trainv = trainv.fillna(-1)

    #Append trainv to train 
    train = train.append(trainv)

    # Drop coupon info
    train = train.drop("COUPON_ID_hash", 1)
    list_col = train.columns.values

    #Add users who have no information (no visit, no purchase) to train dataset
    l1 = train["USER_ID_hash"].drop_duplicates().values
    l2 = ulist["USER_ID_hash"].drop_duplicates().values
    user_toadd = l2[np.in1d(l2,l1, invert = True)]
    user_toadd_index = np.array([ulist["USER_ID_hash"].tolist().index(user) for user in user_toadd])
    #Create a dataframe for such users
    train_toadd = pd.DataFrame(user_toadd, columns = ["USER_ID_hash"])
    #Known user features are taken from ulist
    list_user_feat = ["USER_ID_hash", "AGE", "SEX_ID", "REG_DATE_UNIX"]
    for feat in list_user_feat:
        train_toadd[feat] = ulist[feat].values[user_toadd_index]
    #Other features are taken as the mode (most common value) of said features in train
    list_other_feat = [feat for feat in list_col if feat not in list_user_feat]
    for feat in list_other_feat:
        train_toadd[feat] = train[feat].mode().values[0]      

    # Complete our train dataframe by appending the dataframe 
    # of users who have no visit/purchase information
    train = train.append(train_toadd)

    # Use an sklearn_pandas mapper for Label Encoding
    list_mapper      = []
    for feat in list_col :
        if feat in ["USER_ID_hash", "GENRE_NAME", "large_area_name", "small_area_name"] :
            list_mapper.append((feat, preprocessing.LabelEncoder()))
        else :
            list_mapper.append((feat, None))

    # Fit mapper
    mapper = DataFrameMapper(list_mapper)
    train = mapper.fit_transform(train) 
    train = pd.DataFrame(train, index = None, columns = list_col )

    # Define the list of user and the list of coupon 
    list_user = sorted(list(ulist["USER_ID_hash"].values))
    list_coupon = cplte["COUPON_ID_hash"].values

    list_col_xgb = ["USER_ID_hash", "GENRE_NAME", "large_area_name", "small_area_name", "PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
    "DISPFROM_day", "DISPEND_day", "DISPPERIOD", "AGE", "SEX_ID", "REG_DATE_UNIX"]
    y_train = train["INTEREST"].values.astype(float)
    train = train[list_col_xgb]
    X_train = train.values

    #Store all infos needed to apply classification
    d_info = {}
    d_info["list_user"] = list_user
    d_info["list_coupon"] = list_coupon
    d_info["no_cpt"] = cplte.shape[0]
    d_info["mapper"] = mapper
    d_info["list_col_xgb"] = list_col_xgb
    d_info["list_col_mapper"] = list_col

    return X_train, y_train, d_info


def fit_xgboost(metric):
    """Fit an xgboost model (with the objective metric of interest) 

    returns: d_user_pred, list_user, list_coupon
    list_coupon = list of test coupons 
    list_user = list of user ID 
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon
    """

    #Get data for classification
    X_train, y_train, d_info = prepare_data()

    list_user = d_info["list_user"]
    list_coupon = d_info["list_coupon"]
    no_cpt = d_info["no_cpt"]
    mapper = d_info["mapper"]
    list_col_xgb = d_info["list_col_xgb"]
    list_col_mapper = d_info["list_col_mapper"]

    # Launch xgb
    num_round              = 10
    param                  = {}
    param['objective']     = metric
    param['eval_metric']   = 'rmse'
    param['silent']        = 1
    param['bst:eta']       = 0.1
    param['bst:max_depth'] = 9
    xgmat                  = xgb.DMatrix(X_train, label=y_train)
    plst                   = param.items()+[('eval_metric', 'rmse')]
    watchlist              = [(xgmat, "train")]
    bst                    = xgb.train(plst, xgmat, num_round, watchlist)

    #Store predictions in a dict
    d_user_pred = {}

    #Load test by chunks to avoid memory issues
    for index, test in enumerate(pd.read_csv("../Data/Data_translated/test_supervised_learning.csv", chunksize=1000*no_cpt)) :
        sys.stdout.write("\rProcessing row " + str(index*1000*no_cpt)+" to row "+str((index+1)*1000*no_cpt))
        sys.stdout.flush()
        test = test.fillna(-1)
        temp_list_user = test["USER_ID_hash"].drop_duplicates().values

        test = mapper.transform(test)
        test = pd.DataFrame(test, index = None, columns = list_col_mapper )
        test = test[list_col_xgb]
        X_test = test.values
        y_test = bst.predict(xgb.DMatrix(X_test))
        for i in range(min(1000, len(temp_list_user))):
            user = temp_list_user[i]
            d_user_pred[user] = y_test[i*no_cpt: (i+1)*no_cpt]
    print 

    # Compute score for users, apply MinMaxScaler for blending later on
    for i, user in enumerate(list_user) :
        list_pred = d_user_pred[user]
        MMS = MinMaxScaler()
        pred = MMS.fit_transform(np.ravel(list_pred))
        d_user_pred[user] = pred
        
    # Pickle the predictions for future use
    d_pred = {"list_coupon" : list_coupon.tolist(), "d_user_pred" : d_user_pred}
    with open("../Data/Data_translated/d_pred_xgb_%s.pickle" %("_".join(metric.split(":"))), "w") as f:
        pickle.dump(d_pred, f, protocol = pickle.HIGHEST_PROTOCOL) 

    return d_user_pred, list_user, list_coupon


def fit_SVM():
    """Fit an SVMRank model (from pysofia)

    returns: d_user_pred, list_user, list_coupon
    list_coupon = list of test coupons 
    list_user = list of user ID 
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon
    """
    #Get data for classification
    X_train, y_train, d_info = prepare_data()

    list_user = d_info["list_user"]
    list_coupon = d_info["list_coupon"]
    no_cpt = d_info["no_cpt"]
    mapper = d_info["mapper"]
    list_col_xgb = d_info["list_col_xgb"]
    list_col_mapper = d_info["list_col_mapper"]

    # Create RankSVM
    RSVM = RankSVM(max_iter=10, alpha = 1)
    RSVM.fit(X_train, y_train)

    #Store predictions in a dict
    d_user_pred = {}

    #Load test data by chunks to avoid memory issues
    for index, test in enumerate(pd.read_csv("../Data/Data_translated/test_supervised_learning.csv", chunksize=1000*no_cpt)) :
        sys.stdout.write("\rProcessing row " + str(index*1000*no_cpt)+" to row "+str((index+1)*1000*no_cpt))
        sys.stdout.flush()
        test = test.fillna(-1)
        temp_list_user = test["USER_ID_hash"].drop_duplicates().values

        test = mapper.transform(test)
        test = pd.DataFrame(test, index = None, columns = list_col_mapper )
        test = test[list_col_xgb]
        X_test = test.values
        y_test = RSVM.rank(X_test)
        for i in range(min(1000, len(temp_list_user))):
            user = temp_list_user[i]
            d_user_pred[user] = y_test[i*no_cpt: (i+1)*no_cpt]
    print

    # Compute score for users, apply MinMaxScaler for blending later on
    for i, user in enumerate(list_user) :
        list_pred = d_user_pred[user]
        MMS = MinMaxScaler()
        pred = MMS.fit_transform(np.ravel(list_pred).astype(float))
        d_user_pred[user] = pred

    # Pickle the predictions for future use
    d_pred = {"list_coupon" : list_coupon.tolist(), "d_user_pred" : d_user_pred}
    with open("../Data/Data_translated/d_pred_SVM.pickle", "w") as f:
        pickle.dump(d_pred, f, protocol = pickle.HIGHEST_PROTOCOL) 

    return d_user_pred, list_user, list_coupon

def score_supervised_learning():
    """ Score full xgb model predictions
    """

    # Three possible classifiers
    list_clf_name = ["rank:pairwise", "reg:linear", "SVM"]
    list_file_name = ["xgb_rank_pairwise", "xgb_reg_linear", "SVM"]

    #Loop over classifiers
    for clf_name, file_name in zip(list_clf_name, list_file_name) :
        if clf_name == "rank:pairwise":
            d_user_pred, list_user_full, list_coupon = fit_xgboost(clf_name)
        elif clf_name == "reg:linear":
            d_user_pred, list_user_full, list_coupon = fit_xgboost(clf_name)
        elif clf_name == "SVM":
            d_user_pred, list_user_full, list_coupon = fit_SVM()

        #Write submission
        with open("../Submissions/submission_%s.csv" % file_name, "w") as f:
            f.write("USER_ID_hash,PURCHASED_COUPONS\n")
            for index, user in enumerate(list_user_full) :
                list_pred = d_user_pred[user]
                top_k = np.argsort(-list_pred)[:10]
                user_pred = list_coupon[top_k]
                f.write(user + "," + " ".join(user_pred) + "\n")

if __name__ == "__main__":

    score_supervised_learning()