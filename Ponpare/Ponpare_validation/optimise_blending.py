
import numpy as np
import cPickle as pickle
from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp
import sys
sys.path.append("../Ponpare_utilities/") 
import mean_average_precision as mapr
from sklearn.preprocessing import MinMaxScaler

def optimise_blend(list_week_ID):
    """ Optimise blending weights

    args : list_week_ID = list of validation weeks
    """
    # Initialise dict which will serve later on
    d_user_pred = {}
    d_blend_pred = {}
    d_user_purchase = {}
    d_coupon = {}
    d_user = {}
    for week_ID in list_week_ID :
        d_user_pred[week_ID] = {}
        d_blend_pred[week_ID] = {}

    #Loop over weeks
    for week_ID in list_week_ID :
        #Get actual purchase
        with open( "../Data/Validation/" + week_ID + "/dict_purchase_validation_" + week_ID + ".pickle", "r") as fp:
            d_user_purchase[week_ID] = pickle.load(fp)

        # Open different predictors
        pred_lightfm = {}
        with open("../Data/Validation/%s/d_pred_lightfm_%s.pickle" % (week_ID, week_ID), "r") as f:
            pred_lightfm = pickle.load(f)
        pred_sim = {}
        with open("../Data/Validation/%s/d_pred_similarity_%s.pickle" % (week_ID, week_ID), "r") as f:
            pred_sim = pickle.load(f)
        pred_xgb_rank = {}
        with open("../Data/Validation/%s/d_pred_xgb_rank_pairwise_%s.pickle" % (week_ID, week_ID), "r") as f:
            pred_xgb_rank = pickle.load(f)
        pred_xgb_reg = {}
        with open("../Data/Validation/%s/d_pred_xgb_reg_linear_%s.pickle" % (week_ID, week_ID), "r") as f:
            pred_xgb_reg = pickle.load(f)
        pred_SVM = {}
        with open("../Data/Validation/%s/d_pred_SVM_%s.pickle" % (week_ID, week_ID), "r") as f:
            pred_SVM = pickle.load(f)

        list_coupon = np.array(pred_sim["list_coupon"])
        list_user = pred_sim["d_user_pred"].keys()
        d_coupon[week_ID] = list_coupon
        d_user[week_ID] = list_user

        # Sanity checks
        assert(pred_lightfm["list_coupon"] == pred_sim["list_coupon"])
        assert(pred_lightfm["list_coupon"] == pred_xgb_rank["list_coupon"])
        assert(pred_lightfm["list_coupon"] == pred_xgb_reg["list_coupon"])
        assert(pred_lightfm["list_coupon"] == pred_SVM["list_coupon"])
        assert(set(d_user_purchase[week_ID].keys()) == set(list_user))

        for user in list_user :
            d_user_pred[week_ID][user] = {"sim" : np.array(pred_sim["d_user_pred"][user])}
            d_user_pred[week_ID][user]["xgb_rank"] = np.array(pred_xgb_rank["d_user_pred"][user])
            d_user_pred[week_ID][user]["xgb_reg"] = np.array(pred_xgb_reg["d_user_pred"][user])
            d_user_pred[week_ID][user]["xgb_SVM"] = np.array(pred_SVM["d_user_pred"][user])
            d_user_pred[week_ID][user]["lightfm"] = np.array(pred_lightfm["d_user_pred"][user])
            
            #Scale for blending
            d_user_pred[week_ID][user]["sim"] = MinMaxScaler().fit_transform(d_user_pred[week_ID][user]["sim"].astype(float))
            d_user_pred[week_ID][user]["xgb_rank"] = MinMaxScaler().fit_transform(d_user_pred[week_ID][user]["xgb_rank"].astype(float))
            d_user_pred[week_ID][user]["xgb_reg"] = MinMaxScaler().fit_transform(d_user_pred[week_ID][user]["xgb_reg"].astype(float))
            d_user_pred[week_ID][user]["xgb_SVM"] = MinMaxScaler().fit_transform(d_user_pred[week_ID][user]["xgb_SVM"].astype(float))
            d_user_pred[week_ID][user]["lightfm"] = MinMaxScaler().fit_transform(d_user_pred[week_ID][user]["lightfm"].astype(float))


    def objective_function(x_int):
        objective_function.n_iterations += 1
        list_score = []
        # Parameter to optimise
        w_sim, w_rank, w_reg, w_SVM, w_lightfm = x_int

        for week_ID in list_week_ID :
            for index, user in enumerate(d_user[week_ID]) :
                d_blend_pred[week_ID][user] = d_user_pred[week_ID][user]["sim"]*w_sim
                d_blend_pred[week_ID][user] += d_user_pred[week_ID][user]["xgb_rank"]*w_rank
                d_blend_pred[week_ID][user] += d_user_pred[week_ID][user]["xgb_reg"]*w_reg
                d_blend_pred[week_ID][user] += d_user_pred[week_ID][user]["xgb_SVM"]*w_SVM
                d_blend_pred[week_ID][user] += d_user_pred[week_ID][user]["lightfm"]*w_lightfm
                
                list_pred        = d_blend_pred[week_ID][user]
                top_k = np.argsort(-list_pred)[:10]
                d_blend_pred[week_ID][user] = d_coupon[week_ID][top_k]

            list_user = d_user_purchase[week_ID].keys()
            list_actual = [d_user_purchase[week_ID][key] for key in list_user]
            list_pred = [d_blend_pred[week_ID][key] for key in list_user] 

            list_score.append(mapr.mapk(list_actual, list_pred))

        list_score = np.array(list_score)

        print objective_function.n_iterations, \
            "w_sim, w_rank, w_reg, w_SVM, w_lightfm =", w_sim, w_rank, w_reg, w_SVM, w_lightfm, \
            "\nList_score = ", list_score, \
            "\nMean of MAP = ", np.mean(list_score), \
            "\n Std of MAP = ", np.std(list_score)
        return -np.median(list_score)

    objective_function.n_iterations = 0
    best = hyperopt_fmin(objective_function,
        space=(hp.uniform('w_sim', 0, 1), 
               hp.uniform('w_rank', 0, 1),
               hp.uniform('w_reg', 0, 1),
               hp.uniform('w_SVM', 0, 1),
               hp.uniform('w_lightfm', 0, 1)),
        algo=tpe.suggest,
        max_evals=10)

    print best
    objective_function(tuple([best[key] for key in ["w_sim", "w_rank", "w_reg", "w_SVM", "w_lightfm"]]))    

if __name__ == '__main__':

    list_week_ID = [ "week51"]
    optimise_blend(list_week_ID)
