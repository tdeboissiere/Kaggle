import numpy as np 
import cPickle as pickle
import similarity_distance as sim 
import ponpare_lightfm as pl 
import supervised_learning as sl
import os

def blend_pred():
	""" Blend different predictors (aggregate rank with a weigted arithmetic average)
	"""

	# Open different predictors
	pred_lightfm = {}
	with open("../Data/Data_translated/d_pred_lightfm.pickle", "r") as f:
		pred_lightfm = pickle.load(f)

	pred_sim = {}
	with open("../Data/Data_translated/d_pred_similarity.pickle", "r") as f:
		pred_sim = pickle.load(f)

	pred_xgb_rank = {}
	with open("../Data/Data_translated/d_pred_xgb_rank_pairwise.pickle", "r") as f:
		pred_xgb_rank = pickle.load(f)

	pred_xgb_reg = {}
	with open("../Data/Data_translated/d_pred_xgb_reg_linear.pickle", "r") as f:
		pred_xgb_reg = pickle.load(f)

	pred_SVM = {}
	with open("../Data/Data_translated/d_pred_SVM.pickle", "r") as f:
		pred_SVM = pickle.load(f)

	d_user_pred = {}
	list_coupon = np.array(pred_sim["list_coupon"])
	list_user = pred_sim["d_user_pred"].keys()

	# Blending weights
	w_sim, w_rank, w_reg, w_SVM, w_lightfm = 0.33750809954,0.278275030156,0.23089122473,0.192002801234,0.689639280179

	# Sanity checks
	assert(pred_lightfm["list_coupon"] == pred_sim["list_coupon"])
	assert(pred_lightfm["list_coupon"] == pred_xgb_rank["list_coupon"])
	assert(pred_lightfm["list_coupon"] == pred_xgb_reg["list_coupon"])
	assert(pred_lightfm["list_coupon"] == pred_SVM["list_coupon"])

	# Weighted mean aggregation
	for user in list_user :
		d_user_pred[user] = w_sim*np.array(pred_sim["d_user_pred"][user])
		d_user_pred[user]+=	w_rank*np.array(pred_xgb_rank["d_user_pred"][user])
		d_user_pred[user]+=	w_reg*np.array(pred_xgb_reg["d_user_pred"][user])
		d_user_pred[user]+=	w_SVM*np.array(pred_SVM["d_user_pred"][user])
		d_user_pred[user]+= w_lightfm*np.array(pred_lightfm["d_user_pred"][user])
			
	#write submission
	with open("../Submissions/submission_blended.csv", "w") as f:
		f.write("USER_ID_hash,PURCHASED_COUPONS\n")
		for index, user in enumerate(list_user) :
			list_pred        = d_user_pred[user]
			top_k = np.argsort(-list_pred)[:10]
			user_pred        = list_coupon[top_k]
			f.write(user + "," + " ".join(user_pred) + "\n")

if __name__ == "__main__": 

	#Parameters for the similarity recommender
	var_choice = "1"
	metric = "cosine"

	# If models have not been trained, train them
	if not os.path.isfile("../Data/Data_translated/d_pred_similarity.pickle"):
		print "Fitting similarity model"
		sim.get_similarity_distance(var_choice, metric)
	if not os.path.isfile("../Data/Data_translated/d_pred_lightfm.pickle"):
		print "Fitting lightfm model"
		pl.fit_lightfm_model()
	if not os.path.isfile("../Data/Data_translated/d_pred_xgb_rank_pairwise.pickle"):
		print "Fitting xgb rank_pairwise model"
		sl.fit_xgboost("rank:pairwise")
	if not os.path.isfile("../Data/Data_translated/d_pred_xgb_reg_linear.pickle"):
		print "Fitting xgb reg_linear model"
		sl.fit_xgboost("reg:linear")
	if not os.path.isfile("../Data/Data_translated/d_pred_SVM.pickle"):
		print "Fitting SVM rank model"
		sl.fit_SVM()

	#Blend predictions
	blend_pred()

