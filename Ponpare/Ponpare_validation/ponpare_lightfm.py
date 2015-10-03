import pandas as pd 
import numpy as np 
import scipy.io as spi
import scipy.sparse as sps
import sys
sys.path.append("../Ponpare_utilities/") 
import mean_average_precision as mapr
from lightfm import LightFM
import cPickle as pickle

def fit_model(week_ID, no_comp, lr, ep):
	""" Fit the lightFM model to all weeks in list_week_ID.
	Then print the results for MAPat10
	
	args : week_ID validation test week
	no_comp, lr, ep = (int, float, int) number of components, learning rate, number of epochs for lightFM model

    returns: d_user_pred, list_user, list_coupon
    list_coupon = list of test coupons 
    list_user = list of user ID 
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon

	"""

	print "Fit lightfm model for %s" % week_ID

	#Load data
	Mui_train = spi.mmread("../Data/Validation/%s/biclass_user_item_train_mtrx_%s.mtx" % (week_ID, week_ID))
	uf        = spi.mmread("../Data/Validation/%s/user_feat_mtrx_%s.mtx" % (week_ID, week_ID))
	itrf      = spi.mmread("../Data/Validation/%s/train_item_feat_mtrx_%s.mtx" % (week_ID, week_ID))
	itef      = spi.mmread("../Data/Validation/%s/test_item_feat_mtrx_%s.mtx" % (week_ID, week_ID))

	#Print shapes as a check
	print "user_features shape: %s,\nitem train features shape: %s,\nitem test features shape: %s"   % (uf.shape, itrf.shape, itef.shape)

	#Load test coupon  and user lists
	cplte       = pd.read_csv("../Data/Validation/" + week_ID + "/coupon_list_test_validation_" + week_ID +".csv")
	ulist       = pd.read_csv("../Data/Validation/" + week_ID + "/user_list_validation_" + week_ID +".csv")
	list_coupon = cplte["COUPON_ID_hash"].values
	list_user   = ulist["USER_ID_hash"].values

	#Build model
	no_comp, lr, ep = 10, 0.01, 5
	model = LightFM(no_components=no_comp, learning_rate=lr, loss='warp')
	model.fit_partial(Mui_train, user_features = uf, item_features = itrf, epochs = ep, num_threads = 4, verbose = True)

	test               = sps.csr_matrix((len(list_user), len(list_coupon)), dtype = np.int32)
	no_users, no_items = test.shape
	pid_array          = np.arange(no_items, dtype=np.int32)

	#Create and initialise dict to store predictions
	d_user_pred = {}
	for user in list_user :
		d_user_pred[user] = []
	
	# Loop over users and compute predictions
	for user_id, row in enumerate(test):
		sys.stdout.write("\rProcessing user " + str(user_id)+"/ "+str(len(list_user)))
		sys.stdout.flush()
		uid_array         = np.empty(no_items, dtype=np.int32)
		uid_array.fill(user_id)
		predictions       = model.predict(uid_array, pid_array,user_features = uf, item_features = itef, num_threads=4)
		user              = str(list_user[user_id])
		d_user_pred[user] = predictions

	# Pickle the predictions for future_use
	d_pred = {"list_coupon" : list_coupon.tolist(), "d_user_pred" : d_user_pred}
	with open("../Data/Validation/%s/d_pred_lightfm_%s.pickle" % (week_ID, week_ID), "w") as f:
		pickle.dump(d_pred, f, protocol = pickle.HIGHEST_PROTOCOL)

	return d_user_pred, list_user, list_coupon

def score_lightFM(no_comp, lr, ep):
	"""
	Score the lightFM model for mean average precision at k = 10

	args = no_comp, lr, ep (int, float, int) 
	number of components, learning rate, number of epochs for lightFM model
	"""
       
	list_score = []

	# Loop over validation weeks
	for week_ID in ["week51"] :
		#Get predictions, manually choose metric and classifier
		d_user_pred, list_user_full, list_coupon   = fit_model(week_ID, no_comp, lr, ep)
		#Format predictions
		for index, user in enumerate(list_user_full) :
			list_pred         = d_user_pred[user]
			top_k             = np.argsort(-list_pred)[:10]
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
		
		list_user = d_user_purchase.keys()
		list_actual = [d_user_purchase[key] for key in list_user]
		list_pred = [d_user_pred[key] for key in list_user] 

        list_score.append(mapr.mapk(list_actual, list_pred))
        print list_score

	list_score = np.array(list_score)
	print list_score 
	print str(np.mean(list_score)) + " +/- " + str(np.std(list_score))

if __name__ == "__main__": 

	no_comp, lr, ep = 10, 0.01, 10
	score_lightFM(no_comp, lr, ep)
