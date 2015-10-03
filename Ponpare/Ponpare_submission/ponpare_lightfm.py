import pandas as pd 
import numpy as np 
import scipy.io as spi
import scipy.sparse as sps
import sys
from lightfm import LightFM
import cPickle as pickle
from sklearn.preprocessing import MinMaxScaler

def fit_lightfm_model():
	""" Fit the lightFM model 
	
	returns d_user_pred, list_user, list_coupon
	list_coupon = list of test coupons 
	list_user = list of user ID 
	d_user_pred : key = user, value = predicted ranking of coupons in list_coupon
	"""

	#Load data
	Mui_train = spi.mmread("../Data/Data_translated/biclass_user_item_train_mtrx.mtx")
	uf        = spi.mmread("../Data/Data_translated/user_feat_mtrx.mtx")
	itrf      = spi.mmread("../Data/Data_translated/train_item_feat_mtrx.mtx")
	itef      = spi.mmread("../Data/Data_translated/test_item_feat_mtrx.mtx")
	
	#Print shapes as a check
	print "user_features shape: %s,\nitem train features shape: %s,\nitem test features shape: %s"   % (uf.shape, itrf.shape, itef.shape)
	
	#Load test coupon  and user lists
	cplte       = pd.read_csv("../Data/Data_translated/coupon_list_test_translated.csv")
	ulist       = pd.read_csv("../Data/Data_translated/user_list_translated.csv")
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
		# apply MinMaxScaler for blending later on
		MMS               = MinMaxScaler()
		pred              = MMS.fit_transform(np.ravel(predictions))
		d_user_pred[user] = pred

	# Pickle the predictions for future_use
	d_pred = {"list_coupon" : list_coupon.tolist(), "d_user_pred" : d_user_pred}
	with open("../Data/Data_translated/d_pred_lightfm.pickle", "w") as f:
		pickle.dump(d_pred, f, protocol = pickle.HIGHEST_PROTOCOL)

	return d_user_pred, list_user, list_coupon

def score_lightFM():
	"""
	Score the lightFM model for mean average precision at k = 10
	"""

	d_user_pred, list_user, list_coupon = fit_lightfm_model()

	#write submission
	with open("../Submissions/submission_lightfm.csv", "w") as f:
		f.write("USER_ID_hash,PURCHASED_COUPONS\n")
		for index, user in enumerate(list_user) :
			list_pred = d_user_pred[user]
			top_k = np.argsort(-list_pred)[:10]
			user_pred = list_coupon[top_k]
			f.write(user + "," + " ".join(user_pred) + "\n")
	print

if __name__ == "__main__": 

	score_lightFM()

