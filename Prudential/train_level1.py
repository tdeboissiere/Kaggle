import numpy as np 
import class_utilities_lv1 as class_ut


def train_level1(d_model) : 
	"""
	Main function to call in order to train 
	all level1 models 

	args : d_model (dict) stores all the information (hyperparameter, name etc)
		of level1 models
	"""

	#Create instance for Lvl1 models
	Lv1Model = class_ut.Level1Model(d_model)

	#Prepare predictions for CV
	Lv1Model.save_oos_pred()

	# Predict Lv1 features for test set
	Lv1Model.save_test_pred()


if __name__ == "__main__":

	# Dict to hold all regressor information 
	d_model = {}

	param_bin                  = {}
	param_bin['objective']     = "binary:logistic"
	param_bin['eval_metric']   = 'auc'
	param_bin['silent']        = 1
	param_bin['eta']       = 0.1
	param_bin['max_depth'] = 5
	param_bin["subsample"] = 0.5
	param_bin["colsample_bytree"] = 0.7
	
	d_bin = {"param" : param_bin, 
			"num_round" : 200, 
			"train" : False, 
			"feat_choice":"xgb_bin", 
			"n_folds":10
			}

	d_model["xgb_bin1"] = d_bin.copy()
	d_model["xgb_bin2"] = d_bin.copy()
	d_model["xgb_bin3"] = d_bin.copy()
	d_model["xgb_bin4"] = d_bin.copy()
	d_model["xgb_bin5"] = d_bin.copy()
	d_model["xgb_bin6"] = d_bin.copy()
	d_model["xgb_bin7"] = d_bin.copy()
	d_model["xgb_bin8"] = d_bin.copy()

	# xgb parameters
	param_reg                  = {}
	param_reg['objective']     = "reg:linear"
	param_reg['eval_metric']   = 'rmse'
	param_reg['silent']        = 1
	param_reg['eta']       = 0.01
	param_reg['max_depth'] = 7
	param_reg["subsample"] = 0.5
	param_reg["gamma"] = 1
	param_reg["colsample_bytree"] = 0.5
	# param["min_child_weight"] = 100
	
	d_reg = {"param" : param_reg, 
			"num_round" : 3000, 
			"train" : False, 
			"feat_choice":"xgb_reg", 
			"n_folds":10
			}

	d_model["xgb_reg"] = d_reg

	d_model["knnreg"] = {"train" : False, "feat_choice" : "knn", "n_neighbors" : 9, "n_folds":10}
	d_model["linreg"] = {"train" : False, "feat_choice" : "linreg", "n_folds":10}
	d_model["cosine"] = {"train" : False, "feat_choice" : "cosine", "n_folds":10}
	d_model["logistic"] = {"train" : False, "feat_choice" : "logistic", "n_folds":10}
	
	d_model["keras_reg1"] = {"train" : False, "feat_choice" : "keras_reg1", "n_folds":5}

	# d_model["xgb_class"] = {"param" : param.copy(), "num_round" : 100, "train" : False, "feat_choice":"corr"}

	train_level1(d_model)
