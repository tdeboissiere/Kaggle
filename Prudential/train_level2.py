import class_utilities_lv2 as class_ut
import class_utilities_lv2_hyperopt as class_ut_hyperopt

def train_level2(d_model) : 
	"""
	Main function to call in order to train 
	all level2 models 

	args : d_model (dict) stores all the information (hyperparameter, name etc)
		of level2 models
	"""

	# Create instance for Lvl2 models
	Lv2Model = class_ut.Level2Model(d_model)

	# #Prepare predictions for CV
	Lv2Model.ensemble()

	# Make submission
	Lv2Model.ensemble_submission()

	# #Prepare predictions for CV
	Lv2Model.ensemble()

if __name__ == "__main__":

	# Dict to hold all regressor information 
	d_model = {}

	param_reg                  = {}
	param_reg['objective']     = "reg:linear"
	param_reg['silent']        = 1
	param_reg['eta']       = 0.075
	param_reg['max_depth'] = 3
	param_reg["subsample"] = 0.6
	param_reg["colsample_bytree"] = 0.9
	param_reg["min_child_weight"] = 17
	
	d_reg = {"param" : param_reg, 
			"num_round" : 145, 
			"train" : False, 
			"feat_choice":"lv1bayes", 
			"n_folds":10
			}

	d_model["xgb_reg"] = d_reg
	d_model["linreg"] = {"train" : False, "feat_choice" : "lv1", "n_folds":10}
	d_model["logistic"] = {"train" : False, "feat_choice" : "lv1", "n_folds":10}
	
	train_level2(d_model)
