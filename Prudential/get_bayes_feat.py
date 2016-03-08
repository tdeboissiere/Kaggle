import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import os, sys


def get_bayes_features():
	"""
	Compute Bayes feature on level1 regression models 

	The idea is : let x the predicted value by a regression model 
	We can get P(x|class) by examining the trainig data (where class is the regression target)
	We can estimate P(class) from the training set
	We can then get P(class|x) from Bayes relation (this is the Bayes feature)

	We do this by fitting a KDE to the distribution P(x|class)
	"""

	# Get data for corr feat_choice
	df_train = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
	df_test = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")

	# Initialise dataframes to saves
	df_train_out = df_train[["Id", "Response"]].copy()
	df_test_out = df_test[["Id"]].copy()

	# List of columns of interest
	list_col_reg = ["keras_reg1","knnreg","linreg","xgb_reg"]

	# Clip data
	for col in list_col_reg :
		df_train[col] = np.clip(df_train[col].values,-3,10)

	# Get training data
	d_class = {}
	for i in range(1,9):
		X = df_train[list_col_reg][df_train["Response"]==i].values
		d_class[i-1] = {"keras_reg1": X[:,0].reshape(-1,1),
						"knnreg" : X[:,1].reshape(-1,1),
						"linreg" : X[:,2].reshape(-1,1),
						"xgb_reg" : X[:,3].reshape(-1,1)}

	x_grid = np.linspace(-3,9, 1000)
	# Need to reshape data (1d array are deprecated)
	x_grid = x_grid.reshape(-1,1)

	d_kde = {}
	for i in range(8):
		d_kde[i] = {"keras_reg1" : None, "knnreg": None, "linreg" : None, "xgb_reg" : None}
		for col in list_col_reg :
			kde = KernelDensity(bandwidth=0.2, atol=1E-2, rtol=1E-2)
			print "Fitting class %s" % i, "for", col
			kde.fit(d_class[i][col])
			pdf = np.exp(kde.score_samples(x_grid))
			d_kde[i][col] = {"kde":kde, "pdf":pdf, "pdf_true":d_class[i][col]}

	# Get the train data
	if not os.path.isfile("./Data/Level1_model_files/Train/bayesian_train.csv") :
		for i in range(1,9) :
			for col in list_col_reg :
				print "Computing Bayesian stuff for class %s" %i, "for", col
				# Estimate class probability P(C) from data
				df_train_out["PClass%s_%s" %(col,i)] = float(len(df_train_out[df_train_out["Response"]==i]))/float(len(df_train_out))
				# Get P(X=x|C=c)
				x = df_train[col].values.reshape(-1,1) # Get x, and reshape for kde
				kde = d_kde[i-1][col]["kde"] # Get KDE estimator corresponding to class i
				df_train_out["P%s_%s" % (col,i)] = np.exp(kde.score_samples(x)) # use np.exp because score_sample gives log prob
				df_train_out["FullP%s_%s" % (col,i)] = df_train_out["P%s_%s" % (col,i)]*df_train_out["PClass%s_%s" %(col,i)]

		# Normalise probas and order columns
		for col in list_col_reg :
			lFP = [c for c in df_train_out.columns.values if "FullP%s_" % col in c]
			sum_col = df_train_out[lFP].sum(axis=1).values
			for i in range(1,9):
				df_train_out["FullP%s_%s" % (col,i)]/=sum_col

		# Drop PClass columns
		l_col = [c for c in df_train_out.columns if "Class" in c]
		df_train_out = df_train_out.drop(l_col, 1)

		df_train_out.to_csv("./Data/Level1_model_files/Train/bayesian_train.csv", index = False, float_format = "%.4f")
		sys.exit("Relaunch code now that the file has been created")


	# Get the test data
	if not os.path.isfile("./Data/Level1_model_files/Test/bayesian_test.csv") :
		for i in range(1,9) :
			for col in list_col_reg :
				print "Computing Bayesian stuff for class %s" %i, "for", col
				# Estimate class probability P(C) from data
				df_test_out["PClass%s_%s" %(col,i)] = float(len(df_train_out[df_train_out["Response"]==i]))/float(len(df_train_out))
				# Get P(X=x|C=c)
				x = df_test[col].values.reshape(-1,1) # Get x, and reshape for kde
				kde = d_kde[i-1][col]["kde"] # Get KDE estimator corresponding to class i
				df_test_out["P%s_%s" % (col,i)] = np.exp(kde.score_samples(x)) # use np.exp because score_sample gives log prob
				df_test_out["FullP%s_%s" % (col,i)] = df_test_out["P%s_%s" % (col,i)]*df_test_out["PClass%s_%s" %(col,i)]

		# Normalise probas and order columns
		for col in list_col_reg :
			lFP = [c for c in df_test_out.columns.values if "FullP%s_" % col in c]
			sum_col = df_test_out[lFP].sum(axis=1).values
			for i in range(1,9):
				df_test_out["FullP%s_%s" % (col,i)]/=sum_col

		# Drop PClass columns
		l_col = [c for c in df_test_out.columns if "Class" in c]
		df_test_out = df_test_out.drop(l_col, 1)

		df_test_out.to_csv("./Data/Level1_model_files/Test/bayesian_test.csv", index = False, float_format = "%.4f")
		sys.exit("Relaunch code now that the file has been created")

if __name__ == '__main__':
	
	get_bayes_features()