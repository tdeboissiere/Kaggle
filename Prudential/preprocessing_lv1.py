import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler
from glob import glob
from scipy.spatial  import distance

def compute_distance_feat():
	"""
	Compute distance features :

	For each data sample, we compute the distribution of 
	its correlation with each class (i.e take all members of class 0, compute 
	their correlation distance with the sample => we get the distribution)
	From this distribution, extract percentiles (10,25,50,75,90) which
	will be used as features

	"""
	# Get data
	df_train = pd.read_csv("./Data/Raw/train.csv")
	df_test = pd.read_csv("./Data/Raw/test.csv")

	# Save then drop Id and y
	Id_train = df_train["Id"].values
	y_train = df_train["Response"].values
	df_train = df_train.drop(["Id", "Response"], 1)

	Id_test = df_test["Id"].values
	df_test = df_test.drop(["Id"], 1)

	# Deal with columns with missing values
	df_train = df_train.fillna(-1)
	df_test = df_test.fillna(-1)

	# Encode response
	Enc = LabelEncoder()
	y_train = Enc.fit_transform(y_train)

	# Encode categorical features
	df_train["Product_Info_2"] = Enc.fit_transform(df_train["Product_Info_2"].values)
	df_test["Product_Info_2"] = Enc.transform(df_test["Product_Info_2"].values)

	# Standardize
	s = StandardScaler()
	X_train = df_train.values
	X_test = df_test.values
	X_train = s.fit_transform(X_train)
	X_test = s.transform(X_test)

	X_perc_train = np.zeros((len(df_train),40))
	X_perc_test = np.zeros((len(df_test),40))

	metric = "correlation"
	list_col = []
	step = 1000
	for i in range(8):
		for k in [10, 25, 50, 75, 90]:
			list_col.append(metric+ "_C" + str(i) + "_P" + str(k))

	print "Computing distance data for train"
	for i in range(60):
		sys.stdout.write("\rProcessing row %s to row %s" % (1000*i, 1000*(i+1)))
		sys.stdout.flush()
		vec      = X_train[step*i:min(step*(i+1),X_train.shape[0]),:]
		vec_dist = distance.cdist(vec, X_train, metric = metric)
		for k in range(8):
			X_perc_train[step*i:min(step*(i+1),X_train.shape[0]),5*k:5*(k+1)] = np.percentile(vec_dist[:,y_train==k],[10,25,50,75,90], axis=1).T 
	print

	print "Computing distance data for test"
	for i in range(20):
		sys.stdout.write("\rProcessing row %s to row %s" % (1000*i, 1000*(i+1)))
		sys.stdout.flush()
		vec      = X_test[step*i:min(step*(i+1),X_test.shape[0]),:]
		vec_dist = distance.cdist(vec, X_train, metric = metric)
		for k in range(8):
			X_perc_test[step*i:min(step*(i+1),X_test.shape[0]),5*k:5*(k+1)] = np.percentile(vec_dist[:,y_train==k],[10,25,50,75,90], axis=1).T 

	print

	Id_train = np.reshape(Id_train, (Id_train.shape[0],1))
	y_train = np.reshape(y_train, (y_train.shape[0],1))

	Id_test = np.reshape(Id_test, (Id_test.shape[0],1))

	X_perc_train = np.hstack((Id_train,X_perc_train,y_train))
	X_perc_test = np.hstack((Id_test,X_perc_test))

	df_train_out = pd.DataFrame(X_perc_train, columns = ["Id"] +list_col + ["Response"])
	df_train_out["Id"] = df_train_out["Id"].values.astype(int)
	df_train_out["Response"] = df_train_out["Response"].values.astype(int)+1
	df_train_out.to_csv("./Data/Raw/train_distance_" + metric + ".csv", index=False, float_format='%.3f')

	df_test_out = pd.DataFrame(X_perc_test, columns = ["Id"] +list_col)
	df_test_out["Id"] = df_test_out["Id"].values.astype(int)
	df_test_out.to_csv("./Data/Raw/test_distance_" + metric + ".csv", index=False, float_format='%.3f')


def prepare_lv1_data(feature_choice, file_name):
	"""
	Preprocess the data (feature engineering) to train lv1 models

	args : feature_choice (str) specify the type of feature engineering
		   file_name (str) train or test : preprocess train or test file
	"""

	if feature_choice == "xgb_bin":

		# Get data
		df = pd.read_csv("./Data/Raw/%s.csv" % file_name)
		if file_name == "test":
			df["Response"]=-1
		# Get Id and response
		Id = df["Id"].values
		y = df["Response"].values
		# Drop Id and Response
		df = df.drop(["Id", "Response"], 1)
		# Deal with missing values
		print "Dealing with NaN"
		df["NULL"] = df.isnull().sum(axis=1)
		df = df.fillna(-1)
		#Get tsne data
		print "Getting tsne data"
		df_tsne_full = pd.read_csv("./Data/Raw/tsne_full_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_full", "V2_full"]] = df_tsne_full[["V1", "V2"]]
		df_tsne_binary = pd.read_csv("./Data/Raw/tsne_binary_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_binary", "V2_binary"]] = df_tsne_binary[["V1", "V2"]]
		df_tsne_distance = pd.read_csv("./Data/Raw/tsne_distance_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_distance", "V2_distance"]] = df_tsne_distance[["V1", "V2"]]

		print "Comparison features"
		df["COMP_IH4_IH7"] = df["Insurance_History_4"].values == df["Insurance_History_7"].values
		df["COMP_IH4_IH3"] = np.abs(df["Insurance_History_4"].values - df["Insurance_History_3"].values)
		df["COMP_IH9_IH7"] = np.abs(df["Insurance_History_9"].values - df["Insurance_History_7"].values)
		df["COMP_MH6_MK48"] = np.abs(df["Medical_History_6"].values - df["Medical_Keyword_48"].values)
		df["COMP_MH33_MK23"] = np.abs(df["Medical_History_33"].values - df["Medical_Keyword_23"].values)
		df["COMP_MH37_MK11"] = np.abs(df["Medical_History_37"].values - df["Medical_Keyword_11"].values)
		df["COMP_MH25_MH26"] = np.abs(df["Medical_History_25"].values - df["Medical_History_26"].values)
		
		# factorize categorical variables
		df['Product_Info_2_char'] = df.Product_Info_2.str[0]
		df['Product_Info_2_num'] = df.Product_Info_2.str[1]
		df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
		df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
		df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		return X,y,Id

	elif feature_choice == "knn" or feature_choice == "cosine":

		# Get data
		df = pd.read_csv("./Data/Raw/%s.csv" % file_name)
		if file_name == "test":
			df["Response"]=-1
		# Save then drop Id and y
		Id = df["Id"].values
		y = df["Response"].values
		df = df.drop(["Id", "Response"], 1)
		# Deal with columns with missing values
		df = df.fillna(-1)
		# Encode categorical		
		df['Product_Info_2_char'] = df.Product_Info_2.str[0]
		df['Product_Info_2_num'] = df.Product_Info_2.str[1]
		df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
		df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
		df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]
		
		df['BMI_Age'] = df['BMI'] * df['Ins_Age']
		med_keyword_columns = df.columns[df.columns.str.startswith('Medical_Keyword_')]
		df['Med_Keywords_Count'] = df[med_keyword_columns].sum(axis=1)
		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]
		# # Standardize
		X = StandardScaler().fit_transform(X)

		return X,y,Id

	elif feature_choice in ["linreg", "logistic", "keras_reg1"]:

		print "Preprocessing"
		# Get data
		df = pd.read_csv("./Data/Raw/%s.csv" % file_name)
		if file_name == "test":
			df["Response"]=-1
		# Get Id and response
		Id = df["Id"].values
		y = df["Response"].values
		# Drop Id and Response
		df = df.drop(["Id", "Response"], 1)
		# Deal with missing values
		print "Dealing with NaN"
		df["NULLCOUNT"] = df.isnull().sum(axis=1)
		df = df.fillna(df.median())
		#Get tsne data
		print "Getting tsne data"
		df_tsne_full = pd.read_csv("./Data/Raw/tsne_full_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_full", "V2_full"]] = df_tsne_full[["V1", "V2"]]
		df_tsne_binary = pd.read_csv("./Data/Raw/tsne_binary_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_binary", "V2_binary"]] = df_tsne_binary[["V1", "V2"]]
		df_tsne_ternary = pd.read_csv("./Data/Raw/tsne_ternary_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_ternary", "V2_ternary"]] = df_tsne_ternary[["V1", "V2"]]
		df_tsne_distance = pd.read_csv("./Data/Raw/tsne_distance_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_distance", "V2_distance"]] = df_tsne_distance[["V1", "V2"]]
		df_tsne_cosine = pd.read_csv("./Data/Raw/tsne_cosine_%s.csv" % file_name, usecols = ["V1", "V2"])
		df[["V1_cosine", "V2_cosine"]] = df_tsne_cosine[["V1", "V2"]]

		# Get correlation distance data
		print "Getting correlation data"
		df_distance = pd.read_csv("./Data/Raw/%s_distance_correlation.csv" % file_name)
		list_col_corr = [col for col in df_distance.columns.values if col != "Id" and col !="Response"]
		df[list_col_corr] = df_distance[list_col_corr]

		# Add custom features
		print "Feature engineering"
		df["SUMKEYWORD"] = np.zeros(len(df))
		df["SUMINSURED"] = np.zeros(len(df))
		for col in df.columns.values :
			if "Key" in col :
				df["SUMKEYWORD"]+=df[col]
			if "Insured" in col :
				df["SUMINSURED"]+=df[col]

		df["CINSINF"] = np.zeros(len(df))
		df["CINSINFMAX"] = np.zeros(len(df))
		for i in range(1,8):
			col           = "InsuredInfo_" + str(i)
			min_val       = df[col].value_counts().idxmin()
			max_val       = df[col].value_counts().idxmax()
			df["CINSINF"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
			df["CINSINFMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		df["CINSHIST"] = np.zeros(len(df))
		df["CINSHISTMAX"] = np.zeros(len(df))
		for i in range(1,10):
			if i !=6:
				col            = "Insurance_History_" + str(i)
				min_val        = df[col].value_counts().idxmin()
				max_val        = df[col].value_counts().idxmax()
				df["CINSHIST"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
				df["CINSHISTMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		df["CMEDKEY"] = np.zeros(len(df))
		df["CMEDKEYMAX"] = np.zeros(len(df))
		for i in range(1,49):
			col           = "Medical_Keyword_" + str(i)
			min_val       = df[col].value_counts().idxmin()
			max_val       = df[col].value_counts().idxmax()
			df["CMEDKEY"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
			df["CMEDKEYMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		df["CMEDHIST"] = np.zeros(len(df))
		df["CMEDHISTMAX"] = np.zeros(len(df))
		for i in range(1,42):
			if i not in [1,2,10,15,24]:
				col            = "Medical_History_" + str(i)
				min_val        = df[col].value_counts().idxmin()
				max_val        = df[col].value_counts().idxmax()
				df["CMEDHIST"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
				df["CMEDHISTMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		df["CPRODINFO"] = np.zeros(len(df))
		df["CPRODINFOMAX"] = np.zeros(len(df))
		for i in range(1,8):
			if i not in [2,4]:
				col             = "Product_Info_" + str(i)
				min_val         = df[col].value_counts().idxmin()
				max_val         = df[col].value_counts().idxmax()
				df["CPRODINFO"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
				df["CPRODINFOMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		df["CEMPINFO"] = np.zeros(len(df))
		df["CEMPINFOMAX"] = np.zeros(len(df))
		for i in range(2,6):
			col            = "Employment_Info_" + str(i)
			min_val        = df[col].value_counts().idxmin()
			max_val        = df[col].value_counts().idxmax()
			df["CEMPINFO"] += (df[col]==min_val).apply(lambda x :  1 if x else 0)
			df["CEMPINFOMAX"] += (df[col]==max_val).apply(lambda x :  1 if x else 0)

		print "Comparison features"
		df["COMP_IH4_IH7"] = df["Insurance_History_4"].values == df["Insurance_History_7"].values
		df["COMP_IH4_IH3"] = np.abs(df["Insurance_History_4"].values - df["Insurance_History_3"].values)
		df["COMP_IH9_IH7"] = np.abs(df["Insurance_History_9"].values - df["Insurance_History_7"].values)
		df["COMP_MH6_MK48"] = np.abs(df["Medical_History_6"].values - df["Medical_Keyword_48"].values)
		df["COMP_MH33_MK23"] = np.abs(df["Medical_History_33"].values - df["Medical_Keyword_23"].values)
		df["COMP_MH37_MK11"] = np.abs(df["Medical_History_37"].values - df["Medical_Keyword_11"].values)
		df["COMP_MH25_MH26"] = np.abs(df["Medical_History_25"].values - df["Medical_History_26"].values)
		
		# factorize categorical variables
		df['Product_Info_2_char'] = df.Product_Info_2.str[0]
		df['Product_Info_2_num'] = df.Product_Info_2.str[1]
		df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
		df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
		df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]

		# Custom variables
		print "Kaggle features"
		df['custom_var_1'] = df['Medical_History_15'] < 10
		df['custom_var_3'] = df['Product_Info_4'] < 0.075
		df['custom_var_4'] = df['Product_Info_4'] == 1
		df['custom_var_6'] = (df['BMI'] + 1)**2
		df['custom_var_7'] = df['BMI']**0.8
		df['custom_var_8'] = df['Ins_Age']**8.5
		df['BMI_Age'] = (df['BMI'] * df['Ins_Age'])**2.5
		df['custom_var_10'] = df['BMI'] > np.percentile(df['BMI'], 0.8)
		df['custom_var_11'] = (df['BMI'] * df['Product_Info_4'])**0.9
		age_BMI_cutoff = np.percentile(df['BMI'] * df['Ins_Age'], 0.9)
		df['custom_var_12'] = (df['BMI'] * df['Ins_Age']) > age_BMI_cutoff
		df['custom_var_13'] = (df['BMI'] * df['Medical_Keyword_3'] + 0.5)**3

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		print "Standardizing"
		X = StandardScaler().fit_transform(X)

		return X,y,Id

	elif feature_choice == "xgb_reg":

		print "Preprocessing"
		# Get data
		df = pd.read_csv("./Data/Raw/%s.csv" % file_name)
		if file_name == "test":
			df["Response"]=-1
		# Get Id and response
		Id = df["Id"].values
		y = df["Response"].values
		# Drop Id and Response
		df = df.drop(["Id", "Response"], 1)
		# Deal with missing values
		print "Dealing with NaN"
		df["NULLCOUNT"] = df.isnull().sum(axis=1)
		#Get tsne data
		
		# factorize categorical variables
		df['Product_Info_2_char'] = df.Product_Info_2.str[0]
		df['Product_Info_2_num'] = df.Product_Info_2.str[1]
		df['Product_Info_2'] = pd.factorize(df['Product_Info_2'])[0]
		df['Product_Info_2_char'] = pd.factorize(df['Product_Info_2_char'])[0]
		df['Product_Info_2_num'] = pd.factorize(df['Product_Info_2_num'])[0]

		# Custom variables
		print "Kaggle features"
		df['custom_var_1'] = df['Medical_History_15'] < 10
		df['custom_var_3'] = df['Product_Info_4'] < 0.075
		df['custom_var_4'] = df['Product_Info_4'] == 1
		df['custom_var_6'] = (df['BMI'] + 1)**2
		df['custom_var_7'] = df['BMI']**0.8
		df['custom_var_8'] = df['Ins_Age']**8.5
		df['BMI_Age'] = (df['BMI'] * df['Ins_Age'])**2.5
		df['custom_var_10'] = df['BMI'] > np.percentile(df['BMI'], 0.8)
		df['custom_var_11'] = (df['BMI'] * df['Product_Info_4'])**0.9
		age_BMI_cutoff = np.percentile(df['BMI'] * df['Ins_Age'], 0.9)
		df['custom_var_12'] = (df['BMI'] * df['Ins_Age']) > age_BMI_cutoff
		df['custom_var_13'] = (df['BMI'] * df['Medical_Keyword_3'] + 0.5)**3

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		return X,y,Id



def combine_files(file_name):
	"""
	Combine all LV1 features into a single file 

	args: file_name (str) train or test (preprocess train or test file) 
	"""

	if file_name == "train":

		if os.path.isfile("./Data/Level1_model_files/Train/all_level1_train.csv"):
			os.remove("./Data/Level1_model_files/Train/all_level1_train.csv")

		list_files = glob("./Data/Level1_model_files/Train/*.csv*")
		list_df = []
		for f in list_files :
			list_df.append(pd.read_csv(f))

		for i in range(1,len(list_df)):
			list_df[i] = list_df[i].drop(["Response", "Id"],1)

		# Concat
		df_out = pd.concat(list_df, axis=1)
		# Order columns
		list_col = df_out.columns.values.tolist()
		list_col = sorted(list_col)
		list_col.remove("Response")
		list_col.remove("Id")
		list_col = ["Id"] + list_col + ["Response"]
		df_out = df_out[list_col]
		df_out.to_csv("./Data/Level1_model_files/Train/all_level1_train.csv", index = False)

	elif file_name == "test":

		if os.path.isfile("./Data/Level1_model_files/Test/all_level1_test.csv"):
			os.remove("./Data/Level1_model_files/Test/all_level1_test.csv")

		list_files = glob("./Data/Level1_model_files/Test/*.csv*")
		list_df = []
		for f in list_files :
			list_df.append(pd.read_csv(f))

		for i in range(1,len(list_df)):
			list_df[i] = list_df[i].drop("Id",1)

		# Concat
		df_out = pd.concat(list_df, axis=1)
		# Order columns
		list_col = df_out.columns.values.tolist()
		list_col = sorted(list_col)
		list_col.remove("Id")
		list_col = ["Id"] + list_col 
		df_out = df_out[list_col]
		df_out.to_csv("./Data/Level1_model_files/Test/all_level1_test.csv", index = False)

if __name__ == "__main__":

	compute_distance_feat()

	## Uncomment once level1 files are created
	# combine_files("test")	
	# combine_files("train")
	pass