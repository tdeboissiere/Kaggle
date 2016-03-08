import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_lv2_data(feature_choice, file_name):
	"""
	Preprocess the data (feature engineering) to train lv2 models

	args : feature_choice (str) specify the type of feature engineering
		   file_name (str) train or test : preprocess train or test file
	"""
	if feature_choice =="lv1":

		print "Preprocessing"
		#Initialize dataframe
		df = None
		# Get data
		if file_name == "train":
			pass
			df = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
		elif file_name == "test":
			df = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")
			df["Response"] = -1 # Use this to make the script adapt to both train/test

		# Get Id and response
		Id = df["Id"].values
		y = df["Response"].values
		# Drop Id and Response
		df = df.drop(["Id", "Response"], 1)

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		print "Standardizing"
		X = StandardScaler().fit_transform(X)

		return X,y,Id

	if feature_choice =="lv1bayes":

		print "Preprocessing"
		#Initialize dataframe
		df = None
		# Get data
		if file_name == "train":
			df = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
			df_bayes = pd.read_csv("./Data/Level1_model_files/Train/bayesian_train.csv")
			df_bayes = df_bayes.drop(["Response", "Id"], 1)
			list_col = df_bayes.columns.values.tolist()
			for col in list_col :
				df[col] =  df_bayes[col].values 
		elif file_name == "test":
			df = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")
			df["Response"] = -1 # Use this to make the script adapt to both train/test
			df_bayes = pd.read_csv("./Data/Level1_model_files/Test/bayesian_test.csv")
			df_bayes = df_bayes.drop("Id", 1)
			list_col = df_bayes.columns.values.tolist()
			for col in list_col :
				df[col] =  df_bayes[col].values 

		# Compute Bayes likelihood
		for i in range(1,9):
			list_bayes = [c for c in df.columns.values if "Full" in c and "_" + str(i) in c]
			df["Prob_%s"% i] = np.log(df[list_bayes[0]]+1E-8) # avoid NaN
			for c in list_bayes[1:]:
				df["Prob_%s"% i]+= np.log(df[c]+1E-8) # avoid NaN

		df["BayesPred"] = df[["Prob_%s" % i for i in range(1,9)]].idxmax(axis=1)
		df["BayesPred"] = df["BayesPred"].apply(lambda x : int(x[-1]))

		# Get Id and response
		Id = df["Id"].values
		y = df["Response"].values
		# Drop Id and Response
		df = df.drop(["Id", "Response"], 1)

		# Clip predictions because of some weird value
		df["linreg"] = np.clip(df["linreg"].values,-3,10)

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		print "Standardizing"
		X = StandardScaler().fit_transform(X)

		return X,y,Id

	elif feature_choice == "mix":

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
		df["NULL"] = df.isnull().sum(axis=1)
		df = df.fillna(-1)
		#Get tsne data
		print "Getting tsne data"
		df_tsne_full = pd.read_csv("./Data/Raw/tsne_full_train.csv", usecols = ["V1", "V2"])
		df[["V1_full", "V2_full"]] = df_tsne_full[["V1", "V2"]]
		df_tsne_binary = pd.read_csv("./Data/Raw/tsne_binary_train.csv", usecols = ["V1", "V2"])
		df[["V1_binary", "V2_binary"]] = df_tsne_binary[["V1", "V2"]]
		df_tsne_ternary = pd.read_csv("./Data/Raw/tsne_ternary_train.csv", usecols = ["V1", "V2"])
		df[["V1_ternary", "V2_ternary"]] = df_tsne_ternary[["V1", "V2"]]
		df_tsne_distance = pd.read_csv("./Data/Raw/tsne_distance_train.csv", usecols = ["V1", "V2"])
		df[["V1_distance", "V2_distance"]] = df_tsne_distance[["V1", "V2"]]
		df_tsne_cosine = pd.read_csv("./Data/Raw/tsne_cosine_train.csv", usecols = ["V1", "V2"])
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

		# Get level1 data
		print "Get level1 data"
		if file_name == "train":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
			list_col = [column for column in df_lv1.columns.values if column != "Id" and column !="Response"]
			df[list_col] = df_lv1[list_col]
		elif file_name == "test":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")
			list_col = [column for column in df_lv1.columns.values if column != "Id" and column !="Response"]
			df[list_col] = df_lv1[list_col]

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		print "Standardizing"
		X = StandardScaler().fit_transform(X)

		return X,y,Id

	elif feature_choice == "mixbayes":

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
		df["NULL"] = df.isnull().sum(axis=1)
		df = df.fillna(-1)
		#Get tsne data
		print "Getting tsne data"
		df_tsne_full = pd.read_csv("./Data/Raw/tsne_full_train.csv", usecols = ["V1", "V2"])
		df[["V1_full", "V2_full"]] = df_tsne_full[["V1", "V2"]]
		df_tsne_binary = pd.read_csv("./Data/Raw/tsne_binary_train.csv", usecols = ["V1", "V2"])
		df[["V1_binary", "V2_binary"]] = df_tsne_binary[["V1", "V2"]]
		df_tsne_ternary = pd.read_csv("./Data/Raw/tsne_ternary_train.csv", usecols = ["V1", "V2"])
		df[["V1_ternary", "V2_ternary"]] = df_tsne_ternary[["V1", "V2"]]
		df_tsne_distance = pd.read_csv("./Data/Raw/tsne_distance_train.csv", usecols = ["V1", "V2"])
		df[["V1_distance", "V2_distance"]] = df_tsne_distance[["V1", "V2"]]
		df_tsne_cosine = pd.read_csv("./Data/Raw/tsne_cosine_train.csv", usecols = ["V1", "V2"])
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

		# Get level1 data
		print "Get level1 data"
		if file_name == "train":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
			list_col = [column for column in df_lv1.columns.values if column != "Id" and column !="Response"]
			df[list_col] = df_lv1[list_col]
		elif file_name == "test":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")
			list_col = [column for column in df_lv1.columns.values if column != "Id" and column !="Response"]
			df[list_col] = df_lv1[list_col]

		if file_name == "train":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Train/all_level1_train.csv")
			df_lv1 = df_lv1.drop(["Response", "Id"],1)
			df_bayes = pd.read_csv("./Data/Level1_model_files/Train/bayesian_train.csv")
			df_bayes = df_bayes.drop(["Response", "Id"], 1)

			for col in df_lv1.columns.values :
				df[col] =  df_lv1[col].values 
			for col in df_bayes.columns.values :
				df[col] =  df_bayes[col].values 

		elif file_name == "test":
			df_lv1 = pd.read_csv("./Data/Level1_model_files/Test/all_level1_test.csv")
			df_lv1 = df_lv1.drop("Id",1)
			df_bayes = pd.read_csv("./Data/Level1_model_files/Test/bayesian_test.csv")
			df_bayes = df_bayes.drop("Id", 1)

			for col in df_lv1.columns.values :
				df[col] =  df_lv1[col].values 
			for col in df_bayes.columns.values :
				df[col] =  df_bayes[col].values 

		# Shuffle data
		permut = np.random.choice(len(df), len(df), replace = False)
		df = df.iloc[permut,:]
		X = df.values
		y = y[permut]
		Id = Id[permut]

		return X,y,Id


if __name__ == "__main__":
	pass