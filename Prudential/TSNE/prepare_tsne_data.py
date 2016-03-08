import pandas as pd 
import numpy as np 
import os

def prepare_tsne_full_data():
	"""
	Preprocessing steps to run t-SNE
	on the full data set
	"""

	# global variables
	columns_to_drop = ['Id', 'Response']
	
	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train.csv")
	test = pd.read_csv("../Data/Raw/test.csv")
	
	train = train.drop(columns_to_drop, 1)
	test = test.drop("Id", 1)
	
	# combine train and test
	all_data = train.append(test)
	
	# create any new variables    
	all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
	all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
	
	# factorize categorical variables
	all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
	all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
	all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
	
	all_data["NULL"] = all_data.isnull().sum(axis=1)
	all_data["SUMKEYWORD"] = np.zeros(len(all_data))
	all_data["SUMINSURED"] = np.zeros(len(all_data))

	for col in all_data.columns.values :
		if "Key" in col :
			all_data["SUMKEYWORD"] +=all_data[col]
		if "Insured" in col :
			all_data["SUMINSURED"] +=all_data[col]

	all_data["CINSINF"] = np.zeros(len(all_data))
	all_data["CINSINFMAX"] = np.zeros(len(all_data))
	for i in range(1,8):
		col                    = "InsuredInfo_" + str(i)
		min_val                = all_data[col].value_counts().idxmin()
		max_val                = all_data[col].value_counts().idxmax()
		all_data["CINSINF"]    += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
		all_data["CINSINFMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)

	all_data["CINSHIST"] = np.zeros(len(all_data))
	all_data["CINSHISTMAX"] = np.zeros(len(all_data))
	for i in range(1,10):
		if i !=6:
			col                     = "Insurance_History_" + str(i)
			min_val                 = all_data[col].value_counts().idxmin()
			max_val                 = all_data[col].value_counts().idxmax()
			all_data["CINSHIST"]    += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
			all_data["CINSHISTMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)

	all_data["CMEDKEY"]    = np.zeros(len(all_data))
	all_data["CMEDKEYMAX"] = np.zeros(len(all_data))
	for i in range(1,49):
		col                    = "Medical_Keyword_" + str(i)
		min_val                = all_data[col].value_counts().idxmin()
		max_val                = all_data[col].value_counts().idxmax()
		all_data["CMEDKEY"]    += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
		all_data["CMEDKEYMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)

	all_data["CMEDHIST"]    = np.zeros(len(all_data))
	all_data["CMEDHISTMAX"] = np.zeros(len(all_data))
	for i in range(1,42):
		if i not in [1,2,10,15,24]:
			col                     = "Medical_History_" + str(i)
			min_val                 = all_data[col].value_counts().idxmin()
			max_val                 = all_data[col].value_counts().idxmax()
			all_data["CMEDHIST"]    += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
			all_data["CMEDHISTMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)

	all_data["CPRODINFO"] = np.zeros(len(all_data))
	all_data["CPRODINFOMAX"] = np.zeros(len(all_data))
	for i in range(1,8):
		if i not in [2,4]:
			col = "Product_Info_" + str(i)
			min_val = all_data[col].value_counts().idxmin()
			max_val = all_data[col].value_counts().idxmax()
			all_data["CPRODINFO"] += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
			all_data["CPRODINFOMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)
	
	all_data["CEMPINFO"] = np.zeros(len(all_data))
	all_data["CEMPINFOMAX"] = np.zeros(len(all_data))
	for i in range(2,6):
		col                     = "Employment_Info_" + str(i)
		min_val                 = all_data[col].value_counts().idxmin()
		max_val                 = all_data[col].value_counts().idxmax()
		all_data["CEMPINFO"]    += (all_data[col]==min_val).apply(lambda x :  1 if x else 0)
		all_data["CEMPINFOMAX"] += (all_data[col]==max_val).apply(lambda x :  1 if x else 0)
	
	all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

	all_data = all_data.fillna(all_data.median())

	list_col = all_data.columns.values.tolist()
	# Find which rows are duplicate
	all_data['is_duplicated'] = all_data.duplicated(list_col)

	# For duplicate rows, find the index of the original row
	g = all_data.groupby(list_col)
	df1 = all_data.set_index(list_col)
	df1.index.map(lambda ind: g.indices[ind][0])
	all_data['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])

	data_for_tsne = all_data[all_data["is_duplicated"]==False][list_col]

	# Save the data
	data_for_tsne.to_csv("./full/train_test_full_for_tsne.csv", index = False, float_format = "%.5f")
	all_data[["is_duplicated", "dup_index"]].to_csv("./full/train_test_full_dup_info.csv", index = False)


def prepare_tsne_distance_data():
	"""
	Preprocessing steps to run t-SNE
	on the correlation distance data set
	"""

	# global variables
	columns_to_drop = ['Id', 'Response']
	
	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train_distance_correlation.csv")
	test = pd.read_csv("../Data/Raw/test_distance_correlation.csv")
	
	train = train.drop(columns_to_drop, 1)
	test = test.drop("Id", 1)
	
	# combine train and test
	all_data = train.append(test)

	list_col = all_data.columns.values.tolist()
	# Find which rows are duplicate
	all_data['is_duplicated'] = all_data.duplicated(list_col)

	# For duplicate rows, find the index of the original row
	g = all_data.groupby(list_col)
	df1 = all_data.set_index(list_col)
	df1.index.map(lambda ind: g.indices[ind][0])
	all_data['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])

	data_for_tsne = all_data[all_data["is_duplicated"]==False][list_col]

	# Set seed for reproducibility
	data_for_tsne.to_csv("./distance/train_test_distance_for_tsne.csv", index = False, float_format = "%.5f")
	all_data[["is_duplicated", "dup_index"]].to_csv("./distance/train_test_distance_dup_info.csv", index = False)

def prepare_tsne_cosine_data():
	"""
	Preprocessing steps to run t-SNE
	on the cosine distance data set
	"""

	# global variables
	columns_to_drop = ['Id', 'Response']
	
	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train_cosine_sim.csv")
	test = pd.read_csv("../Data/Raw/test_cosine_sim.csv")
	
	train = train.drop(columns_to_drop, 1)
	test = test.drop("Id", 1)
	
	# combine train and test
	all_data = train.append(test)

	list_col = all_data.columns.values.tolist()
	# Find which rows are duplicate
	all_data['is_duplicated'] = all_data.duplicated(list_col)

	# For duplicate rows, find the index of the original row
	g = all_data.groupby(list_col)
	df1 = all_data.set_index(list_col)
	df1.index.map(lambda ind: g.indices[ind][0])
	all_data['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])

	data_for_tsne = all_data[all_data["is_duplicated"]==False][list_col]

	# Set seed for reproducibility
	data_for_tsne.to_csv("./cosine/train_test_cosine_for_tsne.csv", index = False, float_format = "%.5f")
	all_data[["is_duplicated", "dup_index"]].to_csv("./cosine/train_test_cosine_dup_info.csv", index = False)


def prepare_tsne_binary_data():
	"""
	Preprocessing steps to run t-SNE
	on all columns that have only 2 unique values
	"""

	# global variables
	columns_to_drop = ['Id', 'Response']
	
	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train.csv")
	test = pd.read_csv("../Data/Raw/test.csv")
	
	train = train.drop(columns_to_drop, 1)
	test = test.drop("Id", 1)
	
	# combine train and test
	all_data = train.append(test)
	all_data.fillna(-1)

	# Get list of binary columns 
	list_bin = []
	for col in all_data.columns.values :
		if len(all_data[col].unique()) ==2 :
			# print col, all_data[col].unique()
			list_bin.append(col)

	all_data = all_data[list_bin].copy()
	# Find which rows are duplicate
	all_data['is_duplicated'] = all_data.duplicated(list_bin)

	# For duplicate rows, find the index of the original row
	g = all_data.groupby(list_bin)
	df1 = all_data.set_index(list_bin)
	df1.index.map(lambda ind: g.indices[ind][0])
	all_data['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])

	data_for_tsne = all_data[all_data["is_duplicated"]==False][list_bin]

	# Set seed for reproducibility
	data_for_tsne.to_csv("./binary/train_test_binary_for_tsne.csv", index = False, float_format = "%.5f")
	all_data[["is_duplicated", "dup_index"]].to_csv("./binary/train_test_binary_dup_info.csv", index = False)

def prepare_tsne_ternary_data():
	"""
	Preprocessing steps to run t-SNE
	on all columns that have only 3 unique values
	"""

	# global variables
	columns_to_drop = ['Id', 'Response']
	
	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train.csv")
	test = pd.read_csv("../Data/Raw/test.csv")
	
	train = train.drop(columns_to_drop, 1)
	test = test.drop("Id", 1)
	
	# combine train and test
	all_data = train.append(test)
	all_data.fillna(-1)

	# Get list of ternary columns 
	list_tern = []
	for col in all_data.columns.values :
		if len(all_data[col].unique()) ==3 :
			# print col, all_data[col].unique()
			list_tern.append(col)

	all_data = all_data[list_tern].copy()
	# Find which rows are duplicate
	all_data['is_duplicated'] = all_data.duplicated(list_tern)

	# For duplicate rows, find the index of the original row
	g = all_data.groupby(list_tern)
	df1 = all_data.set_index(list_tern)
	df1.index.map(lambda ind: g.indices[ind][0])
	all_data['dup_index'] = df1.index.map(lambda ind: g.indices[ind][0])

	data_for_tsne = all_data[all_data["is_duplicated"]==False][list_tern]

	# Set seed for reproducibility
	data_for_tsne.to_csv("./ternary/train_test_ternary_for_tsne.csv", index = False, float_format = "%.5f")
	all_data[["is_duplicated", "dup_index"]].to_csv("./ternary/train_test_ternary_dup_info.csv", index = False)

def format_tsne_data(tsne_type):
	"""
	Once t-SNE has been ran on the data with the R script,
	format the data to save it to an exploitable .csv file
	"""

	print("Load the data using pandas")
	train = pd.read_csv("../Data/Raw/train.csv", usecols = ["Id", "Response"])
	test = pd.read_csv("../Data/Raw/test.csv", usecols = ["Id"])

	test["Response"] = -1
	all_data = train.append(test)
	# Rest indices
	all_data = all_data.reset_index()

	# Deal with tsne on binary data
	tsne = pd.read_csv("./%s/tsne_var_%s_train_test.csv" % (tsne_type, tsne_type), usecols = ["V1", "V2"])
	dup_info = pd.read_csv("./%s/train_test_%s_dup_info.csv" % (tsne_type, tsne_type))

	# Initialise V1, V2 and add Id info
	dup_info["Id"] = all_data["Id"]
	dup_info["Response"] = all_data["Response"]
	dup_info["V1"] = np.zeros(len(dup_info))
	dup_info["V2"] = np.zeros(len(dup_info))

	# Fill the non duplicate of V1 and V2

	dup_info.loc[dup_info["is_duplicated"] ==False, "V1"] = tsne["V1"].values
	dup_info.loc[dup_info["is_duplicated"] ==False, "V2"] = tsne["V2"].values

	# # Fill the duplicate
	arr_dup_index = dup_info[dup_info["is_duplicated"]==True]["dup_index"].values
	dup_info.loc[dup_info["is_duplicated"] ==True, "V1" ] = dup_info["V1"].values[arr_dup_index]
	dup_info.loc[dup_info["is_duplicated"] ==True, "V2" ] = dup_info["V2"].values[arr_dup_index]

	tsne_train = dup_info.iloc[:len(train),:].copy()
	tsne_test = dup_info.iloc[len(train):,:].copy()

	tsne_train = tsne_train[["Id", "V1", "V2", "Response"]]
	tsne_test = tsne_test[["Id", "V1", "V2"]]

	tsne_train.to_csv("../Data/Raw/tsne_%s_train.csv" % (tsne_type), index=False)
	tsne_test.to_csv("../Data/Raw/tsne_%s_test.csv" % (tsne_type), index=False)

if __name__ == '__main__':

	for direc in ["./full/", "./binary/", "./ternary/", "./cosine/", "./distance/"]:
		if not os.path.exists(direc):
			os.makedirs(direc)


	# Preprocessing steps
	prepare_tsne_full_data()
	prepare_tsne_distance_data()
	prepare_tsne_cosine_data()
	prepare_tsne_binary_data()
	prepare_tsne_ternary_data()
	
	# ## Run this code after the Rscript
	# for t in ["full", "cosine", "distance", "binary", "ternary"]:
	# 	format_tsne_data(t)
