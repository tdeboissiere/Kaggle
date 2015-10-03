import numpy as np
import pandas as pd
import cPickle as pickle
from datetime import date
import calendar
import os

def create_validation_set(week_test_start, week_test_end, week_ID):
	"""Create Validation dataset (up to 5 weeks prior to the end)
	args : Validation week start and end date, Validation week ID
	"""

	print "Creating validation set for week: %s" % week_ID

	Validation_path = "../Data/Validation/" + week_ID + "/"
	if not os.path.exists(Validation_path) :
		os.makedirs(Validation_path)

	Validation_start = date(int(week_test_start[0]), int(week_test_start[1]), int(week_test_start[2]))
	Validation_end = date(int(week_test_end[0]), int(week_test_end[1]), int(week_test_end[2]))

	Validation_start_stamp = calendar.timegm(Validation_start.timetuple())
	Validation_end_stamp = calendar.timegm(Validation_end.timetuple())

	# # Load data frames
	user_df = pd.read_csv("../Data/Data_translated/user_list_translated.csv")
	train_list_df = pd.read_csv("../Data/Data_translated/coupon_list_train_translated.csv")
	train_visit_df = pd.read_csv("../Data/Data_translated/coupon_visit_train_translated.csv")
	train_detail_df = pd.read_csv("../Data/Data_translated/coupon_detail_train_translated.csv")

	def get_unix_time(row):
		"""Convert to unix time. Neglect time of the day
		"""
		row = row.split(" ")
		row = row[0].split("-")
		y,m,d = int(row[0]), int(row[1]), int(row[2])
		return calendar.timegm(date(y,m,d).timetuple())

	user_df["REG_DATE_UNIX"] = user_df["REG_DATE"].apply(get_unix_time)
	train_list_df["DISPFROM_UNIX"] =train_list_df["DISPFROM"].apply(get_unix_time)
	train_visit_df["I_DATE_UNIX"] =train_visit_df["I_DATE"].apply(get_unix_time)
	train_detail_df["I_DATE_UNIX"] =train_detail_df["I_DATE"].apply(get_unix_time)

	#Get list of users Validation dataset
	#Only save users who registered before test date
	cond = (user_df["REG_DATE_UNIX"] < Validation_start_stamp)
	user_Validation = user_df[cond]
	list_user_Validation = set(user_Validation["USER_ID_hash"].values)
	user_Validation.to_csv(Validation_path + "user_list_validation_" + week_ID + ".csv", index = False)

	# Get list of coupon test Validation dataset
	cond = (train_list_df["DISPFROM_UNIX"] >= Validation_start_stamp) & (train_list_df["DISPFROM_UNIX"] <= Validation_end_stamp)
	test_list_Validation = train_list_df[cond]
	test_list_Validation.to_csv(Validation_path + "coupon_list_test_validation_" + week_ID + ".csv", index = False)

	# Get list of coupon train Validation dataset
	cond = (train_list_df["DISPFROM_UNIX"] < Validation_start_stamp) 
	train_list_Validation = train_list_df[cond]
	train_list_Validation.to_csv(Validation_path + "coupon_list_train_validation_" + week_ID + ".csv", index = False)

	# Get visit of coupon train Validation dataset
	cond = (train_visit_df["I_DATE_UNIX"] < Validation_start_stamp) 
	train_visit_Validation = train_visit_df[cond]
	#Remove users who have viewed but are not registered (not in user_list)
	list_user_Validation_visit = set(train_visit_Validation["USER_ID_hash"].values)
	list_visit_bad = [el for el in list_user_Validation_visit if el not in list_user_Validation]
	filt = np.in1d(train_visit_Validation["USER_ID_hash"].values, list_visit_bad, invert = True)
	train_visit_Validation = train_visit_Validation[filt]
	train_visit_Validation.to_csv(Validation_path + "coupon_visit_train_validation_" + week_ID + ".csv", index = False)

	# Get detail of coupon train Validation dataset
	cond = (train_detail_df["I_DATE_UNIX"] < Validation_start_stamp) 
	train_detail_Validation = train_detail_df[cond]
	train_detail_Validation.to_csv(Validation_path + "coupon_detail_train_validation_" + week_ID + ".csv", index = False)

	#Get dict of actual purchases during test week (key : USER, value = list of purchase)
	cond = (train_detail_df["I_DATE_UNIX"] >= Validation_start_stamp) & (train_detail_df["I_DATE_UNIX"] <= Validation_end_stamp)
	train_purchase_Validation = train_detail_df[cond][["USER_ID_hash", "COUPON_ID_hash"]]
	train_purchase_Validation.to_csv(Validation_path + "coupon_purchase_test_validation_" + week_ID + ".csv", index = False)

	d_user_purchase = {}
	#Initialise
	for user in list_user_Validation :
		d_user_purchase[user] = []
	#Fill
	for x in range(len(train_purchase_Validation)):
		user = train_purchase_Validation["USER_ID_hash"].iloc[x]
		coupon_id = train_purchase_Validation["COUPON_ID_hash"].iloc[x]
		try :
			if coupon_id in test_list_Validation["COUPON_ID_hash"].values:
				d_user_purchase[user].append(coupon_id)
		except KeyError :
			pass
	#Save to pickle
	with open("../Data/Validation/" + week_ID + "/dict_purchase_validation_" + week_ID + ".pickle", "w") as fp:
			pickle.dump(d_user_purchase, fp)

	#Get dict with list of users for various categories
	d_user_list = {}
	d_user_list["all_user"] = set(user_Validation["USER_ID_hash"].values)
	d_user_list["detail_user"] = set(train_detail_Validation["USER_ID_hash"].values)
	d_user_list["view_user"] = set(train_visit_Validation["USER_ID_hash"].values)
	d_user_list["view_no_detail_user"] = [us for us in d_user_list["view_user"] if us not in d_user_list["detail_user"] ]
	d_user_list["no_view_no_detail_user"] = [us for us in d_user_list["all_user"] if us not in d_user_list["detail_user"] and us not in d_user_list["view_user"] ]
	#Save to pickle
	with open("../Data/Validation/" + week_ID + "/dict_user_list_validation_" + week_ID + ".pickle", "w") as fp:
			pickle.dump(d_user_list, fp)


if __name__ == "__main__":

	create_validation_set([2012,06,17], [2012, 06, 23], "week52")
	create_validation_set([2012,06,10], [2012, 06, 16], "week51")

	#Create more validation sets if needed
	# create_validation_set([2012,06,3], [2012, 06, 9], "week50")
	# create_validation_set([2012,05,27], [2012, 06, 2], "week49")
	# create_validation_set([2012,05,20], [2012, 05, 26], "week48")
