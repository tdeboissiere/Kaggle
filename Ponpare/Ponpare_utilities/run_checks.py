import numpy as np
import pandas as pd

#This script to run checks on the data

def check_user_visit():
	"""Check whether all user in user list have at least visited
	"""
	#Use this
	# df to get the full list of users
	user_df = pd.read_csv("../Data/Data_translated/user_list_translated.csv")
	user_list = sorted(user_df["USER_ID_hash"].values.tolist())

	#Use this df to get the visit history of users
	train_visit_df = pd.read_csv("../Data/Data_translated/coupon_visit_train_translated.csv")
	visit_list = sorted(set(train_visit_df["USER_ID_hash"].values.tolist()))

	assert(user_list == visit_list)


def check_test_set() :
	""" Check when coupons in test set have been emitted
	Answer : only during test set week
	"""
	#Use this df to get the list of test coupons
	test_coupon_df = pd.read_csv("../Data/Data_translated/coupon_list_test_translated.csv")
	list_dates = test_coupon_df["DISPFROM"].values

	lm, ld = [], []

	for el in list_dates :
		el = el.split(" ")
		el = el[0].split("-")
		lm.append(el[1])	
		ld.append(el[2])

	for el in lm :
		if el != "06":
			print el 

	for el in ld :
		print el
		if int(el) >30 or int(el) <24 :
			print el

def check_user_visit_detail():
	"""Check that all users who have bought have visited
	"""
	
	#Get df with full list of user visit
	train_visit_df = pd.read_csv("../Data/Data_translated/coupon_visit_train_translated.csv")
	visit_list = sorted(set(train_visit_df["USER_ID_hash"].values.tolist()))

	#Get df with full list of purchase
	train_detail_df = pd.read_csv("../Data/Data_translated/coupon_detail_train_translated.csv")
	detail_list = sorted(set(train_detail_df["USER_ID_hash"].values.tolist()))

	c1 = 0
	for el in detail_list :
		if el not in visit_list :
			c1+=1
			print el
	print "%s users have bought without visiting" % c1

def check_coupon_from_view_detail() :
	""" Check that all coupons viewed/detailed are in the coupon train list
	FALSE => there are viewed coupons not in the coupon train list.
	"""

	train_visit_df = pd.read_csv("../Data/Data_translated/coupon_visit_train_translated.csv")
	visit_list = sorted(set(train_visit_df["VIEW_COUPON_ID_hash"].values))
	print len(visit_list)

	train_detail_df = pd.read_csv("../Data/Data_translated/coupon_detail_train_translated.csv")
	detail_list = sorted(set(train_detail_df["COUPON_ID_hash"].values))

	coupon_list_train = pd.read_csv("../Ponpare/Data_translated/coupon_list_train_translated.csv")
	coupon_list = sorted(set(coupon_list_train["COUPON_ID_hash"].values))
	print len(coupon_list)

	#VIEW / LIST COUPON
	list_view_no_list = []
	for el in visit_list :
		if el not in coupon_list :
			list_view_no_list.append(el)

	list_list_no_view = []
	for el in coupon_list :
		if el not in visit_list :
			list_list_no_view.append(el)

	print len(list_view_no_list)
	print len(list_list_no_view)

	#DETAIL / LIST COUPON
	list_detail_no_list = []
	for el in detail_list :
		if el not in coupon_list :
			list_detail_no_list.append(el)

	list_list_no_detail = []
	for el in coupon_list :
		if el not in detail_list :
			list_list_no_detail.append(el)

	print len(list_detail_no_list)
	print len(list_list_no_detail)


if __name__ == "__main__":

	# check_user_visit()
	# check_user_visit_detail()
	# check_coupon_from_view_detail()