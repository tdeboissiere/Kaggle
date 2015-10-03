import pandas as pd
import sys 
reload(sys) 
sys.setdefaultencoding('utf8')

def encode_text(data):
	# This function converts the data into type "str".
    try:
        text = encode_text(data)
    except:
        text = data.encode("UTF-8")
    return text

def translate():
	""" Translate data files from Japanese to English
	"""

	#Dict for prefecture translation
	d_pref = {}
	with open("./prefecture.txt", "r") as f:
		stuff = f.readlines()
		for line in stuff:
			line = line.rstrip().split(",")
			d_pref[encode_text(line[0])] = line[1]

	# Path to files downloaded from Kaggle
	path = "../Data/Data_japanese/" 
	df = pd.read_excel(path + "/documentation/documentation/CAPSULE_TEXT_Translation.xlsx",skiprows=5)
	
	# Dict for feature "CAPSULE_TEXT" translation.
	k = [ encode_text(x) for x in df["CAPSULE_TEXT"] ] 
	v = [ encode_text(x) for x in df["English Translation"] ]
	capsuleText = dict(zip(k,v))
	
	# Dict for feature "GENRE_NAME" translation.
	k = df["CAPSULE_TEXT.1"].dropna()
	v = df["English Translation.1"].dropna()

	k = [ encode_text(x) for x in k ]
	v = [ encode_text(x) for x in v ]
	genreName = dict(zip(k,v))
	
	#translating the columns from japanese to english.
	files = ["coupon_list_train","coupon_list_test"]
	files +=["coupon_detail_train", "coupon_visit_train"]
	files += ["user_list"]
	
	for f in files:
		print "Processing file: %s" % f
		df = pd.read_csv(path + f + ".csv")
		if "CAPSULE_TEXT" in df.columns.values :
			df["CAPSULE_TEXT"] = [ capsuleText[encode_text(x)] for x in df["CAPSULE_TEXT"] ]
		if "GENRE_NAME" in df.columns.values :
			df["GENRE_NAME"] = [ genreName[encode_text(x)] for x in df["GENRE_NAME"] ]
		if "large_area_name" in df.columns.values :
			df["large_area_name"] = [d_pref[encode_text(x)] for x in df["large_area_name"]]
		if "ken_name" in df.columns.values :
			df["ken_name"] = [d_pref[encode_text(x)] for x in df["ken_name"]]
		if "small_area_name" in df.columns.values :
			df["small_area_name"] = [d_pref[encode_text(x)] for x in df["small_area_name"]]
		if "SMALL_AREA_NAME" in df.columns.values :
			df["SMALL_AREA_NAME"] = [d_pref[encode_text(x)] for x in df["SMALL_AREA_NAME"]]
		if "PREF_NAME" in df.columns.values :
			#Use a try except because PREF_NAME is not always known
			l = []
			for x in df["PREF_NAME"]:
				try :
					l.append(d_pref[encode_text(x)])
				except AttributeError :
					l.append("unknown")
			df["PREF_NAME"] = l
		#Save file
		df.to_csv("../Data/Data_translated/%s_translated.csv" % f,index = False)

if __name__ == '__main__':
	
	translate()