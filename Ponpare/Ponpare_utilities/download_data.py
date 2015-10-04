import os
import getpass


def download_and_unzip_data():
    """ Download data from Kaggle. Need Kaggle ID and pwd """

    username = raw_input("Enter Kaggle username: ")
    pwd = getpass.getpass("Enter Kaggle password: ")

    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/coupon_list_test.csv.zip" %
        (username, pwd))
    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/coupon_list_train.csv.zip" %
        (username, pwd))
    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/coupon_detail_train.csv.zip" %
        (username, pwd))
    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/coupon_detail_test.csv.zip" %
        (username, pwd))
    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/user_list.csv.zip" %
        (username, pwd))
    os.system(
        "wget -P ../Data/Data_japanese/ --user=%s --password %s https://www.kaggle.com/c/coupon-purchase-prediction/download/documentation.zip" %
        (username, pwd))


if __name__ == '__main__':

    download_and_unzip_data()
