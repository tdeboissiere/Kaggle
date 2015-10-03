#Ponpare utilities


## Evaluation metric
Implement Mean Average Precision @ k

##Master script

	python configure_competition.py

With this script : 
- Data is translated to English
- Validation data sets are created (see below for details)
- Preprocessing steps are carried out on the data and validation data (see below for details)

N.B. It takes a while for all preprocessing steps to complete

### Create validation data sets

	python create_validation.py

This creates a Validation/week_ID/ directory in Ponpare_submission.  
- Up to five different choices of validation weeks (dubbed week48 to week52).  
- This scripts create training files (detail_train, visit_train, coupon_list_train), a list of user (user_list) and a list of test coupon (coupon_list_test).  
- The coupon_list_test file is built such that it contains only coupons emitted during the validation week.  
- It also saves python dict to store the actual purchases of each user as well as different user lists.


###Data preprocessing (submission data)

    python preprocessing_submission.py

Clean, preprocess and transform the data for use in the various recommendation scripts.  
This takes a while and needs quite a bit of memory.

###Data preprocessing (validation data)

    python preprocessing_validation.py

Clean, preprocess and transform the validation data for use in the various recommendation scripts.  
This takes a while and needs quite a bit of memory.

##Run some checks on the data

    python run_checks.py

See inline documentation for more details.