#Ponpare competition

##General information

This folder is divided into 6 subfolders  
- **Ponpare_utilities:** notably stores the preprocessing scripts
- **Ponpare_submission:** code used to create a submission.
- **Ponpare_validation:** to validate models on training data.
- **Ponpare_visual:** to visualise the data.
- **Data:** to store the data
- **Submissions:** to store submission files

##Using scripts

All folders contain detailed README.md and the code is heavily commented.  
To use the code, follow these steps :  
1. Clone to a local repository  
2. Download the [data](https://www.kaggle.com/c/coupon-purchase-prediction/data).  
3. Put the data into the /Data/Data_japanese/ folder and unzip.  
4. cd /Ponpare_utilities && python configure_competition.py  
5. cd /Ponpare_submission && python blending.py to create a submission file blending several models.    
6. cd /Ponpare_validation && python script.py or python optimize_XXX.py to evaluate various models on validation data.    