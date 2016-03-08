#Prudential competition

![prudential](../Images/prudential.png)  

Below you will find the code that allowed me to get a top 5 % ranking at the recent Kaggle [Prudential competition](https://www.kaggle.com/c/prudential-life-insurance-assessment)

##Prerequisites

- Up to date versions of scipy, numpy, pandas, xgboost, keras and scikit-learn
- The tsne package for R.  

##General information

There are two subfolders
- **Data/Raw,** where you must put the unzipped Data downloaded from Kaggle
- **TSNE,** where TSNE features are computed

The rest are scripts to process the data, build a stacked model and make submissions

##Using scripts

A lof of the documentation is inline to help you navigate the code.
There may be some directory conflicts due to code refactoring but they should easily be solvable

The general idea to get predictions is to :

1. Clone to a local repository  
2. Download the [data](https://www.kaggle.com/c/prudential-life-insurance-assessment/data).  
3. Put the data into the /Data/Raw/ folder and unzip.   
4. cd /TSNE and follow instructions to build the t-SNE features
5. Then cd .. and run `python preprocessing_lv1.py` to compute correlation features
6. Then `python train_lv1.py` to build lv1 features (regression/classification output of a first batch of models)
7. Then uncomment the lines at the bottom of `python preprocessing_lv1.py` to create a single level1 feature file
8. Then run `python get_bayes_feat.py` to create Bayes inspired features for lv2 models
9. Then `python train_lv2.py` to build lv2 models, check CV score and/or build submissions    

The final solution I submitted is a majority vote of a combination of models produced from this scripts.