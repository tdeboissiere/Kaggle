#Ponpare validation scripts


##Recommendation through similarity

    python similarity_distance.py

This script computes a similarity score between users and test items.  
- User and items features are represented in the same vector space.
- The similarity can be computed with any metric in [scipy.spatial.distance](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
- A weight is added to each group of features to tune their relative importanc.e

##Recommendation through supervised learning methods 

    python supervised_learning.py

This script fits a supervised learning method (implemented here : xgboost regressor and an SVM based learning to rank method).  
The regressor target is modeled from the detail and view information.  
**N.B. I compute the cartesian product of the list of test coupons and users. This takes time and quite a bit of memory**.  

##Recommendation through hybrid matrix factorisation method 

    python ponpare_lightfm.py

This script fits the [lightfm](https://github.com/lyst/lightfm) hybrid matrix factorisation model to the data. 
- Can incorporate user and item metadata. 
- Can be adapted to explicit or implicit feedback.

##Similarity Model optimisation

    python optimise_similarity.py

Use [hyperopt](https://github.com/hyperopt/hyperopt) to find the best weights in the similarity model.

##Blending optimisation

    python optimise_blending.py

Use [hyperopt](https://github.com/hyperopt/hyperopt) to find the best weights to aggregate predictions.
