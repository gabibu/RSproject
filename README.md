
implementation of xDeepFM: Combining Explicit and Implicit Feature Interactions
for Recommender Systems 
https://arxiv.org/pdf/1803.05170v3.pdf

Datasets:

Anime Recommendations Database 
https://www.kaggle.com/CooperUnion/anime-recommendations-database?select=rating.csv

Click-Through Rate Prediction on Avazu

https://www.kaggle.com/c/avazu-ctr-prediction

To run:

1. download the dataset for the links.
2. install requirement.txt
3. for every dataset run the pre processing notebook and save the results to csv. 
annime_preprocessing.ipynb,  avazu_preprocessing.ipynb

4. to train: xdfm train_xdfm_anime.ipynb and train_xdfm_avanzu.ipynb (update the dataset path to where you kept the data in 2)
update the model saving path 

5. train xgboost train_xg_avanzu.ipynb and train_xgbboost_anime.ipynb
6. train improved xdfm train_improved_xdfm_anime.ipynb
7. metrics - use avanzu_metrics.ipynb and anime_metrics.ipynb - load the models from the path in 4. 
