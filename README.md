# BidBot-Identification
ML based Bot Identification in Bidding Systems (Course Project for CSCE 633)

Dataset: https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data


Code Details:
1. Libraries required:
   1. pandas
   2. numpy
   3. tensorflow
   4. keras
   5. scikit-learn
   7. matplotlib


Steps to run:
Before executing any of the following commands, ‘train.csv’, 'bids.csv' and ‘test.csv’ files of the  dataset must be present in the ‘data’ directory (download them from the link above).
   1. Create and save features
      1. Run the command:
         $python main.py create
      2. It will create and save pickle files of the features in ‘model’ directory. 
         Please note that there should be a directory named 'model' created before running this command.
   
   2. Generate predictions from the features created in step 1
      1. Run the command:
         $python main.py test
      2. It will prompt you to select the type of classifier. 
      3. After choosing an option, the selected model will be created along with its 5-fold cross-validation roc curve genrated and auc score calculated. 
      4. Predictions will be generated for the test set and saved in submissions.csv file under 'data' directory.

* Code is developed using python 3.6.1 and is not compatible with python 2.7
