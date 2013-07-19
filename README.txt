ra2616SET-HW2 contains the following files and folders:-
============================================
nltk/
nltk-2.0.3-py2.7.egg-info/
    Folders containing the NLTK llibrary
ra2616Train.py 
    File containing the classifier
ra2616Test.py
    File containing the predictor based on model
test
    script calling ra2616Test.py
train
    script calling ra2616Train.py
negative-words.txt
    List of negative words
positive-words.txt
    List of positive words
ra2616Report.txt
    Write up describing the methodology and design choices.
README.txt
    This file
---------------------------------------------------------------------------

HOW TO RUN:
===========

To train the classifier and build a model, run the following commands:

chmod +x train
./train training_set_file.csv model_file_name 

OR alternatively 

python ra2616.py training_set_file.csv model_file_name 

----------------------------------------------------------------------------

To get predictions based on model_file_name, run the following commands:

chmod +x test
./test model_file_name test_set_file.csv output_Prediction_file

OR alternatively 

python ra2616Predict.py model_file_name test_set_file.csv output_Prediction_file 