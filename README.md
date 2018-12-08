# Document-Classification-Singular-Value-Decomposition
* This project uses singular value decomposition(SVD) approach which improved the accuracy of the classifier by 6% over traditional
appraoches.
* The clasifier used here is the SVM, SVD is used to reduce the no of features to 300.
* The data processing is done in python and the model is built in R.

* Different files and folders description is as follows:
* Download the files from the below link:
* https://drive.google.com/file/d/1YeEkHMFiJLXll1zaqRYM8Fv86iQh4rpO/view?usp=sharing
* data_preprocessing.ipynb : File used for preprocessing the data.
* text_classifier_SVD.R : File used for the classification. (model)
* data_final.csv : Processed file after wrangling the train data
* test_final.csv : Processed file after wrangling the test data
* testing_labels_pred.txt : Final predicted labels

* Steps to reproduce:
* Unzip the zipped folder to any location on the local system.
* Download all the required packages, software and files.
* Run the jupyter notebook file data_preprocessing.ipynb file.
* The above python file creates data_final.csv
* Run the R file text_classifier_SVD.R.
* It builds the SVM model using the data created.
