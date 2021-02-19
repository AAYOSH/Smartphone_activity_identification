# Smartphone_activity_identification
Project developed in the post graduate subject IA048-Machine Learning at FEEC,UNICAMP. Multi-class classification using logistic regression(softmax) and k-nearest neighbors

# Logistic Regression
For logistic regression problem, it was used the holdout cross-validation method, and for the model it was used a baseline method(no grid-search for hyperparameters) just to compare with KNN algorithm.

# KNN
For KNN problem, the number of neighbors was determined by using a 5-fold cross-validation method, and by analyzing the F1-score metric, the number found was k = 16.

### F1-score macro-average
As the problem consists of a 5 classes prediction, global accuracy F1-score macro-average was used trying to avoid the misleading information that the umbalanced dataset could lead us.

# Results:
By comparing the F1-score accuracy and the runtime, the Logistic Regression outperforms the KNN algorithm in both metrics.