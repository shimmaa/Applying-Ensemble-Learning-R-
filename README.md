# Applying-Ensemble-Learning-R-
Use the different ensemble methods to classify the handwritten digits in MNIST Dataset. In order to make the implementation easier we have stripped it down to only 2 digits: 2 and 3 .
You need to submit your source code as well as your assignment report.
–
Bagging
a. Train a bagging classifier on training data using 50 bootstrap samples. Measure its misclassification
rate.
b. Draw a figure that shows change in misclassification rate of bagging classifier if we add samples. (x
axis: number of datasets 1 50 ; y axis: misclassification rate)
–
Random Forest
a. Train a simple random forest type classifier, where each tree is constructed using only a subset of
m = 50 variables. Again use 50 bootstrap samples. Measure its misclassification rate.
b. Try values 10 , 50 and 300 for m and draw similar figure as in 1 .b, but now with all the random
forest and bagging curves. What can be said about the results?
c. Try to classify the data using the random Forest package. See if it performs better than the
previous options. Try to adjust some parameters, to see if you can improve the prediction further.
–
Boosting
a. Train Adaboost classifier on our example doing 50 steps. Measure its misclassification rate.
b. The default tree parameters might provide too rich model for boosting and therefore induce
overfitting. To avoid this, limit the tree depth to 2 (can be set as control = list( maxdepth = 2 ) in
rpart ) and train again the classifier for 50 steps. Measure its misclassification rate.
c. Combine the curves for all classifiers to one figure as in Exercises 1 .b and 2 .b. Add also the single
tree error to the picture. Interpret the results from previous figure. What methods perform the
best. How does changing the model complexity affect the ensemble performance in bagging and in
boosting
