# Metrics

## Accuracy:
The proportion of all predictions (both positive and negative) that were correct

## Macro-averaged metrics (treats all classes equally):


### Precision_macro: 
The average precision across all classes, where precision is the ratio of true positives to all positive predictions
### Recall_macro: 
The average recall across all classes, where recall is the ratio of true positives to all actual positives
### F1_macro: 
The harmonic mean of precision and recall, averaged across all classes


## Weighted-averaged metrics (accounts for class imbalance):

### Precision_weighted: 
Precision averaged across all classes, weighted by the number of instances in each class
### Recall_weighted: 
Recall averaged across all classes, weighted by the number of instances in each class
### F1_weighted: 
F1 score averaged across all classes, weighted by the number of instances in each class

## Time metrics:

### Training Time:
How long it took to train the model
### Prediction Time: 
How long it took to make predictions on the test set