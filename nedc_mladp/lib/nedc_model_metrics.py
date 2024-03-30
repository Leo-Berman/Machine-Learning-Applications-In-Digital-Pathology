# import python libraries
#
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

# plot confusion matrix 
#
def plot_confusion_matrix(model,labels,data,outputpath):
    
    # generate model predicitions
    #
    predictions = model.predict(data)
    
    # generate confusion matrix with labels and predictions
    #
    conf_mat = confusion_matrix(labels, predictions)
    
    # heatmap the confusion matrix
    #
    seaborn.heatmap(conf_mat, cmap='Blues')
    
    # save the figure
    #
    plt.savefig(outputpath)

# find the mean confidence %
#
def mean_confidence(model,data):

    # find predictions
    #
    class_predictions=model.predict_proba(data)

    # number for total max predictions
    #
    total_max_predictions = 0
    for x in class_predictions:
        total_max_predictions+=max(x)

    # return the mean of all the confidences
    #
    return total_max_predictions/len(class_predictions)