# import python libraries
#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn
import matplotlib.pyplot as plt
import polars
import os


def plot_histogram(labels,histogram_output):
    '''
    do the thing
    '''

    types = ["norm", "bckg", "artf", "null", "nneo", "infl", "susp", "dcis", "indc"]
    colors = ["lightpink", "peachpuff", "#CBC3DB", "#BAD9BB", "lightblue", "thistle", "#BED4E9", "pink", "#C5CDBA"]

    label_count = {}

    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    for t in types:
        if t not in label_count:
            label_count[t] = 0

    l_types = list(label_count.keys())
    counts = list(label_count.values())

    plt.bar(l_types, counts, color=colors)
    plt.xlabel('Label Types')
    plt.ylabel('Number of Labels')

    plt.savefig(histogram_output)
    plt.cla()
# plot confusion matrix 
#
def plot_confusion_matrix(model,inlabels,data,outputpath):
    """
        Objective:
            Plots the confusion matrix.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param labels: List of labels of labeled windows.
        :type labels: list of strings

        :param data: x
        :type data: x

        :param outputpath: Directory path for the output to be stored.
        :type outputpath: path
    """
    
    # generate model predicitions
    #
    predictions = model.predict(data)
    
    # generate confusion matrix with labels and predictions
    #
    conf_mat = confusion_matrix(inlabels, predictions,labels=list(set(inlabels)))
    print(list(set(inlabels)))
    # heatmap the confusion matrix
    #
    seaborn.heatmap(conf_mat, cmap='Blues',yticklabels=list(set(inlabels)),xticklabels=list(set(inlabels)))
    
    # save the figure
    #
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(outputpath)
    plt.cla()
# find the mean confidence %
#
def mean_confidence(model,data):
    """
        Objective:
            Finds the mean of all confidence percentages.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param data: x
        :type data: x
        
        :return: confidence average
        :rtype: float
    """

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

def plot_decisions(model,data,output_path,frame_locs,framesizes,header):
    """
        Objective:
            Plots regions on an image with associated labels based on predictions.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param output_path: Directory path for the output to be stored.
        :type output_path: path

        :param frame_locs: x
        :type frame_locs: x

        :param framesizes: x
        :type framesizes: x

    """
    
    # get the predictions
    #
    class_predictions=model.predict(data)
    rows = []

    # append the predictions and the guesses
    #
    for i,x in enumerate(class_predictions):
        rows.append([x,framesizes[i],frame_locs[i][0],frame_locs[i][1]])

    # set the column titles and write to csv
    #

    if os.path.exists(output_path):
        os.remove(output_path)
    
    with open(output_path,'a') as file:
        file.write(header + '% ')
        MySchem = ["labels",'framesizes','top_left_x','top_left_y']
        df = polars.DataFrame(rows,schema=MySchem)
        df.write_csv(file)


