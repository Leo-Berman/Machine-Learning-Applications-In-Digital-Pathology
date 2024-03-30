# import python libraries
#
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import polars

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

def plot_decisions(model,data,output_path,frame_locs,framesizes):
    class_predictions=model.predict(data)
    print(class_predictions)
    rows = []

    for i,x in enumerate(class_predictions):
        rows.append([x,framesizes[i],frame_locs[i][0],frame_locs[i][1]])

    MySchem = ["labels",'framesizes','top_left_x','top_left_y']
    df = polars.DataFrame(rows,schema=MySchem)
    df.write_csv(output_path)