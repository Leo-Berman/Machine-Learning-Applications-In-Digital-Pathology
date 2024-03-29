from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import numpy
def plot_confusion_matrix(model,labels,data,outputpath):
    predictions = model.predict(data)
    conf_mat = confusion_matrix(labels, predictions)
    seaborn.heatmap(conf_mat, cmap='Blues')
    plt.savefig(outputpath)

def mean_confidence(model,data):
    class_predictions=model.predict_proba(data)
    total_max_predictions = 0
    for x in class_predictions:
        total_max_predictions+=max(x)

    return total_max_predictions/len(class_predictions)