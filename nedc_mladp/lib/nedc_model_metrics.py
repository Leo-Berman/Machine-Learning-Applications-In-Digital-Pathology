from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import numpy
def plot_confusion_matrix(model,labels,data,outputpath):
    predictions = model.predict(data)
    conf_mat = confusion_matrix(labels, predictions)
    seaborn.heatmap(conf_mat, cmap='Blues')
    plt.savefig(outputpath)

def zscore(model,labels,data):
    predictions = list(model.predict(data))
    labelnames=['norm','bckg','artf','null','nneo','infl','susp','dcis','indc']
    labeldict={}
    for i,x in enumerate(labelnames):
        labeldict[x]=i
    labelnums=[labeldict[x] for x in labels]
    my_mean=numpy.mean(labelnums)
    my_std=numpy.std(labelnums)
    ztotal = 0
    for x in labelnums:
        ztotal+=(x-my_mean)/my_std
    print(ztotal)
    zscore = ztotal/len(labelnums)
    return zscore