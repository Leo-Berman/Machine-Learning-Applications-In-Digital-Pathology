from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv
import numpy as np
def main():

    train_list = ["QDATrain1", "QDATrain2"]
    # train_list = ["QDATrain2"]
    read = []
    for x in train_list:
        # read the rows into memory
        #
        relpath = "./DATA/"+x+".csv"
        file = open(relpath,'r')
        data = csv.reader(file)
        for row in data:
            read.append(row)
        file.close()

    # list for holding the labels and data
    #
    mydata = []
    labels = []
    
    #  iterate through the data
    #
    for x in read:

        # add the label to a list
        #
        label = x.pop(0)
        labels.append(label)

        # add the data into a separate list
        #
        mydata.append(np.array([float(z) for z in x]))

    # reshape the arrays
    #
    labels = np.array(labels).ravel()
    mydata = np.array(mydata)
    print(len(labels))
    # Fit the model
    #
    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(mydata, labels)

    print(427 * QDA.score(mydata,labels))
    print(420/427)
    eval_list = ["QDATrain1", "QDATrain2","QDAEval1","QDAEval2"]
    # eval_list = ["QDATrain2","QDAEval1","QDAEval2"]
    for x in eval_list:
        
        # read the rows into memory
        
        relpath = "./DATA/"+x+".csv"
        file = open(relpath,'r')
        data = csv.reader(file)
        read = []
        for row in data:
            read.append(row)
        file.close()

        #  iterate through the data
        #
        labels = []
        mydata = []
        for y in read:

            # add the label to a list
            #
            label = y.pop(0)
            labels.append(label)

            # add the data into a separate list
            #
            mydata.append(np.array([float(z) for z in y]))
        
        labels = np.array(labels).ravel()
        mydata = np.array(mydata)
    
        
        print(x + " score = ",QDA.score(mydata,labels))
        # print the score 
        #

        # get a list of guesses
        # 
        guesses = []
        for x in mydata:
            guesses.append(QDA.predict(x.reshape(1,-1))[0])


        result = filter(lambda x: x != 'bckg',guesses)
        # print(list(result))
    
    

    

main()