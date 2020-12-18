import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def report_result(y_predicted,y_true,classifier_name,labels, title):

    print("{} classifer".format(classifier_name))
    print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_predicted)))
    print(classification_report(y_true, y_predicted, labels=labels))
    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    print("Confusion matrix:\n{}".format(cm))
    img = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    figure, axes = plt.subplots(figsize=(10, 10))
    img.plot(ax=axes, xticks_rotation='vertical')
    axes.set_title(title)
    plt.show()

def plot_history(history, title):

    plt.figure(figsize=(6,5))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    #plt.xticks(np.arrange(0,20,steps=1))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

