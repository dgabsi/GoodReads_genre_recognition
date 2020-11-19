import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def report_result(y_predicted,y_true,best_val_score,classifier_name,params,run_time, labels):

    print("{} classifer .Training time: {:.2f}".format(classifier_name, run_time))
    if params is not None:
        print("Best parameters: {}".format(params))
    if best_val_score is not None:
        print("Validation score: {:.2f}".format(best_val_score))
    print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_predicted)))
    print(classification_report(y_true, y_predicted, labels=labels))
    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    print("Confusion matrix:\n{}".format(cm))
    img = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    figure, axes = plt.subplots(figsize=(10, 10))
    img.plot(ax=axes, xticks_rotation='vertical')
    plt.show()

def plot_history(history):

    plt.figure(figsize=(10,10))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Run history")
    plt.show()
