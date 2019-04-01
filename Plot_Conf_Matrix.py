'''Plot a normalized and not-normalized confustion matrix of a classification problem'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
    

def plot_conf_matrix(y_true, y_pred, classes, normalize=False, 
                     title=None, cmap = plt.cm.Blues):
    ''' plot confustion matrix; can also plot a normalized matrix using "normalize=True"
    '''
    
    
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
        else:
            title = 'Confusion Matrix, non-Normalized'
            
    #create matrix
    cm = confusion_matrix(y_true, y_pred)
    #use lables that appear in the data
    classes = unique_labels(classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusiton Mattrix, non-Normalized')
        
    print(cm)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax=ax)
    #show all the ticks
    ax.set(xticks=np.arange(cm.shape[1]),
          yticks = np.arange(cm.shape[0]),
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')
    #make the tick labels look nice
    plt.setp(ax.get_xticklabels(), rotation = 45, ha='right',
            rotation_mode = 'anchor')
    
    #loop over data dimension and create text annotation.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i,j] > thresh else 'black')
    fig.tight_layout()
    return ax