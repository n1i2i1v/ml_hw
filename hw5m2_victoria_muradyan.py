import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import scipy 
import matplotlib.pyplot as plt

def prob_dist(x, sigma, mean):
    down = np.sqrt(2*np.pi*(sigma**2))
    up = np.exp(-((x-mean)**2)/(2.0*(sigma**2)))
    return up/down
x = np.linspace(0, 1, num=1000)
real = prob_dist(x,0.15,0.5)
predicted = prob_dist(x,0.13,0.6)

def get_tpr_fpr_auc(real, predicted):
    #sum values
    sum_pred = np.sum(predicted)
    sum_real = np.sum(real)
    #culminative values
    cum_tp = 0
    cum_fp = 0
    #tpr, fpr lists init
    TPR=[]
    FPR=[]
    for i in range(len(x)):
        #only positive vals are valid
        if predicted[i]>0:
            cum_tp+=predicted[len(x)-1-i]
            cum_fp+=real[len(x)-1-i]
        #get fpr, tpr for each val and append 
        fpr=cum_fp/sum_pred
        tpr=cum_tp/sum_real
        TPR.append(tpr)
        FPR.append(fpr)
        
    #get the area score
    auc=scipy.integrate.simps(TPR, FPR)
    
    return TPR, FPR, auc

def plot_roc_curve(real, predicted):
    tpr, fpr, auc = get_tpr_fpr_auc(real, predicted)
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(fpr, tpr)
    ax.plot(x,x, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid()

tpr, fpr, auc = get_tpr_fpr_auc(real, predicted)
print("The Auc is")
print(auc)
plot_roc_curve(real, predicted)
