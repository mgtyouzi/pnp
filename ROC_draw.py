import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score, accuracy_score

def draw_ROC(y_true, y_score):
    y_true0, y_sore0 = y_true, y_score
    y_true1, y_sore1 = y_true, y_score
    y_true2, y_sore2 = y_true, y_score
    y_true3, y_sore3 = y_true, y_score
    y_true4, y_sore4 = y_true, y_score

    fpr0, tpr0, thresholds0 = roc_curve(y_true0,y_sore0)
    fpr1, tpr1, thresholds1 = roc_curve(y_true1,y_sore1)
    fpr2, tpr2, thresholds2 = roc_curve(y_true2,y_sore2)
    fpr3, tpr3, thresholds3 = roc_curve(y_true3,y_sore3)
    fpr4, tpr4, thresholds4 = roc_curve(y_true4,y_sore4)


    roc_auc0 = auc(fpr0, tpr0)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)

    plt.title('Receiver Operating Characteristic')
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    # 设置标题大小
    plt.rcParams['font.size'] = '16'
    plt.plot(fpr0, tpr0, 'k-',color='k',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'AA_AUC = %0.5f'% roc_auc0)
    plt.plot(fpr1, tpr1, 'k-',color='grey',linestyle='-.',linewidth=3,label=u'A_AUC = %0.5f'% roc_auc1)
    plt.plot(fpr2, tpr2, 'k-',color='r',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'B_AUC = %0.5f'% roc_auc2)
    plt.plot(fpr3, tpr3, 'k-',color='red',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'C_AUC = %0.5f'% roc_auc3)
    plt.plot(fpr4, tpr4, 'k-',color='y',linestyle='-.',linewidth=3,markerfacecolor='none',label=u'D_AUC = %0.5f'% roc_auc4)

    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    y_true = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    y_score = [0.31689620142873609, 0.32367439192936548, 0.42600526758001989, 0.38769987193780364, 0.3667541015524296,
               0.39760831479768338, 0.42017521636505745, 0.41936155918127238, 0.33803961944475219, 0.33998332945141224]
    draw_ROC(y_true, y_score)
