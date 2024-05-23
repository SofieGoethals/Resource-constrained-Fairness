from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import LFR
import aif360
from aif360.metrics import BinaryLabelDatasetMetric
from IPython.display import Markdown, display
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.sklearn.preprocessing import LearnedFairRepresentations as LFR_sk
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import MetaFairClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing as EQ
import fairlearn
from fairlearn.postprocessing import ThresholdOptimizer
from statistics import mean
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

#from functions_with_validation import *
import os


def assign_label(C,p,results):
    N_pri=round(C*(100-p)/100)
    N_pro=round(C*p/100)
    if N_pri>sum(results.protected==False):
        N_pro+=N_pri-sum(results.protected==False)
        N_pri=sum(results.protected==False)
    if N_pro>sum(results.protected==True):
        N_pri+=N_pro-sum(results.protected==True)
        N_pro=sum(results.protected==True)
    if N_pro>sum(results.protected==True) and N_pri>sum(results.protected==False):
        print('Error: capacity is higher than the number of instances in the dataset')
        return
    #print(N_pro)
    #print(N_pri)
    scores_pri=results[results.protected==False].biased_scores
    score_index_pairs_pri= [(score, index) for index, score in scores_pri.items()]
    # Sort the list of tuples in descending order of scores
    sorted_pairs_pri = sorted(score_index_pairs_pri, key=lambda x: x[0], reverse=True)
    indices_pri = [t[1] for t in sorted_pairs_pri[0:N_pri]]
    scores_pro=results[results.protected==True].biased_scores
    score_index_pairs_pro= [(score, index) for index, score in scores_pro.items()]
    # Sort the list of tuples in descending order of scores
    sorted_pairs_pro = sorted(score_index_pairs_pro, key=lambda x: x[0], reverse=True)
    indices_pro = [t[1] for t in sorted_pairs_pro[0:N_pro]]
    preds= pd.Series([1 if i in indices_pro or i in indices_pri else 0 for i in results.index], index=results.index)
    #preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
    return preds


def plot_allocation_by_resourcelevel(metrics, results, C_list, index, dataset, ymin=None,ymax=None, legend=False):
    C = C_list[index]
    percentages = []
    avg_prec = []
    preds = {}
    for percentage in range(0, 101):
        preds[percentage] = assign_label(C, percentage, results)
        avg_prec.append(mean(results[preds[percentage] == 1].target))
        percentages.append(percentage)

    # Plot the results
    plt.facecolor = (1, 1, 1)
    
    plt.plot(percentages, avg_prec)
    plt.xlabel('Percentage assigned to protected group', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('{} - (Selection rate = {}%)'.format(dataset, index), fontsize=18)
    optimal_allocation = percentages[avg_prec.index(max(avg_prec))]
    plt.plot(optimal_allocation - 1, max(avg_prec), 'o', markersize=10, color='blue', label='Optimal allocation')
    plt.plot(0, avg_prec[0], 'o', markersize=10, color='red', label='Privileged group first')
    plt.plot(100, avg_prec[100], 'o', markersize=10, color='purple', label='Protected group first')
    plt.plot(metrics['preds_unfair'][C][results.protected == True].value_counts()[1] / C * 100,
             metrics['prec_unfair'][index], 'o', markersize=10, color='lightblue', label='Unfair model')
    plt.plot(metrics['preds_dp'][C][results.protected == True].value_counts()[1] / C * 100,
             metrics['prec_dp'][index], 'o', markersize=10, color='lightgreen', label='Fair allocation (DP)')
    plt.plot(metrics['preds_eo'][C][results.protected == True].value_counts()[1] / C * 100,
             metrics['prec_eo'][index], 'o', markersize=10, color='darkgreen', label='Fair allocation (EO)')
    if legend==True:
        plt.legend(fontsize=13)
    if ymin!=None:
        plt.ylim([ymin, ymax])  
    #plt.ylim([0,1])
    # Create the folder if it doesn't exist
    folder_name = os.path.join('Figures', dataset)
    os.makedirs(folder_name, exist_ok=True)

    # Save the figure in the folder
    file_name = os.path.join(folder_name, 'resourcelevel_{}.png'.format(index))
    plt.savefig(file_name)
    plt.show()
    
    
    
    

def plot_cost(performance_unfair, performance_fair,C_list, results, performance_metric, fairness_metric, dataset_name,  performance_fair2=None, fairness_metric2=None):
    max_capacity = max(C_list)
    min_capacity = min(C_list)  
    plt.figure(facecolor=(1, 1, 1))
    plt.plot(C_list, [a - b for a, b in zip(performance_unfair, performance_fair)], color='blue', label=fairness_metric)
    plt.ylabel('Cost of fairness \n ($\Delta$ in {})'.format(performance_metric), fontsize=15)
    plt.xlabel('Selection rate (in %)', fontsize=15)
    #plt.ylim([-0.01,0.1])
    plt.legend()
    # Generate 10 equally spaced numbers between min_capacity and max_capacity
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.axvline(x=(mean(results[results.protected==True].target)*max_capacity), color='red', linestyle='--', label='Base rate protected group')
    plt.axvline(x=(mean(results[results.protected==False].target)*max_capacity), color='pink', linestyle='--', label='Base rate privileged group')
    if performance_fair2!=None:
        plt.plot(C_list, [a - b for a, b in zip(performance_unfair, performance_fair2)], color='lightblue', label=fairness_metric2)
    plt.legend(fontsize=12)
    plt.title('{}: Cost of fairness vs selection rate'.format(dataset_name), fontsize = 17)    
    plt.show()

def plot_distribution(performance_unfair, performance_fair,disparity_unfair, disparity_fair, C_list, performance_metric, fairness_metric, dataset_name):
    plt.figure(facecolor=(1, 1, 1))
    max_capacity = max(C_list)
    min_capacity = min(C_list)
    plt.plot(C_list, performance_unfair, color='red', label='ML model')
    plt.plot(C_list,performance_fair, color='blue',label =fairness_metric)
    plt.ylabel(performance_metric)
    plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])

    plt.legend()
    # Create the second plot
    ax2 = plt.twinx()
    ax2.plot(C_list, disparity_unfair, color='red', linestyle='--', label='ML model')
    ax2.plot(C_list, disparity_fair, color='blue', linestyle='--', label=fairness_metric)
    ax2.set_ylabel('$\Delta$ in {}'.format(fairness_metric))
    ax2.set_ylim([0,1])
    plt.title('{} \n {} vs selection rate'.format(dataset_name, performance_metric))
    plt.show()
    
def plot_distribution_prot_priv(performance_prot, performance_priv, results, C_list, performance_metric, dataset_name, label1, label2):
    plt.figure(facecolor=(1, 1, 1))
    max_capacity = max(C_list)
    min_capacity = min(C_list)
    plt.plot(C_list, performance_prot, color='red', label=label1)
    plt.plot(C_list,performance_priv, color='blue',label =label2)
    plt.ylabel(performance_metric)
    plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.axvline(x=(mean(results[results.protected==True].target)*max_capacity), color='red', linestyle='--', label='Base rate protected group')
    plt.axvline(x=(mean(results[results.protected==False].target)*max_capacity), color='pink', linestyle='--', label='Base rate privileged group')
    plt.legend()
    plt.title('{} \n {} vs selection rate'.format(dataset_name, performance_metric))
    plt.show()