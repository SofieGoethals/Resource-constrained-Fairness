
#pip install fairlearn

#from utils import *
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
def run_trade_off(X,y, sens_var, sensitive_value, name):
    metrics=['demographic_parity', 'true_positive_rate_parity', 'equalized_odds']
    results=run_constraints(X,y, sens_var, sensitive_value, metrics)
    print('Calculate metrics')
    print('The accuracy of the biased model is:', accuracy_score(results.target,results.biased_preds))
    print('The AUC of the biased model is:', roc_auc_score(results.target,results.biased_scores))
    print('The accuracy of the biased model for the protected group is:', accuracy_score(results[results.protected==True].target,results[results.protected==True].biased_preds))
    print('The AUC of the biased model for the protected group is:', roc_auc_score(results[results.protected==True].target,results[results.protected==True].biased_scores))
    print('The accuracy of the biased model for the privileged group is:', accuracy_score(results[results.protected==False].target,results[results.protected==False].biased_preds))
    print('The AUC of the biased model for the privileged group is:', roc_auc_score(results[results.protected==False].target,results[results.protected==False].biased_scores))
    print('Calculate DP')
    C_list=[i for i in np.linspace(start=1, stop=len(results), num=100, dtype=int)]
    preds_unfair,preds_dp={},{}
    for C in C_list:
        preds_unfair[C]=pd.Series(assign_label_unfair(C,results), index=results.index)
        preds_dp[C]=pd.Series(assign_label_dp(C,results), index=results.index)
    acc_unfair, acc_dp,dd_unfair,dd_dp=[],[],[],[]
    for C in C_list:
        acc_unfair.append(accuracy_score(results.target,preds_unfair[C]))
        acc_dp.append(accuracy_score(results.target,preds_dp[C]))
        dd_unfair.append(preds_unfair[C][results.protected==False].mean()-preds_unfair[C][results.protected==True].mean())
        dd_dp.append(preds_dp[C][results.protected==False].mean()-preds_dp[C][results.protected==True].mean())
    print('plot DP')
    max_capacity = max(C_list)
    min_capacity = min(C_list)
    plt.figure(facecolor=(1, 1, 1))
    plt.plot(C_list, acc_unfair, color='red', label='ML model')
    plt.plot(C_list,acc_dp, color='blue',label ='DP')
    plt.ylabel('Accuracy')
    plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.legend()
    ax2 = plt.twinx()
    ax2.plot(C_list, dd_unfair, color='red', linestyle='--', label='ML model')
    ax2.plot(C_list, dd_dp, color='blue', linestyle='--', label='DP')
    ax2.set_ylabel('Demographic disparity')
    ax2.set_ylim([0,1])
    plt.title(name + ': Selection rate - Demographic Parity')
    plt.show()
    print('Calculate EO')
    C_list=[i for i in np.linspace(start=1, stop=len(results), num=100, dtype=int)]
    preds_eo={}
    for C in C_list:
       preds_eo[C]=pd.Series(assign_label_eo_corr(C,results), index=results.index)
    acc_unfair, acc_eo,eod_unfair,eod_eo=[],[],[],[]
    for C in C_list:
        acc_unfair.append(accuracy_score(results.target,preds_unfair[C]))
        acc_eo.append(accuracy_score(results.target,preds_eo[C]))
        eod_unfair.append(preds_unfair[C][(results.protected==False) & (results.target==True)].mean()-preds_unfair[C][(results.protected==True) & (results.target==True)].mean())
        eod_eo.append(preds_eo[C][(results.protected==False) & (results.target==True)].mean()-preds_eo[C][(results.protected==True) & (results.target==True)].mean()) 
    print('plot EO')
    plt.figure(facecolor=(1, 1, 1))
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.plot(C_list, acc_unfair, color='red', label='ML model')
    plt.plot(C_list,acc_eo, color='blue',label ='EO')
    plt.ylabel('Accuracy')
    plt.xlabel('Selection rate (in %)')
    plt.legend()
    plt.title(name + ': Selection rate - Equal Opportunity')
    ax2 = plt.twinx()
    ax2.plot(C_list, eod_unfair, color='red', linestyle='--', label='ML model')
    ax2.plot(C_list, eod_eo, color='blue', linestyle='--', label='EO')
    ax2.set_ylabel('Equal Opportunity Difference')
    ax2.set_ylim([0,1])
    plt.show()
    #plot together
    plt.figure(facecolor=(1, 1, 1))
    acc_diff_eo=[a - b for a, b in zip(acc_unfair, acc_eo)]
    acc_diff_dp=[a - b for a, b in zip(acc_unfair, acc_dp)]
    plt.plot(C_list,acc_diff_eo, color='lightblue', label='EO')
    plt.plot(C_list, acc_diff_dp, color='blue', label='DP')
    plt.ylabel('Cost of fairness \n ($\Delta$ in accuracy)')
    plt.xlabel('Selection rate (in %)')
    plt.legend()
    plt.ylim([-0.015, 0.15])
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.axvline(x=(mean(results[results.protected==True].target)*max_capacity), color='red', linestyle='--', label='Base rate protected group')
    plt.axvline(x=(mean(results[results.protected==False].target)*max_capacity), color='pink', linestyle='--', label='Base rate privileged group')
    plt.legend()
    plt.title(name)
    plt.show()
    print('Maximum cost of fairness (accuracy) for DP: {} and EO {}'.format(max(acc_diff_dp), max(acc_diff_eo)))
    prec_diff_dp, prec_diff_eo=calculate_precision(results, preds_unfair, preds_dp, preds_eo, C_list, name, good_outcome=1)
    print('Maximum cost of fairness (precision) for DP: {} and EO {}'.format(max(prec_diff_dp), max(prec_diff_eo)))
    recall_diff_dp, recall_diff_eo=calculate_recall(results, preds_unfair, preds_dp, preds_eo, C_list, name, good_outcome=1)
    print('Maximum cost of fairness (recall) for DP: {} and EO {}'.format(max(recall_diff_dp), max(recall_diff_eo)))
    #return max(acc_diff_dp), max(acc_diff_eo), max(prec_diff_dp), max(prec_diff_eo), max(recall_diff_dp), max(recall_diff_eo)
    return mean(acc_diff_dp), mean(acc_diff_eo), mean(prec_diff_dp), mean(prec_diff_eo), mean(recall_diff_dp), mean(recall_diff_eo)

def calculate_precision(results, preds_unfair, preds_dp, preds_eo, C_list, name, good_outcome=1):
    prec_unfair, prec_dp, prec_eo=[],[],[]
    max_capacity = max(C_list)
    min_capacity = min(C_list)
    for C in C_list:
        prec_unfair.append(results[preds_unfair[C]==good_outcome].target.value_counts(normalize=True)[good_outcome])
        try: 
            prec_dp.append(results[preds_dp[C]==good_outcome].target.value_counts(normalize=True)[good_outcome])
        except KeyError:
            if len(results[preds_dp[C]==good_outcome])==0:
                prec_dp.append(1)
            else:
                prec_dp.append(0)         
    for C in C_list:
        try: 
            prec_eo.append(results[preds_eo[C]==good_outcome].target.value_counts(normalize=True)[good_outcome])
        except KeyError:
            if len(results[preds_eo[C]==good_outcome])==0:
                prec_eo.append(1)
            else:
                prec_eo.append(0)
    plt.plot(C_list, prec_unfair, color='red', label='ML model')
    plt.plot(C_list,prec_dp, color='blue',label ='DP')
    plt.plot(C_list,prec_eo, color='lightblue',label ='EO')
    plt.ylabel('Precision')
    plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    #plt.ylim([0,1])
    plt.legend()
    plt.show()
    prec_diff_eo=[a - b for a, b in zip(prec_unfair, prec_eo)]
    prec_diff_dp=[a - b for a, b in zip(prec_unfair, prec_dp)]
    plt.plot(C_list, prec_diff_eo, color='lightblue', label='EO')
    plt.plot(C_list, prec_diff_dp, color='blue', label='DP')
    plt.ylabel('Cost of fairness \n ($\Delta$ in precision )')
    plt.xlabel('Selection rate (in %)')
    plt.ylim([-0.015, 0.15])
    plt.legend()
    # Generate 10 equally spaced numbers between min_capacity and max_capacity
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.axvline(x=(mean(results[results.protected==True].target)*max_capacity), color='red', linestyle='--', label='Base rate protected group')
    plt.axvline(x=(mean(results[results.protected==False].target)*max_capacity), color='pink', linestyle='--', label='Base rate privileged group')
    plt.legend()
    plt.title(name)
    plt.show()
    return prec_diff_dp, prec_diff_eo
    
def calculate_recall(results, preds_unfair, preds_dp, preds_eo, C_list, name, good_outcome=1):
    #recall: out of the instances that are actually positive, how many did i find?
# out of all predicted positives, how much are actually positive?
    recall_unfair, recall_dp, recall_eo=[],[],[]
    max_capacity = max(C_list)
    min_capacity = min(C_list)
    for C in C_list:
        recall_unfair.append(preds_unfair[C][results.target==good_outcome].value_counts(normalize=True)[good_outcome])
        recall_dp.append(preds_dp[C][results.target==good_outcome].value_counts(normalize=True)[good_outcome])
        try:
            recall_eo.append(preds_eo[C][results.target==good_outcome].value_counts(normalize=True)[good_outcome])
        except KeyError:
            recall_eo.append(0)
    plt.plot(C_list, recall_unfair, color='red', label='ML model')
    plt.plot(C_list,recall_dp, color='blue',label ='DP')
    plt.plot(C_list,recall_eo, color='lightblue',label ='EO')
    plt.ylabel('Recall')
    plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    #plt.ylim([0,1])
    plt.legend()
    plt.show()
    recall_diff_eo=[a - b for a, b in zip(recall_unfair, recall_eo)]
    recall_diff_dp=[a - b for a, b in zip(recall_unfair, recall_dp)]
    plt.plot(C_list, recall_diff_eo, color='lightblue', label='EO')
    plt.plot(C_list, recall_diff_dp, color='blue', label='DP')
    #plt.plot(C_list, [a - b for a, b in zip(recall_unfair, recall_eo)], color='lightblue', label='EO')
    #plt.plot(C_list, [a - b for a, b in zip(recall_unfair, recall_dp)], color='blue', label='DP')
    plt.ylabel('Cost of fairness \n ($\Delta$ in recall )')
    plt.xlabel('Selection rate (in %)')
    plt.ylim([-0.015, 0.15])
    plt.legend()
    plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    plt.axvline(x=(mean(results[results.protected==True].target)*max_capacity), color='red', linestyle='--', label='Base rate protected group')
    plt.axvline(x=(mean(results[results.protected==False].target)*max_capacity), color='pink', linestyle='--', label='Base rate privileged group')
    plt.legend()
    plt.title(name)
    plt.show()
    return recall_diff_dp, recall_diff_eo

#%% new functions

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
    preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
    return preds


def assign_label_unfair(C,results):
    scores=results.biased_scores
    score_index_pairs= [(score, index) for index, score in scores.items()]
    # Sort the list of tuples in descending order of scores
    sorted_pairs= sorted(score_index_pairs, key=lambda x: x[0], reverse=True)
    indices = [t[1] for t in sorted_pairs[0:C]]
    preds = [1 if i in indices else 0 for i in results.index]
    return preds


def assign_label_dp(C,results):
    p_pri=sum(results.protected==False)/len(results)
    p_pro=sum(results.protected==True)/len(results)
    N_pri=round(C*p_pri)#int(C*p_pri)
    N_pro=round(C*p_pro)#int(C*p_pro)
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
    preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
    return preds

def assign_label_eo(C,results):
    p_pri=sum((results.protected==False) & (results.target==1))/sum(results.target==1)
    p_pro=sum((results.protected==True) & (results.target==1))/sum(results.target==1)
    N_pri=round(C*p_pri)#int(C*p_pri)
    N_pro=round(C*p_pro)#int(C*p_pro)
    if N_pri>sum(results.protected==False):
        N_pro+=N_pri-sum(results.protected==False)
        N_pri=sum(results.protected==False)
    if N_pro>sum(results.protected==True):
        N_pri+=N_pro-sum(results.protected==True)
        N_pro=sum(results.protected==True)
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
    preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
    return preds

def assign_label_eo_corr(C,results):
    p_list = [i for i in range(1, 101)]
    preds={}
    tpr_pro, tpr_pri=[],[]
    for p in p_list:
        preds[p]=pd.Series(assign_label(C,p, results), index=results.index) 
        tpr_pri.append(mean(preds[p][(results.protected==False) & (results.target==1)]))
        tpr_pro.append(mean(preds[p][(results.protected==True) & (results.target==1)]))
    closest_index=find_closest_index(tpr_pro,tpr_pri)
    return preds[p_list[closest_index]]


def assign_label_eo_corr_optimized(C, results):
    p_list = [i for i in range(1, 101)]
    preds = {}
    tpr_pro, tpr_pri = [], []
    protected_true = results.protected == True
    protected_false = results.protected == False
    target_true = results.target == 1
    for p in p_list:
        preds[p] = pd.Series(assign_label(C, p, results), index=results.index)
        tpr_pri.append(preds[p][protected_false & target_true].mean())
        tpr_pro.append(preds[p][protected_true & target_true].mean())
    closest_index = find_closest_index(tpr_pro, tpr_pri)
    return preds[p_list[closest_index]]

def find_closest_index(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    min_diff = float('inf')
    closest_index = -1

    for i in range(len(list1)):
        diff = abs(list1[i] - list2[i])
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    #print('The minumum difference is {}'.format(min_diff))
    return closest_index


# def assign_label_eo(C, results):
#     # Separate the results into protected and non-protected groups
#     results_pro = results[results.protected==True]
#     results_pri = results[results.protected==False]

#     # Calculate the number of positive instances in each group
#     p_pro = sum(results_pro.labels==1)
#     p_pri = sum(results_pri.labels==1)

#     # Calculate the number of instances to select from each group
#     N_pro = int(C * (p_pro / (p_pro + p_pri)))
#     N_pri = C - N_pro

#     # Get the scores and indices for the positive instances in each group
#     score_index_pairs_pro = [(score, index) for index, score in results_pro[results_pro.actual_labels==1].biased_scores.items()]
#     score_index_pairs_pri = [(score, index) for index, score in results_pri[results_pri.actual_labels==1].biased_scores.items()]

#     # Sort the list of tuples in descending order of scores
#     sorted_pairs_pro = sorted(score_index_pairs_pro, key=lambda x: x[0], reverse=True)
#     sorted_pairs_pri = sorted(score_index_pairs_pri, key=lambda x: x[0], reverse=True)

#     # Select the top N instances from each group
#     indices_pro = [t[1] for t in sorted_pairs_pro[0:N_pro]]
#     indices_pri = [t[1] for t in sorted_pairs_pri[0:N_pri]]

#     # Assign labels based on the selected indices
#     preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]

#     return preds

def assign_label_fair(C,results):
    
    N_pri=int(C*(100-p)/100)
    N_pro=int(C*p/100)
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
    preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
    return preds


# #capacity problem
# def assign_label(C,p,results):
#     N_pri=int(C*(1-p))
#     N_pro=int(C*p)
#     if N_pri>sum(results.protected==False):
#         N_pro+=N_pri-sum(results.protected==False)
#         N_pri=sum(results.protected==False)
#     if N_pro>sum(results.protected==True):
#         N_pri+=N_pro-sum(results.protected==True)
#         N_pro=sum(results.protected==True)
#     if N_pro>sum(results.protected==True) and N_pri>sum(results.protected==False):
#         print('Error: capacity is higher than the number of instances in the dataset')
#         return
#     scores_pri=results[results.protected==False].biased_scores
#     score_index_pairs_pri= [(score, index) for index, score in scores_pri.items()]
#     # Sort the list of tuples in descending order of scores
#     sorted_pairs_pri = sorted(score_index_pairs_pri, key=lambda x: x[0], reverse=True)
#     indices_pri = [t[1] for t in sorted_pairs_pri[0:N_pri]]
#     scores_pro=results[results.protected==True].biased_scores
#     score_index_pairs_pro= [(score, index) for index, score in scores_pro.items()]
#     # Sort the list of tuples in descending order of scores
#     sorted_pairs_pro = sorted(score_index_pairs_pro, key=lambda x: x[0], reverse=True)
#     indices_pro = [t[1] for t in sorted_pairs_pro[0:N_pro]]
#     preds = [1 if i in indices_pro or i in indices_pri else 0 for i in results.index]
#     return preds

#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def run_tf_model(X_train, X_test, y_train, y_test):
    # Define the model architecture
    tf.reset_default_graph()
    num_epochs = 50
    batch_size = 128
    classifier_num_hidden_units = 200
    model = Sequential()
    model.add(Dense(classifier_num_hidden_units, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Fit the model on training data
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)
    # Predict scores and labels on test data
    test_scores = model.predict(X_test)
    test_labels = (test_scores > 0.5).astype(int)  # Assuming binary classification, adjust threshold if needed
    return test_labels, test_scores, model



def fairlearn_to(X_train, X_test, y_train, sens_var,  biased_model, metric):
    TO = ThresholdOptimizer(estimator=biased_model,constraints=metric, predict_method='predict', prefit=True)
    TO.fit(X_train, y_train, sensitive_features=X_train[sens_var])
    to_preds=TO.predict(X_test, sensitive_features=X_test[sens_var]) 
    return to_preds                                                                     

    
def run_constraints(X,y, sens_var, sensitive_value, metrics):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_tf_model(X_train, X_test, y_train, y_test)
    print('Postprocess:')
    for metric in metrics:
        test_results['to_' + metric]=fairlearn_to(X_train, X_test, y_train, sens_var,  biased_model, metric)
    return test_results




#%% old functions
#%%
from aif360.metrics import BinaryLabelDatasetMetric
def preprocess_lfr(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    unprivileged_groups = [{sens_var: sensitive_value}]
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    privileged_groups = [{sens_var: nonsensitive_value }] 
    lfr=LFR(unprivileged_groups, privileged_groups, k=10, Ax=0.1, Ay=1.0, Az=10.0, verbose=0).fit(train_data, maxiter=10000) #settings from the demo of AIF360! ONly change Az from 10 to 20
    #lfr=LFR(unprivileged_groups, privileged_groups, k = 10).fit(train_data)
    transf_train,transf_test = lfr.transform(train_data), lfr.transform(test_data)
    #convert to dataframe
    pandas_transf_train, dataset_info=transf_train.convert_to_dataframe()
    pandas_transf_test,_=transf_test.convert_to_dataframe()
    pandas_transf_train, pandas_transf_test = pandas_transf_train.drop(columns=['target']),pandas_transf_test.drop(columns=['target'])
    pandas_transf_train.index, pandas_transf_test.index=X_train.index, X_test.index
    return pandas_transf_train, pandas_transf_test
 

def preprocess_rw(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    unprivileged_groups = [{sens_var: sensitive_value}]
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    privileged_groups = [{sens_var: nonsensitive_value }] 
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    RW.fit(train_data)
    transf_train, transf_test = RW.transform(train_data), RW.transform(test_data)
    #convert to dataframe
    pandas_transf_train, dataset_info=transf_train.convert_to_dataframe()
    pandas_transf_test,_=transf_test.convert_to_dataframe()
    pandas_transf_train, pandas_transf_test = pandas_transf_train.drop(columns=['target']),pandas_transf_test.drop(columns=['target'])
    return pandas_transf_train, pandas_transf_test,dataset_info


def preprocess_dir(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    unprivileged_groups = [{sens_var: sensitive_value}]
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    privileged_groups = [{sens_var: nonsensitive_value }] 
    di=DisparateImpactRemover(repair_level=1)
    di.fit(train_data)
    transf_train,transf_test = di.fit_transform(train_data), di.fit_transform(test_data)
    #convert to dataframe
    pandas_transf_train, dataset_info=transf_train.convert_to_dataframe()
    pandas_transf_test,_=transf_test.convert_to_dataframe()
    pandas_transf_train, pandas_transf_test = pandas_transf_train.drop(columns=['target']),pandas_transf_test.drop(columns=['target'])
    pandas_transf_train.index, pandas_transf_test.index=X_train.index, X_test.index
    return pandas_transf_train, pandas_transf_test

#does not work! internal error
def inprocess_pr(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    PR = PrejudiceRemover(sensitive_attr=sens_var,eta=25).fit(train_data)
    PR_pred = PR.predict(test_data).labels #does not work! (error in pr)
    PR_pred = PR.predict(test_data).scores 
    return PR_pred, PR_scores


def inprocess_MFC(X_train, X_test, y_train, y_test, sens_var):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    MFC = MetaFairClassifier(sensitive_attr=sens_var, type="fdr").fit(train_data)
    MFC_preds = MFC.predict(test_data).labels #does not work! (error in pr)
    MFC_scores = MFC.predict(test_data).scores 
    return MFC_preds, MFC_scores
    
    
def inprocess_adv(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    tf.reset_default_graph()
    sess_adv = tf.Session()
    privileged_groups = [{sens_var: nonsensitive_value}]
    unprivileged_groups = [{sens_var:sensitive_value }]
    adv_model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                 privileged_groups=privileged_groups,
                 scope_name='adv',
                 debias=True,
                 sess=sess_adv,
                 adversary_loss_weight=0.3,
                 seed=111).fit(train_data)

    adv_preds = adv_model.predict(test_data).labels
    adv_scores = adv_model.predict(test_data).scores
    return adv_preds, adv_scores

def postprocess_roc(X_train, X_test, y_train, y_test, sens_var, sensitive_value, biased_model, X_train_model=None, X_test_model=None):
    if X_train_model is None: #when we dont use unaware models, this is just the same
        X_train_model, X_test_model=X_train, X_test
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    train_data_pred, test_data_pred=train_data.copy(),test_data.copy()
    train_data_pred.scores,test_data_pred.scores  = biased_model.predict(X_train_model), biased_model.predict(X_test_model)
    train_data_pred.labels, test_data_pred.labels = (biased_model.predict(X_train_model) > 0.5).astype(int) , (biased_model.predict(X_test_model) > 0.5).astype(int) 
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    privileged_groups = [{sens_var: nonsensitive_value}]
    unprivileged_groups = [{sens_var:sensitive_value }]
    ROC=RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups,metric_name='Statistical parity difference')
    #creating the dataframes, printing the metrics and visaling in one iteration (because the frames get altered)
    clf_roc=ROC.fit(train_data, train_data_pred)
    roc_preds=clf_roc.predict(test_data_pred).labels
    roc_scores=clf_roc.predict(test_data_pred).scores
    return roc_preds, roc_scores

def postprocess_eq(X_train, X_test, y_train, y_test, sens_var, sensitive_value, biased_model, X_train_model=None, X_test_model=None):
    if X_train_model is None: #when we dont use unaware models, this is just the same
        X_train_model, X_test_model=X_train, X_test
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    train_data_pred, test_data_pred=train_data.copy(),test_data.copy()
    train_data_pred.scores,test_data_pred.scores  = biased_model.predict(X_train_model), biased_model.predict(X_test_model)
    train_data_pred.labels, test_data_pred.labels = (biased_model.predict(X_train_model) > 0.5).astype(int) , (biased_model.predict(X_test_model) > 0.5).astype(int) 
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    privileged_groups = [{sens_var: nonsensitive_value}]
    unprivileged_groups = [{sens_var:sensitive_value }]
    EQO= EQ(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups)
    EQO.fit(train_data, train_data_pred)
    eqo_preds=EQO.predict(test_data_pred).labels
    eqo_scores=EQO.predict(test_data_pred).scores
    return eqo_preds, eqo_scores


def postprocess_to(X_train, X_test, y_train, sens_var,  biased_model,X_train_model=None, X_test_model=None):
    if X_train_model is None: #when we dont use unaware models, this is just the same
        X_train_model, X_test_model=X_train, X_test
    TO = ThresholdOptimizer(estimator=biased_model,constraints="demographic_parity", predict_method='predict', prefit=True)
    TO.fit(X_train_model, y_train, sensitive_features=X_train[sens_var])
    to_preds=TO.predict(X_test_model, sensitive_features=X_test[sens_var]) 
    return to_preds                                                                     

def run_biased_model(X_train, X_test, y_train, y_test, sens_var, sensitive_value):
    train_data = BinaryLabelDataset(df=X_train.assign(target = y_train), label_names=['target'], protected_attribute_names=[sens_var])
    test_data = BinaryLabelDataset(df=X_test.assign(target=y_test), label_names=['target'], protected_attribute_names=[sens_var])
    tf.reset_default_graph()
    nonsensitive_value= np.delete(X_train[sens_var].unique(), np.where(X_train[sens_var].unique()== sensitive_value))[0]
    sess_adv = tf.Session()
    privileged_groups = [{sens_var: nonsensitive_value}]
    unprivileged_groups = [{sens_var:sensitive_value }]
    biased_model = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                 privileged_groups=privileged_groups,
                 scope_name='model',
                 debias=False,
                 sess=sess_adv,
                 adversary_loss_weight=0.3,
                 seed=111).fit(train_data)
    biased_preds = biased_model.predict(test_data).labels
    biased_scores = biased_model.predict(test_data).scores
    return biased_preds, biased_scores, biased_model
    
def run_all(X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_tf_model(X_train, X_test, y_train, y_test)
    print('Postprocess:')
    test_results['roc_preds'], test_results['roc_scores']=postprocess_roc(X_train, X_test, y_train, y_test, sens_var, sensitive_value,biased_model)
    test_results['eq_preds'], test_results['eq_scores']=postprocess_eq(X_train, X_test, y_train, y_test, sens_var, sensitive_value, biased_model)
    test_results['to_preds']=postprocess_to(X_train, X_test, y_train, sens_var,  biased_model)
    print('LFR')
    lfr_train, lfr_test=preprocess_lfr(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['lfr_preds'], test_results['lfr_scores'],_=run_tf_model(lfr_train, lfr_test, y_train, y_test)
    #test_results['lfr_preds'], test_results['lfr_scores'],_=run_tf_model(lfr_train.drop(columns=[sens_var]), lfr_test.drop(columns=[sens_var]), y_train, y_test)
    print('DIR')
    dir_train, dir_test=preprocess_dir(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['dir_preds'], test_results['dir_scores'],_=run_tf_model(dir_train, dir_test, y_train, y_test)
    test_results['adv_preds'], test_results['adv_scores']=inprocess_adv(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['mfc_preds'], test_results['mfc_scores']=inprocess_MFC(X_train, X_test, y_train, y_test, sens_var)
    return test_results

def run_all_unaware(X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_tf_model(X_train.drop(columns=[sens_var]), X_test.drop(columns=[sens_var]), y_train, y_test)
    print('Postprocess:')
    test_results['roc_preds'], test_results['roc_scores']=postprocess_roc(X_train, X_test, y_train, y_test, sens_var, sensitive_value,biased_model,X_train.drop(columns=[sens_var]), X_test.drop(columns=[sens_var]))
    test_results['eq_preds'], test_results['eq_scores']=postprocess_eq(X_train, X_test, y_train, y_test, sens_var, sensitive_value, biased_model,X_train.drop(columns=[sens_var]), X_test.drop(columns=[sens_var]))
    test_results['to_preds']=postprocess_to(X_train, X_test, y_train, sens_var,  biased_model, X_train.drop(columns=[sens_var]), X_test.drop(columns=[sens_var]),)
    print('LFR')
    lfr_train, lfr_test=preprocess_lfr(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['lfr_preds'], test_results['lfr_scores'],_=run_tf_model(lfr_train.drop(columns=[sens_var]), lfr_test.drop(columns=[sens_var]), y_train, y_test)
    print('DIR')
    dir_train, dir_test=preprocess_dir(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['dir_preds'], test_results['dir_scores'],_=run_tf_model(dir_train.drop(columns=[sens_var]), dir_test.drop(columns=[sens_var]), y_train, y_test)
    test_results['adv_preds'], test_results['adv_scores']=inprocess_adv(X_train, X_test, y_train, y_test, sens_var, sensitive_value)
    test_results['mfc_preds'], test_results['mfc_scores']=inprocess_MFC(X_train, X_test, y_train, y_test, sens_var)
    return test_results

#%% 
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def run_tf_model(X_train, X_test, y_train, y_test):
    # Define the model architecture
    tf.reset_default_graph()
    num_epochs = 50
    batch_size = 128
    classifier_num_hidden_units = 200
    model = Sequential()
    model.add(Dense(classifier_num_hidden_units, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Fit the model on training data
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)
    # Predict scores and labels on test data
    test_scores = model.predict(X_test)
    test_labels = (test_scores > 0.5).astype(int)  # Assuming binary classification, adjust threshold if needed
    return test_labels, test_scores, model


#%% visualizations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
def plot_scores_bars(dataset, mitigation_method, test_results, test_results_preds, save=False, visualize=False):
    bin_edges = np.linspace(0, 1, 11)
    hist_privileged,_ = np.histogram(test_results[test_results.protected==False].biased_scores, bins=bin_edges)
    hist_protected,_ = np.histogram(test_results[test_results.protected==True].biased_scores, bins=bin_edges)
    hist1,_ = np.histogram(test_results[( test_results_preds==0)&(test_results.protected==False)].biased_scores, bins=bin_edges)
    hist2,_ = np.histogram(test_results[( test_results_preds==1)&(test_results.protected==False)].biased_scores, bins=bin_edges)
    hist3,_ = np.histogram(test_results[( test_results_preds==0)&(test_results.protected==True)].biased_scores, bins=bin_edges)
    hist4,_ = np.histogram(test_results[( test_results_preds==1)&(test_results.protected==True)].biased_scores, bins=bin_edges)
    plt.bar(bin_edges[0:10], height=np.where(hist_privileged != 0, hist1 / hist_privileged, 0),color='red',width=0.03, alpha=0.7, label='Negative')
    plt.bar(bin_edges[0:10], height=np.where(hist_privileged != 0, hist2 / hist_privileged, 0), color='green', width=0.03, alpha=0.7, label = 'Positive', bottom=np.where(hist_privileged != 0, hist1 / hist_privileged, 0))
    #plt.bar(bin_edges[0:10], height=hist3/hist_women,color='red',width=0.035, hatch='x', alpha=0.7, label = 'Women (negative change)')
    #plt.bar(bin_edges[0:10], height=hist4/hist_women, color='green', width=0.035, hatch = 'x', alpha=0.7, label='Women (positive change)')
    plt.bar(bin_edges[0:10]+0.035, height=np.where(hist_protected != 0, hist3 / hist_protected, 0), hatch='.',color='red',width=0.03,  alpha=0.7)
    plt.bar(bin_edges[0:10]+0.035, height=np.where(hist_protected != 0, hist4 / hist_protected, 0), hatch='.', color='green', width=0.03, alpha=0.7, bottom=np.where(hist_protected != 0, hist3 / hist_protected, 0))
    legend_patches = [
    Patch(facecolor='white', edgecolor='black', hatch=' ', label='Privileged group'),
    Patch(facecolor='white', edgecolor='black', hatch='.', label='Protected group'),]
    plt.xticks(bin_edges)
    plt.ylabel('Label (in %)')
    plt.ylim([0,1.1])
    plt.xlabel('Prediction score')
    plt.title(mitigation_method)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + legend_patches, labels + ['Privileged group', 'Protected group'], loc='lower right')
    if save==True:
        file_path='C:\\Users\\sgoethals\\Dropbox\\PC (2)\\Documents\\Research\\Transparency_fairness\\Critique on current bias mitigation benchmarks\\Bias_mitigation_code\\Bias-Mitigation\\Figures\\Score distributions\\'+ dataset+ '\\' + mitigation_method + '.png'
        plt.savefig(file_path, bbox_inches='tight') 
    if visualize==True:   
        plt.show()
    else:
        plt.close()


#%% print metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
def print_metrics(results,X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    score_lists={'Biased model': results.biased_scores, 'LFR': results.lfr_scores,'DIR': results.dir_scores, 'ADV': results.adv_scores, 'MFC': results.mfc_scores,'ROC': results.roc_scores,'EQ': results.eq_scores , 'TO': results.biased_scores}
    for method, scores in score_lists.items():
        print('AUC of {}:{:.3f}'.format(method, roc_auc_score(y_test, scores)))
        print('AUC for the minority group of {}:{:.3f}'.format(method, roc_auc_score(y_test[X_test[sens_var]==sensitive_value], scores[X_test[sens_var]==sensitive_value])))
        print('AUC for the majority group of {}:{:.3f}'.format(method, roc_auc_score(y_test[X_test[sens_var]!=sensitive_value], scores[X_test[sens_var]!=sensitive_value])))
    pred_lists={'Biased model': results.biased_preds, 'LFR': results.lfr_preds,'DIR': results.dir_preds, 'ADV': results.adv_preds, 'MFC': results.mfc_preds,'ROC': results.roc_preds,'EQ': results.eq_preds, 'TO': results.to_preds}
    print(' ')
    for method, preds in pred_lists.items():
        conf_matrix_all = confusion_matrix(y_test, preds)
        conf_matrix_minority = confusion_matrix(y_test[X_test[sens_var]==sensitive_value], preds[X_test[sens_var]==sensitive_value])
        conf_matrix_majority = confusion_matrix(y_test[X_test[sens_var]!=sensitive_value], preds[X_test[sens_var]!=sensitive_value])
        print('Accuracy of {}:{:.3f}'.format(method, ((conf_matrix_all[1,1]+conf_matrix_all[0,0])/np.sum(conf_matrix_all))))
        print('Accuracy of minority group {}:{:.3f}'.format(method, ((conf_matrix_minority[1,1]+conf_matrix_minority[0,0])/np.sum(conf_matrix_minority))))
        print('Accuracy of majority group {}:{:.3f}'.format(method, ((conf_matrix_majority[1,1]+conf_matrix_majority[0,0])/np.sum(conf_matrix_majority))))
        print('PR of {}:{:.3f}'.format(method, ((conf_matrix_all[1,1]+conf_matrix_all[0,1])/np.sum(conf_matrix_all))))
        print('SPD {}:{:.3f}'.format(method, (((conf_matrix_minority[1,1]+conf_matrix_minority[0,1])/np.sum(conf_matrix_minority))-((conf_matrix_majority[1,1]+conf_matrix_majority[0,1])/np.sum(conf_matrix_majority)))))
        print('PR of minority group {}:{:.3f}'.format(method, ((conf_matrix_minority[1,1]+conf_matrix_minority[0,1])/np.sum(conf_matrix_minority))))
        print('PR of majority group {}:{:.3f}'.format(method, ((conf_matrix_majority[1,1]+conf_matrix_majority[0,1])/np.sum(conf_matrix_majority))))
        print('TPR of {}:{:.3f}'.format(method, (conf_matrix_all[1,1]/(conf_matrix_all[1,1]+conf_matrix_all[0,1]))))
        print('EOD {}:{:.3f}'.format(method, ((conf_matrix_minority[1,1]/(conf_matrix_minority[1,1]+conf_matrix_minority[0,1]))-(conf_matrix_majority[1,1]/(conf_matrix_majority[1,1]+conf_matrix_majority[0,1])))))
        print('TPR of minority group {}:{:.3f}'.format(method, (conf_matrix_minority[1,1]/(conf_matrix_minority[1,1]+conf_matrix_minority[0,1]))))
        print('TPR of majority group {}:{:.3f}'.format(method, (conf_matrix_majority[1,1]/(conf_matrix_majority[1,1]+conf_matrix_majority[0,1]))))
        print(' ')

#create dictionary with results:

def create_dict_results(dataset, results):
    dict={}
    dict[dataset]={}
    for metriek in ['AUC' , '$AUC^{pro} , AUC^{pri}$', 'ACC', 'SPD', 'EOD' , 'PR']:
        dict[dataset][metriek]={}
    score_lists={'Biased model': results.biased_scores, 'LFR': results.lfr_scores,'DIR': results.dir_scores, 'ADV': results.adv_scores, 'MFC': results.mfc_scores,'ROC': results.roc_scores,'EOP': results.eq_scores , 'TO': results.biased_scores}
    for method, scores in score_lists.items():
        auc=round(roc_auc_score(results.target, scores), 3)
        dict[dataset]['AUC'][method]=auc
        auc_min=round(roc_auc_score(results[results.protected==True].target, scores[results.protected==True]),3)
        auc_maj=round(roc_auc_score(results[results.protected==False].target, scores[results.protected==False]),3)
        dict[dataset]['$AUC^{pro} , AUC^{pri}$'][method]=auc_min, auc_maj
    pred_lists={'Biased model': results.biased_preds, 'LFR': results.lfr_preds,'DIR': results.dir_preds, 'ADV': results.adv_preds, 'MFC': results.mfc_preds,'ROC': results.roc_preds,'EOP': results.eq_preds, 'TO': results.to_preds}
    for method, preds in pred_lists.items():
        conf_matrix_all = confusion_matrix(results.target, preds)
        conf_matrix_minority = confusion_matrix(results[results.protected==True].target, preds[results.protected==True])
        conf_matrix_majority = confusion_matrix(results[results.protected==False].target, preds[results.protected==False])
        acc=round(((conf_matrix_all[1,1]+conf_matrix_all[0,0])/np.sum(conf_matrix_all)), 3)
        dict[dataset]['ACC'][method]=acc
        pr=round(((conf_matrix_all[1,1]+conf_matrix_all[0,1])/np.sum(conf_matrix_all)), 3)
        dict[dataset]['PR'][method]=pr
        spd=round((((conf_matrix_minority[1,1]+conf_matrix_minority[0,1])/np.sum(conf_matrix_minority))-((conf_matrix_majority[1,1]+conf_matrix_majority[0,1])/np.sum(conf_matrix_majority))), 3)
        dict[dataset]['SPD'][method]=spd
        eod=round(((conf_matrix_minority[1,1]/(conf_matrix_minority[1,1]+conf_matrix_minority[0,1]))-(conf_matrix_majority[1,1]/(conf_matrix_majority[1,1]+conf_matrix_majority[0,1]))), 3)
        dict[dataset]['EOD'][method]=eod
    return dict


#%% old code



#does not work! internal error



