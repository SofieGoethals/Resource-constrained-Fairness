
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
#import fairlearn
#from fairlearn.postprocessing import ThresholdOptimizer
from statistics import mean
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
#from functions_general import *
from tqdm import tqdm
import xgboost as xgb

#%% Assign and allocate functions



#%% run all 
def run_all_results_clean(test_results, val_results, good_outcome):
    C_list=[i for i in np.linspace(start=1, stop=len(test_results), num=100, dtype=int)]
    preds_unfair_test, preds_dp_test={},{}
    acc_unfair_test, acc_dp_test=[],[]
    dd_unfair_test,dd_dp_test=[],[]
    print('Calculate demographic parity')
    for C in tqdm(C_list):
        #validation set
        preds_dp_test[C], threshold_pri, threshold_pro=calculate_threshold_dp(C, test_results)
        preds_unfair_test[C]=pd.Series(assign_label_unfair(C,test_results), index=test_results.index)
        #preds_dp_test[C] = pd.Series([1 if (row.protected and row.biased_scores > threshold_pro) or (not row.protected and row.biased_scores > threshold_pri) else 0 for index, row in test_results.iterrows()], index=test_results.index)
        acc_unfair_test.append(accuracy_score(test_results.target,preds_unfair_test[C]))
        acc_dp_test.append(accuracy_score(test_results.target,preds_dp_test[C]))
        dd_unfair_test.append(preds_unfair_test[C][test_results.protected==False].mean()-preds_unfair_test[C][test_results.protected==True].mean())
        dd_dp_test.append(preds_dp_test[C][test_results.protected==False].mean()-preds_dp_test[C][test_results.protected==True].mean())
    print('Calculate equality of opportunity')
    preds_eo_test, preds_eo_val={},{}
    acc_eo_test=[]
    eod_unfair_test, eod_eo_test=[],[]
    for C in tqdm(C_list):
        preds_eo_val[C], allocation_pri, allocation_pro=calculate_allocation_eo(C, val_results)
        #test set
        num_scores_pri = round(allocation_pri * C)  # Calculate the number of scores to select
        highest_scores_pri = test_results[test_results.protected == False].nlargest(num_scores_pri, 'biased_scores')
        num_scores_pro = round(allocation_pro* C)  # Calculate the number of scores to select
        highest_scores_pro = test_results[test_results.protected == True].nlargest(num_scores_pro, 'biased_scores')
        preds_eo_test[C]=pd.Series([1 if i in highest_scores_pri.index or i in highest_scores_pro.index else 0 for i in test_results.index], index=test_results.index)
    print('Calculate precision')
    prec_unfair_test, prec_dp_test, prec_eo_test=[],[],[]
    for C in tqdm(C_list):
        for prec,pred,res in zip([prec_unfair_test, prec_dp_test, prec_eo_test],[preds_unfair_test, preds_dp_test, preds_eo_test],[test_results, test_results, test_results]):
            if sum(pred[C] == good_outcome)==0:
                prec.append(1)
            elif sum(res[pred[C]==good_outcome].target==good_outcome)==0:
                prec.append(0)
            else:
                prec.append(res[pred[C]==1].target.value_counts(normalize=True)[good_outcome])
    print('Calculate recall')
    rec_unfair_test, rec_dp_test, rec_eo_test=[],[],[]
    for C in tqdm(C_list):
        for rec,pred,res in zip([rec_unfair_test, rec_dp_test, rec_eo_test],[preds_unfair_test, preds_dp_test, preds_eo_test],[test_results, test_results, test_results]):
            if sum(res.target == good_outcome)==0:
                rec.append(1)
            elif sum(pred[C][res.target==good_outcome]==good_outcome)==0:
                rec.append(0)
            else:
                rec.append(pred[C][res.target==good_outcome].value_counts(normalize=True)[good_outcome])
    test_metrics={}
    test_metrics['acc_unfair'], test_metrics['acc_dp'], test_metrics['acc_eo'],test_metrics['prec_unfair'], test_metrics['prec_dp'], test_metrics['prec_eo'],test_metrics['rec_unfair'], test_metrics['rec_dp'], test_metrics['rec_eo'], test_metrics['dd_unfair'], test_metrics['dd_dp'],test_metrics['eod_unfair'], test_metrics['eod_eo'] = acc_unfair_test, acc_dp_test, acc_eo_test, prec_unfair_test, prec_dp_test, prec_eo_test, rec_unfair_test, rec_dp_test, rec_eo_test, dd_unfair_test, dd_dp_test, eod_unfair_test, eod_eo_test
    test_metrics['preds_unfair'], test_metrics['preds_dp'], test_metrics['preds_eo'] = preds_unfair_test, preds_dp_test, preds_eo_test
    return test_metrics


### modeling
def run_xgb_model(X_train, X_test, y_train, y_test):
    # Convert the dataset into an optimized data structure called DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # Basic hyperparameters
    params = {'max_depth': 6,'eta': 0.3,'objective': 'binary:logistic','eval_metric': 'logloss',}
    # Number of boosting rounds
    num_boost_round = 999
    # Perform cross-validation: update the params and num_boost_round based on CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'logloss'},
        early_stopping_rounds=10
    )
    # Update num_boost_round to best iteration
    num_boost_round = cv_results.shape[0]
    
    # Train the model with the optimal number of boosting rounds
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
    )
    # Predict the probabilities of the positive class
    test_scores = bst.predict(dtest)
    # Convert probabilities to labels
    test_labels = (test_scores > 0.5).astype(int)
    return test_labels, test_scores, bst

def run_constraints_xgb(X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,  stratify=pd.concat([X[sens_var], y], axis=1), random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,  stratify=pd.concat([X_test[sens_var], y_test], axis=1), random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_xgb_model(X_train, X_test, y_train, y_test)
    test_results['AUC'], test_results['AUC_priv'], test_results['AUC_prot']=roc_auc_score(test_results.target,test_results.biased_scores), roc_auc_score(test_results[test_results.protected==False].target,test_results[test_results.protected==False].biased_scores), roc_auc_score(test_results[test_results.protected==True].target,test_results[test_results.protected==True].biased_scores)
    #validation set (use the same model)
    val_results=pd.DataFrame()#X_test.copy()
    val_results['protected']=X_val[sens_var]==sensitive_value
    val_results = val_results.assign(target = y_val)
    val_results['biased_scores'] = biased_model.predict(xgb.DMatrix(X_val, label=y_val))
    val_results['biased_preds']= (biased_model.predict(xgb.DMatrix(X_val, label=y_val)) > 0.5).astype(int) 
    val_results['AUC'], val_results['AUC_priv'], val_results['AUC_prot']=roc_auc_score(val_results.target,val_results.biased_scores), roc_auc_score(val_results[val_results.protected==False].target,val_results[val_results.protected==False].biased_scores), roc_auc_score(val_results[val_results.protected==True].target,val_results[val_results.protected==True].biased_scores)
    print('The AUC of the biased model (validation set) is:', roc_auc_score(val_results.target,val_results.biased_scores))
    print('The AUC of the biased model (test set) is:', roc_auc_score(test_results.target,test_results.biased_scores))
    print('The AUC of the biased model for the protected group (validation set) is:', roc_auc_score(val_results[val_results.protected==True].target,val_results[val_results.protected==True].biased_scores))
    print('The AUC of the biased model for the privileged group (validation set) is:', roc_auc_score(val_results[val_results.protected==False].target,val_results[val_results.protected==False].biased_scores))
    print('The AUC of the biased model for the protected group (test set) is:', roc_auc_score(test_results[test_results.protected==True].target,test_results[test_results.protected==True].biased_scores))
    print('The AUC of the biased model for the privileged group (test set) is:', roc_auc_score(test_results[test_results.protected==False].target,test_results[test_results.protected==False].biased_scores))
    # print('Postprocess:')
    # for metric in metrics:
    #     test_results['to_' + metric]=fairlearn_to(X_train, X_test, y_train, sens_var,  biased_model, metric)
    # #repeat for validation set
    # print('Postprocess:')
    # for metric in metrics:
    #     val_results['to_' + metric]=fairlearn_to(X_train, X_val, y_train, sens_var,  biased_model, metric)
    return test_results, val_results

#%% utils 

def calculate_allocation_eo(C, results):
    p_list = range(1, 101)
    preds = {p: pd.Series(assign_label(C, p, results), index=results.index) for p in p_list}


    tpr_pro = [mean(preds[p][results.protected & results.target]) for p in p_list]
    tpr_pri = [mean(preds[p][~results.protected & results.target]) for p in p_list]

    closest_index = find_closest_index(tpr_pro, tpr_pri)
    preds_final = preds[p_list[closest_index]]


    preds_pri = preds_final.loc[~results.protected]
    preds_pro = preds_final.loc[results.protected]

    allocation_pri = preds_pri.value_counts().get(1, 0) / C
    allocation_pro = preds_pro.value_counts().get(1, 0) / C

    return pd.Series(preds_final, index=results.index), allocation_pri, allocation_pro

def assign_label_unfair(C,results):
    scores=results.biased_scores
    score_index_pairs= [(score, index) for index, score in scores.items()]
    # Sort the list of tuples in descending order of scores
    sorted_pairs= sorted(score_index_pairs, key=lambda x: x[0], reverse=True)
    indices = [t[1] for t in sorted_pairs[0:C]]
    preds = [1 if i in indices else 0 for i in results.index]
    return preds

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

def calculate_threshold_dp(C, results):
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
    preds_final = pd.Series([1 if i in indices_pro or i in indices_pri else 0 for i in results.index], index=results.index)
    #preds_pri = [1 if i in indices_pri else 0 for i in results[results.protected==False].index]
    #preds_pro = [1 if i in indices_pro else 0 for i in results[results.protected==True].index]
    preds_pri = preds_final[results.protected==False]
    preds_pro = preds_final[results.protected==True]
    if sum(preds_pri)==0:
        threshold_pri=1
    else:
        threshold_pri=min(score for score, label in zip(scores_pri, preds_pri) if label == 1)
    if sum(preds_pro)==0:
        threshold_pro=1
    else:
        threshold_pro=min(score for score, label in zip(scores_pro, preds_pro) if label == 1)
    return pd.Series(preds_final, index=results.index), threshold_pri, threshold_pro

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
