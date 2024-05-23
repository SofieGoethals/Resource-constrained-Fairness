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
from functions_general import *
from tqdm import tqdm

# def assign_label(C, p, results):
#     N_pri = round(C * (100 - p) / 100)
#     N_pro = round(C * p / 100)
    
#     if N_pri > np.sum(results.protected == False):
#         N_pro += N_pri - np.sum(results.protected == False)
#         N_pri = np.sum(results.protected == False)
        
#     if N_pro > np.sum(results.protected == True):
#         N_pri += N_pro - np.sum(results.protected == True)
#         N_pro = np.sum(results.protected == True)
        
#     if N_pro > np.sum(results.protected == True) and N_pri > np.sum(results.protected == False):
#         print('Error: capacity is higher than the number of instances in the dataset')
#         return
    
#     scores_pri = results[results.protected == False].biased_scores.values
#     indices_pri = np.argsort(scores_pri)[::-1][:N_pri]
    
#     scores_pro = results[results.protected == True].biased_scores.values
#     indices_pro = np.argsort(scores_pro)[::-1][:N_pro]
    
#     preds = np.zeros(len(results))
#     preds[indices_pri] = 1
#     preds[indices_pro] = 1
#     return preds.tolist()





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


### NEW
import xgboost as xgb

def run_xgb_model(X_train, X_test, y_train, y_test):
    # Convert the dataset into an optimized data structure called DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Basic hyperparameters
    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
    }
    
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

def run_constraints_mlp(X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,  stratify=pd.concat([X[sens_var], y], axis=1), random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,  stratify=pd.concat([X_test[sens_var], y_test], axis=1), random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_tf_model(X_train, X_test, y_train, y_test)
    test_results['AUC'], test_results['AUC_priv'], test_results['AUC_prot']=roc_auc_score(test_results.target,test_results.biased_scores), roc_auc_score(test_results[test_results.protected==False].target,test_results[test_results.protected==False].biased_scores), roc_auc_score(test_results[test_results.protected==True].target,test_results[test_results.protected==True].biased_scores)
    #validation set (use the same model)
    val_results=pd.DataFrame()#X_test.copy()
    val_results['protected']=X_val[sens_var]==sensitive_value
    val_results = val_results.assign(target = y_val)
    val_results['biased_scores'] = biased_model.predict(X_val)
    val_results['biased_preds']= (biased_model.predict(X_val) > 0.5).astype(int) 
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


def run_constraints_validation_unaware(X,y, sens_var, sensitive_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,  stratify=pd.concat([X[sens_var], y], axis=1), random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,  stratify=pd.concat([X_test[sens_var], y_test], axis=1), random_state=0)
    test_results=pd.DataFrame()#X_test.copy()
    test_results['protected']=X_test[sens_var]==sensitive_value
    test_results = test_results.assign(target = y_test)
    print('Run biased model')
    test_results['biased_preds'], test_results['biased_scores'], biased_model=run_tf_model(X_train.drop(columns=[sens_var]), X_test.drop(columns=[sens_var]), y_train, y_test)
    test_results['AUC'], test_results['AUC_priv'], test_results['AUC_prot']=roc_auc_score(test_results.target,test_results.biased_scores), roc_auc_score(test_results[test_results.protected==False].target,test_results[test_results.protected==False].biased_scores), roc_auc_score(test_results[test_results.protected==True].target,test_results[test_results.protected==True].biased_scores)
    #validation set (use the same model)
    val_results=pd.DataFrame()#X_test.copy()
    val_results['protected']=X_val[sens_var]==sensitive_value
    val_results = val_results.assign(target = y_val)
    val_results['biased_scores'] = biased_model.predict(X_val.drop(columns=[sens_var]))
    val_results['biased_preds']= (biased_model.predict(X_val.drop(columns=[sens_var]),) > 0.5).astype(int) 
    val_results['AUC'], val_results['AUC_priv'], val_results['AUC_prot']=roc_auc_score(val_results.target,val_results.biased_scores), roc_auc_score(val_results[val_results.protected==False].target,val_results[val_results.protected==False].biased_scores), roc_auc_score(val_results[val_results.protected==True].target,val_results[val_results.protected==True].biased_scores)
    print('The AUC of the biased model (validation set) is:', roc_auc_score(val_results.target,val_results.biased_scores))
    print('The AUC of the biased model (test set) is:', roc_auc_score(test_results.target,test_results.biased_scores))
    print('The AUC of the biased model for the protected group (validation set) is:', roc_auc_score(val_results[val_results.protected==True].target,val_results[val_results.protected==True].biased_scores))
    print('The AUC of the biased model for the privileged group (validation set) is:', roc_auc_score(val_results[val_results.protected==False].target,val_results[val_results.protected==False].biased_scores))
    print('The AUC of the biased model for the protected group (test set) is:', roc_auc_score(test_results[test_results.protected==True].target,test_results[test_results.protected==True].biased_scores))
    print('The AUC of the biased model for the privileged group (test set) is:', roc_auc_score(test_results[test_results.protected==False].target,test_results[test_results.protected==False].biased_scores))
    return test_results, val_results


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
""" 
def calculate_threshold_eo(C,results):
    p_list = [i for i in range(1, 101)]
    preds={}
    #tpr_pro, tpr_pri=[],[]
    for p in p_list:
        preds[p]=pd.Series(assign_label(C,p, results), index=results.index) 
        #tpr_pri.append(mean(preds[p][(results.protected==False) & (results.target==1)]))
        #tpr_pro.append(mean(preds[p][(results.protected==True) & (results.target==1)]))
    tpr_pro = [mean(preds[p][(results.protected==True) & (results.target==1)]) for p in p_list]
    tpr_pri = [mean(preds[p][(results.protected==False) & (results.target==1)]) for p in p_list]
    closest_index=find_closest_index(tpr_pro,tpr_pri)
    preds=preds[p_list[closest_index]]
    scores_pro=results[results.protected==True].biased_scores
    scores_pri=results[results.protected==False].biased_scores
    preds_pri = [1 if preds[i]==1 else 0 for i in results[results.protected==False].index]
    preds_pro = [1 if preds[i]==1 else 0 for i in results[results.protected==True].index]
    if sum(preds_pri)==0:
        threshold_pri=1
    else:
        threshold_pri=min(score for score, label in zip(scores_pri, preds_pri) if label == 1)
    if sum(preds_pro)==0:
        threshold_pro=1
    else:
        threshold_pro=min(score for score, label in zip(scores_pro, preds_pro) if label == 1)
    return pd.Series(preds, index=results.index), threshold_pri, threshold_pro """

def calculate_threshold_eo(C,results):
    p_list = [i for i in range(1, 101)]
    preds={}
    #tpr_pro, tpr_pri=[],[]
    for p in p_list:
        preds[p]=assign_label(C,p,results)#pd.Series(assign_label(C,p, results), index=results.index) 
        #tpr_pri.append(mean(preds[p][(results.protected==False) & (results.target==1)]))
        #tpr_pro.append(mean(preds[p][(results.protected==True) & (results.target==1)]))
    tpr_pro = [mean(preds[p][(results.protected==True) & (results.target==1)]) for p in p_list]
    tpr_pri = [mean(preds[p][(results.protected==False) & (results.target==1)]) for p in p_list]
    closest_index=find_closest_index(tpr_pro,tpr_pri)
    preds_final=preds[p_list[closest_index]]
    scores_pro=results[results.protected==True].biased_scores
    scores_pri=results[results.protected==False].biased_scores
    #preds_pri = [1 if preds[i]==1 else 0 for i in results[results.protected==False].index]
    #preds_pro = [1 if preds[i]==1 else 0 for i in results[results.protected==True].index]
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

def run_all_results_update(X,y, sens_var, sensitive_value, good_outcome , model ='xgb'):
    #Calculate DP
    #test_results, val_results=run_constraints_validation(X,y, sens_var, sensitive_value)
    print('Updated run')
    if model == 'xgb':
        test_results, val_results=run_constraints_xgb(X,y, sens_var, sensitive_value)
    elif model == 'mlp':
        test_results, val_results=run_constraints_mlp(X,y, sens_var, sensitive_value)
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
        #validation set
        preds_eo_val[C], threshold_pri, threshold_pro=calculate_threshold_eo(C, val_results)
    #test set
        preds_eo_test[C] = pd.Series([1 if (row.protected and row.biased_scores >= threshold_pro) or (not row.protected and row.biased_scores >= threshold_pri) else 0 for index, row in test_results.iterrows()], index=test_results.index)
        acc_eo_test.append(accuracy_score(test_results.target,preds_eo_test[C]))
        eod_unfair_test.append(preds_unfair_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_unfair_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
        eod_eo_test.append(preds_eo_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_eo_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
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
    return test_results,  test_metrics

def run_all_results(X,y, sens_var, sensitive_value, good_outcome , model ='xgb'):
    #Calculate DP
    #test_results, val_results=run_constraints_validation(X,y, sens_var, sensitive_value)
    if model == 'xgb':
        test_results, val_results=run_constraints_mlp(X,y, sens_var, sensitive_value)
    elif model == 'mlp':
        test_results, val_results=run_constraints_xgb(X,y, sens_var, sensitive_value)
    C_list=[i for i in np.linspace(start=1, stop=len(test_results), num=100, dtype=int)]
    preds_unfair_test, preds_unfair_val,preds_dp_test, preds_dp_val={},{},{},{}
    acc_unfair_test, acc_dp_test, acc_unfair_val, acc_dp_val=[],[],[],[]
    dd_unfair_test,dd_dp_test, dd_unfair_val, dd_dp_val=[],[],[],[]
    print('Calculate demographic parity')
    for C in tqdm(C_list):
        #validation set
        preds_dp_val[C], threshold_pri, threshold_pro=calculate_threshold_dp(C, val_results)
        preds_unfair_val[C]=pd.Series(assign_label_unfair(C,val_results), index=val_results.index)
        acc_unfair_val.append(accuracy_score(val_results.target,preds_unfair_val[C]))
        acc_dp_val.append(accuracy_score(val_results.target,preds_dp_val[C]))
        dd_unfair_val.append(preds_unfair_val[C][val_results.protected==False].mean()-preds_unfair_val[C][val_results.protected==True].mean())
        dd_dp_val.append(preds_dp_val[C][val_results.protected==False].mean()-preds_dp_val[C][val_results.protected==True].mean())
        #test set
        preds_unfair_test[C]=pd.Series(assign_label_unfair(C,test_results), index=test_results.index)
        preds_dp_test[C] = pd.Series([1 if (row.protected and row.biased_scores > threshold_pro) or (not row.protected and row.biased_scores > threshold_pri) else 0 for index, row in test_results.iterrows()], index=test_results.index)
        acc_unfair_test.append(accuracy_score(test_results.target,preds_unfair_test[C]))
        acc_dp_test.append(accuracy_score(test_results.target,preds_dp_test[C]))
        dd_unfair_test.append(preds_unfair_test[C][test_results.protected==False].mean()-preds_unfair_test[C][test_results.protected==True].mean())
        dd_dp_test.append(preds_dp_test[C][test_results.protected==False].mean()-preds_dp_test[C][test_results.protected==True].mean())
    print('Calculate equality of opportunity')
    preds_eo_test, preds_eo_val={},{}
    acc_eo_test, acc_eo_val=[],[]
    eod_unfair_val,eod_eo_val, eod_unfair_test, eod_eo_test=[],[],[],[]
    for C in tqdm(C_list):
        #validation set
        preds_eo_val[C], threshold_pri, threshold_pro=calculate_threshold_eo(C, val_results)
        acc_eo_val.append(accuracy_score(val_results.target,preds_eo_val[C]))
        eod_unfair_val.append(preds_unfair_val[C][(val_results.protected==False) & (val_results.target==True)].mean()-preds_unfair_val[C][(val_results.protected==True) & (val_results.target==True)].mean())
        eod_eo_val.append(preds_eo_val[C][(val_results.protected==False) & (val_results.target==True)].mean()-preds_eo_val[C][(val_results.protected==True) & (val_results.target==True)].mean())
        #test set
        preds_eo_test[C] = pd.Series([1 if (row.protected and row.biased_scores > threshold_pro) or (not row.protected and row.biased_scores > threshold_pri) else 0 for index, row in test_results.iterrows()], index=test_results.index)
        acc_eo_test.append(accuracy_score(test_results.target,preds_eo_test[C]))
        eod_unfair_test.append(preds_unfair_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_unfair_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
        eod_eo_test.append(preds_eo_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_eo_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
    print('Calculate precision')
    prec_unfair_test, prec_dp_test, prec_eo_test,prec_unfair_val, prec_dp_val, prec_eo_val=[],[],[],[],[],[]
    for C in tqdm(C_list):
        for prec,pred,res in zip([prec_unfair_test, prec_dp_test, prec_eo_test, prec_unfair_val, prec_dp_val, prec_eo_val],[preds_unfair_test, preds_dp_test, preds_eo_test, preds_unfair_val, preds_dp_val, preds_eo_val],[test_results, test_results, test_results, val_results, val_results, val_results]):
            if sum(pred[C] == good_outcome)==0:
                prec.append(1)
            elif sum(res[pred[C]==good_outcome].target==good_outcome)==0:
                prec.append(0)
            else:
                prec.append(res[pred[C]==1].target.value_counts(normalize=True)[good_outcome])
    print('Calculate recall')
    rec_unfair_test, rec_dp_test, rec_eo_test,rec_unfair_val, rec_dp_val, rec_eo_val=[],[],[],[],[],[]
    for C in tqdm(C_list):
        for rec,pred,res in zip([rec_unfair_test, rec_dp_test, rec_eo_test, rec_unfair_val, rec_dp_val, rec_eo_val],[preds_unfair_test, preds_dp_test, preds_eo_test, preds_unfair_val, preds_dp_val, preds_eo_val],[test_results, test_results, test_results, val_results, val_results, val_results]):
            if sum(res.target == good_outcome)==0:
                rec.append(1)
            elif sum(pred[C][res.target==good_outcome]==good_outcome)==0:
                rec.append(0)
            else:
                rec.append(pred[C][res.target==good_outcome].value_counts(normalize=True)[good_outcome])
    test_metrics={}
    test_metrics['acc_unfair'], test_metrics['acc_dp'], test_metrics['acc_eo'],test_metrics['prec_unfair'], test_metrics['prec_dp'], test_metrics['prec_eo'],test_metrics['rec_unfair'], test_metrics['rec_dp'], test_metrics['rec_eo'], test_metrics['dd_unfair'], test_metrics['dd_dp'],test_metrics['eod_unfair'], test_metrics['eod_eo'] = acc_unfair_test, acc_dp_test, acc_eo_test, prec_unfair_test, prec_dp_test, prec_eo_test, rec_unfair_test, rec_dp_test, rec_eo_test, dd_unfair_test, dd_dp_test, eod_unfair_test, eod_eo_test
    test_metrics['preds_unfair'], test_metrics['preds_dp'], test_metrics['preds_eo'] = preds_unfair_test, preds_dp_test, preds_eo_test
    val_metrics={}
    val_metrics['acc_unfair'], val_metrics['acc_dp'], val_metrics['acc_eo'],val_metrics['prec_unfair'], val_metrics['prec_dp'], val_metrics['prec_eo'],val_metrics['rec_unfair'], val_metrics['rec_dp'], val_metrics['rec_eo'], val_metrics['dd_unfair'], val_metrics['dd_dp'],val_metrics['eod_unfair'], val_metrics['eod_eo'] = acc_unfair_val, acc_dp_val, acc_eo_val, prec_unfair_val, prec_dp_val, prec_eo_val, rec_unfair_val, rec_dp_val, rec_eo_val, dd_unfair_val, dd_dp_val, eod_unfair_val, eod_eo_val                                                                                                                                                                                                                                                                               
    val_metrics['preds_unfair'], val_metrics['preds_dp'], val_metrics['preds_eo'] =preds_unfair_val, preds_dp_val, preds_eo_val
    return test_results, val_results, test_metrics, val_metrics


def run_all_vision(test_results, val_results, good_outcome):
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
        #validation set
        preds_eo_val[C], threshold_pri, threshold_pro=calculate_threshold_eo(C, val_results)
    #test set
        preds_eo_test[C] = pd.Series([1 if (row.protected and row.biased_scores > threshold_pro) or (not row.protected and row.biased_scores > threshold_pri) else 0 for index, row in test_results.iterrows()], index=test_results.index)
        acc_eo_test.append(accuracy_score(test_results.target,preds_eo_test[C]))
        eod_unfair_test.append(preds_unfair_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_unfair_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
        eod_eo_test.append(preds_eo_test[C][(test_results.protected==False) & (test_results.target==True)].mean()-preds_eo_test[C][(test_results.protected==True) & (test_results.target==True)].mean())
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

def run_trade_off_xgb(X,y, sens_var, sensitive_value, name, model ='xgb'):
    metrics=['demographic_parity', 'true_positive_rate_parity', 'equalized_odds']
    if model =='xgb':
        test_results, val_results=run_constraints_xgb(X,y, sens_var, sensitive_value)
    else:
        test_results, val_results=run_constraints_mlp(X,y, sens_var, sensitive_value)
    results=val_results
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
    print('Calculate EO')
    C_list=[i for i in np.linspace(start=1, stop=len(results), num=100, dtype=int)]
    preds_eo={}
    for C in C_list:
       preds_eo[C]=pd.Series(assign_label_eo_corr(C,results), index=results.index)
    print('Calculate precision')
    prec_diff_dp, prec_diff_eo=calculate_precision(results, preds_unfair, preds_dp, preds_eo, C_list, name, good_outcome=1)
    print('Maximum cost of fairness (precision) for DP: {} and EO {}'.format(max(prec_diff_dp), max(prec_diff_eo)))
    return  mean(prec_diff_dp), mean(prec_diff_eo)


#%%
def run_trade_off_validation(X,y, sens_var, sensitive_value, name):
    metrics=['demographic_parity', 'true_positive_rate_parity', 'equalized_odds']
    test_results, val_results=run_constraints_validation(X,y, sens_var, sensitive_value)
    results=val_results
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
    # plt.figure(facecolor=(1, 1, 1))
    # plt.plot(C_list, acc_unfair, color='red', label='ML model')
    # plt.plot(C_list,acc_dp, color='blue',label ='DP')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Selection rate (in %)')
    ticks = np.linspace(min_capacity, max_capacity, 11)
    percentage_ticks = [x/max_capacity * 100 for x in ticks]
    # plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    # plt.legend()
    # ax2 = plt.twinx()
    # ax2.plot(C_list, dd_unfair, color='red', linestyle='--', label='ML model')
    # ax2.plot(C_list, dd_dp, color='blue', linestyle='--', label='DP')
    # ax2.set_ylabel('Demographic disparity')
    # ax2.set_ylim([0,1])
    # plt.title(name + ': Selection rate - Demographic Parity')
    # plt.show()
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
    # plt.figure(facecolor=(1, 1, 1))
    # plt.xticks(ticks=ticks, labels=['{:.0f}'.format(x) for x in percentage_ticks])
    # plt.plot(C_list, acc_unfair, color='red', label='ML model')
    # plt.plot(C_list,acc_eo, color='blue',label ='EO')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Selection rate (in %)')
    # plt.legend()
    # plt.title(name + ': Selection rate - Equal Opportunity')
    # ax2 = plt.twinx()
    # ax2.plot(C_list, eod_unfair, color='red', linestyle='--', label='ML model')
    # ax2.plot(C_list, eod_eo, color='blue', linestyle='--', label='EO')
    # ax2.set_ylabel('Equal Opportunity Difference')
    # ax2.set_ylim([0,1])
    # plt.show()
    #plot together
    plt.figure(facecolor=(1, 1, 1))
    acc_diff_eo=[a - b for a, b in zip(acc_unfair, acc_eo)]
    acc_diff_dp=[a - b for a, b in zip(acc_unfair, acc_dp)]
    plt.plot(C_list,acc_diff_eo, color='lightblue', label='EO')
    plt.plot(C_list, acc_diff_dp, color='blue', label='DP')
    plt.ylabel('Cost of fairness \n ($\Delta$ in accuracy)')
    plt.xlabel('Selection rate (in %)')
    plt.legend()
    plt.ylim([-0.015, 0.25])
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
    return mean(acc_diff_dp), mean(acc_diff_eo), mean(prec_diff_dp), mean(prec_diff_eo), mean(recall_diff_dp), mean(recall_diff_eo)#,roc_auc_score(results[results.protected==True].target,results[results.protected==True].biased_scores),roc_auc_score(results[results.protected==False].target,results[results.protected==False].biased_scores)