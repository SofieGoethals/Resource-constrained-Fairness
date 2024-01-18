
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

#%% new functions



#capacity problem
def assign_label(C,p,results):
    N_pri=int(C*(1-p))
    N_pro=int(C*p)
    if N_pri>len(results.protected==False):
        N_pro+=N_pri-len(results.protected==False)
        N_pri=len(results.protected==False)
    if N_pro>len(results.protected==True):
        N_pri+=N_pro-len(results.protected==True)
        N_pro=len(results.protected==True)
    if N_pro>len(results.protected==True) and N_pri>len(results.protected==False):
        print('Error: capacity is higher than the number of instances in the dataset')
        return
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



