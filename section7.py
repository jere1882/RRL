from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.cluster import FeatureAgglomeration
from sklearn import datasets, cluster
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


""" FEATURE SELECTION AND DIMENSIONALITY REDUCTION """

def univariate_selection_f_classif(train="b234",test="b278"):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()
    scores = {}

############ USING FEATURE SELECTION
    for i in range(2,62,4): 

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", SelectKBest(f_classif, k=i)),
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "features selected= "+str(i)) 

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
        
        scores[i]=robust_auc

    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 

    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    scores[len(X.columns)]=robust_auc

    ########## PLOT ALL SCORES 

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Univarate feature selection. f_classif ; SelectKBest. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"f_classif_train="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    plt.title('Univarate feature selection. f_classif ; SelectKBest. ' + str(train) + ' testing in '+str(test))
    plt.savefig(save_folder+"f_classif_train="+train+"test="+test+"_scores.png")

    return (robust_auc,scores)



def univariate_selection_f_classif_full_experiment():

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    scores1= univariate_selection_f_classif(train="b278",test="b234")
    scores2= univariate_selection_f_classif(train="b278",test="b261")
    scores3= univariate_selection_f_classif(train="b234",test="b261")
    scores4= univariate_selection_f_classif(train="b234",test="b360")
    scores5= univariate_selection_f_classif(train="b261",test="b360")
    scores6= univariate_selection_f_classif(train="b261",test="b278")
    scores7= univariate_selection_f_classif(train="b360",test="b278")
    scores8= univariate_selection_f_classif(train="b360",test="b234")

    fig, ax = plt.subplots()
    ax.plot(list(scores1[1].keys()), list(scores1[1].values()),label="278->234") 
    ax.plot(list(scores2[1].keys()), list(scores2[1].values()),label="278->261") 
    ax.plot(list(scores3[1].keys()), list(scores3[1].values()),label="234->261") 
    ax.plot(list(scores4[1].keys()), list(scores4[1].values()),label="234->360") 
    ax.plot(list(scores5[1].keys()), list(scores5[1].values()),label="261->360") 
    ax.plot(list(scores6[1].keys()), list(scores6[1].values()),label="261->278") 
    ax.plot(list(scores7[1].keys()), list(scores7[1].values()),label="360->278") 
    ax.plot(list(scores8[1].keys()), list(scores8[1].values()),label="360->234") 
    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('Univarate feature selection. f_classif ; SelectKBest. ')
    plt.savefig(save_folder+"f_classif_global_scores.png")
    plt.show()


def univariate_selection_mutual_info_classif(train="b234",test="b278"):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()
    scores = {}

############ USING FEATURE SELECTION
    for i in range(2,62,4): 

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", SelectKBest(mutual_info_classif, k=i)),
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "features selected= "+str(i)) 

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
        
        scores[i]=robust_auc

    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 

    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    scores[len(X.columns)]=robust_auc

    ########## PLOT ALL SCORES 

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Univarate feature selection. mutual_info_classif ; SelectKBest. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"mutual_info_classif_train="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    plt.title('Univarate feature selection. f_classif ; SelectKBest. ' + str(train) + ' testing in '+str(test))
    plt.savefig(save_folder+"mutual_info_classif_train="+train+"test="+test+"_scores.png")

    return (robust_auc,scores)



def univariate_selection_mutual_info_classif_full_experiment():

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    scores1= univariate_selection_mutual_info_classif(train="b278",test="b234")
    scores2= univariate_selection_mutual_info_classif(train="b278",test="b261")
    scores3= univariate_selection_mutual_info_classif(train="b234",test="b261")
    scores4= univariate_selection_mutual_info_classif(train="b234",test="b360")
    scores5= univariate_selection_mutual_info_classif(train="b261",test="b360")
    scores6= univariate_selection_mutual_info_classif(train="b261",test="b278")
    scores7= univariate_selection_mutual_info_classif(train="b360",test="b278")
    scores8= univariate_selection_mutual_info_classif(train="b360",test="b234")

    fig, ax = plt.subplots()
    ax.plot(list(scores1[1].keys()), list(scores1[1].values()),label="278->234") 
    ax.plot(list(scores2[1].keys()), list(scores2[1].values()),label="278->261") 
    ax.plot(list(scores3[1].keys()), list(scores3[1].values()),label="234->261") 
    ax.plot(list(scores4[1].keys()), list(scores4[1].values()),label="234->360") 
    ax.plot(list(scores5[1].keys()), list(scores5[1].values()),label="261->360") 
    ax.plot(list(scores6[1].keys()), list(scores6[1].values()),label="261->278") 
    ax.plot(list(scores7[1].keys()), list(scores7[1].values()),label="360->278") 
    ax.plot(list(scores8[1].keys()), list(scores8[1].values()),label="360->234") 
    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('Univarate feature selection. mutual_info_classif ; SelectKBest. ')
    plt.savefig(save_folder+"mutual_info_classif_global_scores.png")
    plt.show()

def univariate_selection_chi2(test="b278",train="b234"):
    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()
    
    for i in range(2,55,4):

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("feature_selector", SelectKBest(chi2, k=i)),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "features selected= "+str(i)) 

    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Univarate feature selection. chi2 ; SelectKBest. ' + str(train) + ' testing in '+str(test))

    plt.show()

def plot_univariate_variable_importance(test="b278"):
    
    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    plt.figure(1)
    plt.clf()

    X,y = retrieve_tile(test,"full") 
    X_indices = np.arange(X.shape[-1])

    # f_classif
    selector = SelectKBest(f_classif, k=4)
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("feature_selector", selector )])
    clf.fit(X, y)

    f_scores = selector.scores_ / sum(selector.scores_)

    #mutual_info
    selector_mi = SelectKBest(mutual_info_classif, k=4)
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("feature_selector", selector_mi )])

    clf.fit(X, y)

    m_scores = selector_mi.scores_ / sum(selector_mi.scores_)

        
    #chi2
    c_selector = SelectKBest(chi2, k=4)
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("feature_selector", c_selector )])

    clf.fit(X, y)

    c_scores = c_selector.scores_ / sum(c_selector.scores_)

    #Svm
    #svm = LinearSVC(dual=False,max_iter=100000,C=1)
    #clf = Pipeline( 
    #[("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
    # ("scaler",StandardScaler()), 
    # ("svm", svm)])
    #clf.fit(X, y)

    #svm_weights = np.abs(svm.coef_) / np.abs(svm.coef_).sum(axis=1)


    plt.bar(X_indices , f_scores, width=.2,  label="Univariate score (ANOVA F-value) ")
    plt.bar(X_indices +0.2, m_scores , width=.2,  label="Univariate score (Mutual information)")
    plt.bar(X_indices +0.4, c_scores , width=.2,  label="Univariate scire (chi-squared)")
    #plt.bar(X_indices +0.6, svm_weights.tolist()[0] , width=.2,  label="abs(weight) SVM Linear (no FS)")

    plt.title("Comparing feature score in tile " + test)
    plt.xlabel('Feature number')
    plt.ylabel('Normalized score')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.savefig(save_folder+"test="+test+"_varaible_importance.png")


def plot_ml_importance():
    rate = "full"
    X,y=retrieve_tile("b234")

    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    X_indices = np.arange(X.shape[-1])
    importance = clf.feature_importances_
    plt.bar(X_indices , importance, width=.2)
    plt.xlabel('Feature number')
    plt.ylabel('Random forest importance score')
    plt.yticks(())
    plt.axis('tight')


    svc = LinearSVC(verbose=3, max_iter=10000, C=0.1, dual=False)
    clf2 = Pipeline([
    ("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
    ('scaler', StandardScaler()),
    ('clf', svc) ])

    clf2.fit(X,y)

    X_indices = np.arange(X.shape[-1])
    plt.bar(X_indices , np.abs(svc.coef_.tolist()[0]), width=.2)
    plt.xlabel('Feature number')
    plt.ylabel('Linear SVM abs weight')
    plt.yticks(())
    plt.axis('tight')
    plt.show()

    importance = clf.feature_importances_ / sum(clf.feature_importances_)
    plt.bar(X_indices , importance, width=.2,label="RF Gini")
    plt.bar(X_indices+0.2 , np.abs(svc.coef_.tolist()[0])/sum( np.abs(svc.coef_.tolist()[0])), width=.2,label="abs SVM weight")
    plt.xlabel('Feature number')
    plt.ylabel('Score (normalized)')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')

########################################### DIMENSIONALITY REDUCTION ############################################


def dimensionality_reduction_pca(train="b234",test="b278",whiten=False):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()
    scores = {}
    scores_pafr = {}

############ USING FEATURE SELECTION
    for i in range(2,62,3): 

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", PCA(n_components=i,whiten=whiten)),
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "features selected= "+str(i)) 

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
    
        recall_interpolated    = np.linspace(0, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
        pafr = max(precision_interpolated[recall_interpolated >= 0.9])   

        scores[i]=robust_auc
        scores_pafr[i]=pafr
    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=4,label=" no feature selection ") 

    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    recall_interpolated    = np.linspace(0, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
    pafr = max(precision_interpolated[recall_interpolated >= 0.9])   
#scores[len(X.columns)]=robust_auc

    ########## PLOT ALL SCORES 

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Select k best PCA features. ' + str(train) + ' testing in '+str(test))

    if whiten:
        plt.savefig(save_folder+"pcaW="+train+"test="+test+"_curves.png")
    else:
        plt.savefig(save_folder+"pca="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    plt.title('pca ; SelectKBest. ' + str(train) + ' testing in '+str(test))

    if whiten:
        plt.savefig(save_folder+"pcaW="+train+"test="+test+"_scores.png")
    else:
        plt.savefig(save_folder+"pca="+train+"test="+test+"_scores.png")


    fig, ax = plt.subplots()
    ax.plot(list(scores_pafr.keys()), list(scores_pafr.values())) 

    horiz_line_data = np.array([pafr for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Precision at a fixed recall of 0.5')
    plt.title('pca ; SelectKBest. ' + str(train) + ' testing in '+str(test))

    if whiten:
        plt.savefig(save_folder+"pcaW="+train+"test="+test+"_scores_pr.png")
    else:
        plt.savefig(save_folder+"pca="+train+"test="+test+"_scores_pr.png")

    return (robust_auc,scores)



def pca_full_experiment(whiten=False):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    scores1= dimensionality_reduction_pca(train="b278",test="b234",whiten=whiten)
    scores2= dimensionality_reduction_pca(train="b278",test="b261",whiten=whiten)
    scores3= dimensionality_reduction_pca(train="b234",test="b261",whiten=whiten)
    scores4= dimensionality_reduction_pca(train="b234",test="b360",whiten=whiten)
    scores5= dimensionality_reduction_pca(train="b261",test="b360",whiten=whiten)
    scores6= dimensionality_reduction_pca(train="b261",test="b278",whiten=whiten)
    scores7= dimensionality_reduction_pca(train="b360",test="b278",whiten=whiten)
    scores8= dimensionality_reduction_pca(train="b360",test="b234",whiten=whiten)

    fig, ax = plt.subplots()
    ax.plot(list(scores1[1].keys()), list(scores1[1].values()),label="278->234") 
    ax.plot(list(scores2[1].keys()), list(scores2[1].values()),label="278->261") 
    ax.plot(list(scores3[1].keys()), list(scores3[1].values()),label="234->261") 
    ax.plot(list(scores4[1].keys()), list(scores4[1].values()),label="234->360") 
    ax.plot(list(scores5[1].keys()), list(scores5[1].values()),label="261->360") 
    ax.plot(list(scores6[1].keys()), list(scores6[1].values()),label="261->278") 
    ax.plot(list(scores7[1].keys()), list(scores7[1].values()),label="360->278") 
    ax.plot(list(scores8[1].keys()), list(scores8[1].values()),label="360->234") 
    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('PCA SelectKBest (whitening included). ')

    if whiten:
        plt.savefig(save_folder+"pcaW_scores_global.png")
    else:
        plt.savefig(save_folder+"pca_scores_global.png")



def feature_agglomeration_with_kbins(test="b278",train="b234"):
    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()

    for i in range(2,62,4):

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("ds", cluster.FeatureAgglomeration(n_clusters=i)),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "nclusters= "+str(i)) 

    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('FeatureAgglomeration. ' + str(train) + ' testing in '+str(test))

    plt.show()


def agglomeration_with_kbins(train="b234",test="b278",linkage="ward"):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    fig, ax = plt.subplots()
    scores = {}

############ USING FEATURE SELECTION
    for i in range(2,62,4): 

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", FeatureAgglomeration(n_clusters=i,linkage=linkage)),
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "features selected= "+str(i)) 

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
        
        scores[i]=robust_auc

    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 

    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    scores[len(X.columns)]=robust_auc

    ########## PLOT ALL SCORES 

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('DS REDUCTION Feature agglomeration. Linkage= ' + linkage + " "  + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"feature_agglomeration+linkage="+linkage+"train="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    plt.title('DS REDUCTION Feature agglomeration. ' + str(train) + ' testing in '+str(test))
    plt.savefig(save_folder+"feature_agglomeration+linkage="+linkage+"train="+train+"test="+test+"_scores.png")

    return (robust_auc,scores)

def feature_agglomeration_full_experiment(linkage):
    
    save_folder = "/home/jere/Desktop/section7-fs/automated/"
    
    scores1= agglomeration_with_kbins(train="b278",test="b234",linkage=linkage)
    scores2= agglomeration_with_kbins(train="b278",test="b261",linkage=linkage)
    scores3= agglomeration_with_kbins(train="b234",test="b261",linkage=linkage)
    scores4= agglomeration_with_kbins(train="b234",test="b360",linkage=linkage)
    scores5= agglomeration_with_kbins(train="b261",test="b360",linkage=linkage)
    scores6= agglomeration_with_kbins(train="b261",test="b278",linkage=linkage)
    scores7= agglomeration_with_kbins(train="b360",test="b278",linkage=linkage)
    scores8= agglomeration_with_kbins(train="b360",test="b234",linkage=linkage)

    fig, ax = plt.subplots()
    ax.plot(list(scores1[1].keys()), list(scores1[1].values()),label="278->234") 
    ax.plot(list(scores2[1].keys()), list(scores2[1].values()),label="278->261") 
    ax.plot(list(scores3[1].keys()), list(scores3[1].values()),label="234->261") 
    ax.plot(list(scores4[1].keys()), list(scores4[1].values()),label="234->360") 
    ax.plot(list(scores5[1].keys()), list(scores5[1].values()),label="261->360") 
    ax.plot(list(scores6[1].keys()), list(scores6[1].values()),label="261->278") 
    ax.plot(list(scores7[1].keys()), list(scores7[1].values()),label="360->278") 
    ax.plot(list(scores8[1].keys()), list(scores8[1].values()),label="360->234") 
    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('Feature agglomeration after kbins. Linkage='+linkage)

    plt.savefig(save_folder+"feature_agglomeration_global_score_linkage="+linkage+".png")


def correlation_matrix(tile="b278",method='pearson'):
    X,y = retrieve_tile(tile)
    corr_df =  X.corr(method=method) 
    #hmap=sns.heatmap(corr_df,xticklabels=True, yticklabels=True)

    df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))
    hmap=sns.heatmap(df_lt,cmap="Spectral",xticklabels=True, yticklabels=True)


# Calculate AUC-PRC when one feature of each pair with correlation > t is removed.
def calculate_auc_correlation_threshold(threshold=0.5,train="b278",test="b360",method="pearson"):
    save_folder = "/home/jere/Desktop/section7-fs/automated/"
    can_drop = True
    X,y = retrieve_tile(train)
    Xt,yt=retrieve_tile(test) 

    dropped_columns = []
    score = {}
    fig, ax = plt.subplots(figsize=(15.0, 10.0))
    while (can_drop):
        # Calculate correlation 
        corr_df =  X.corr(method=method) 
        corr_df[corr_df==1]=0
        max_val = np.abs(corr_df).values.max()
        if (max_val < threshold or len(X.columns)==1):
            can_drop = False
        else:
            index = np.where(np.abs(corr_df)==max_val)[0]
            cols = X.columns
            print("Features "+cols[index[0]]+" and "+cols[index[1]]+" have correlation coefficient=" + str(max_val)+". Removing "+str(cols[index[1]]))
            X = X.drop([cols[index[1]]],axis=1)
            dropped_columns = dropped_columns + [cols[index[1]]]
            clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])
            clf.fit(X,y)
            Xt = Xt.drop([cols[index[1]]],axis=1)
            decs  = clf.decision_function(Xt)
            p,r,t = metrics.precision_recall_curve(yt,decs)
            ax.plot(r,p,label=" Threshold "+ str(max_val))
            p,r,t = metrics.precision_recall_curve(yt,decs)
            precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)
            score[max_val] = robust_auc

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Removing correlated features. method= ' + method + " "  + str(train) + ' testing in '+str(test))
    plt.savefig(save_folder+"remove_correlated+method="+method+"train="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(score.keys(),score.values())

    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    X,y = retrieve_tile(train)
    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linestyle='-',linewidth=3,label=" no feature selection ") 
    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)

    
    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(0,1,0.01), horiz_line_data, 'r--',label="Baseline")

    plt.xlabel('Correlation threshold to remove')
    plt.ylabel('ROBUST AUC PRC')
    plt.title('Removing correlated features. method= ' + method + " "  + str(train) + ' testing in '+str(test))
    plt.savefig(save_folder+"remove_correlated+method="+method+"train="+train+"test="+test+"_scores.png")
    
    return score


def drop_correlated_features_full_experiment(method="pearson"):
    
    save_folder = "/home/jere/Desktop/section7-fs/automated/"
    
    scores1= calculate_auc_correlation_threshold(train="b278",test="b234",method=method)
    scores2= calculate_auc_correlation_threshold(train="b278",test="b261",method=method)
    scores3= calculate_auc_correlation_threshold(train="b234",test="b261",method=method)
    scores4= calculate_auc_correlation_threshold(train="b234",test="b360",method=method)
    scores5= calculate_auc_correlation_threshold(train="b261",test="b360",method=method)
    scores6= calculate_auc_correlation_threshold(train="b261",test="b278",method=method)
    scores7= calculate_auc_correlation_threshold(train="b360",test="b278",method=method)
    scores8= calculate_auc_correlation_threshold(train="b360",test="b234",method=method)

    fig, ax = plt.subplots(figsize=(15.0, 10.0))
    ax.plot(list(scores1[1].keys()), list(scores1[1].values()),label="278->234") 
    ax.plot(list(scores2[1].keys()), list(scores2[1].values()),label="278->261") 
    ax.plot(list(scores3[1].keys()), list(scores3[1].values()),label="234->261") 
    ax.plot(list(scores4[1].keys()), list(scores4[1].values()),label="234->360") 
    ax.plot(list(scores5[1].keys()), list(scores5[1].values()),label="261->360") 
    ax.plot(list(scores6[1].keys()), list(scores6[1].values()),label="261->278") 
    ax.plot(list(scores7[1].keys()), list(scores7[1].values()),label="360->278") 
    ax.plot(list(scores8[1].keys()), list(scores8[1].values()),label="360->234") 
    plt.xlabel('Correlation threshold to remove')
    plt.ylabel('ROBUST AUC PRC')
    leg = ax.legend()
    plt.title('Remove correlated features. method='+method)

    plt.savefig(save_folder+"overall_drop_correlated_features_method="+method+".png")
