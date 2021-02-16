from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.cluster import FeatureAgglomeration
from sklearn import datasets, cluster
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

exec(open("/home/jere/Dropbox/University/Tesina/src/section4+5+6.py").read())
save_folder = "/home/jere/Desktop/section7-fs/"

""" FEATURE SELECTION: UNIVARIATE SELECTION """

""" Generate data for a given pair of tiles"""
def univariate_selection_experiment(train="b234",test="b278",method="f_classif",kernel="linear"):
    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)   
    if (method=="f_classif"):
        fc = f_classif
    elif (method=="mutual_info_classif"):
        fc = mutual_info_classif
    elif (method=="chi2"):
        fc = chi2

    # GET THE RANKING
    if (kernel=="linear"):
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", SelectKBest(fc, k="all")),
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10))])
    elif (kernel=="rbf"):
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", SelectKBest(fc, k="all")),
             ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=100000))])
             
    clf.fit(X,y)

    feature_importance = list(clf["feature_selector"].scores_)
    feature_names = list(X.columns)

    with open(save_folder+"UnivariateFeatureSelection/"+kernel+"_"+method+"_IMPORTANCE_SCORES_"+"train="+train+".pkl", 'wb') as output:
        pickle.dump(feature_importance,output, pickle.HIGHEST_PROTOCOL)    
        
    curves = {}

    nfeatures = len(feature_names)

    # GET THE AUC REMOVING 1 BY 1
    for i in range(0,nfeatures):
        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=100000))])
        clf.fit(X,y)    
        test_predictions = clf.decision_function(Xt)
        p, r, t = metrics.precision_recall_curve(yt, test_predictions)
        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
        curves[i] = (p,r)      
        print("Evaluating performance with "+str(nfeatures-i)+" features gave AUC="+str(robust_auc))
        # Pick the next feature to be removed
        min_importance = min(feature_importance)
        min_index = feature_importance.index(min_importance)
        min_name = feature_names[min_index]
        print("Removing feature " + min_name + " whose importance is "+ str(min_importance))
        # Remove it
        X = X.drop(columns=[min_name])
        Xt = Xt.drop(columns=[min_name])

        del feature_importance[min_index]
        del feature_names[min_index]
        
    with open(save_folder+"UnivariateFeatureSelection/"+kernel+"_"+method+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)       
                
""" Generate data for all tiles"""
def univariate_selection_experiment_all_tiles(method="f_classif",kernel="linear"):
    univariate_selection_experiment(train="b278",test="b234",method=method,kernel=kernel)
    univariate_selection_experiment(train="b278",test="b261",method=method,kernel=kernel)
    univariate_selection_experiment(train="b234",test="b261",method=method,kernel=kernel)
    univariate_selection_experiment(train="b234",test="b360",method=method,kernel=kernel)
    univariate_selection_experiment(train="b261",test="b360",method=method,kernel=kernel)
    univariate_selection_experiment(train="b261",test="b278",method=method,kernel=kernel)
    univariate_selection_experiment(train="b360",test="b278",method=method,kernel=kernel)
    univariate_selection_experiment(train="b360",test="b234",method=method,kernel=kernel)

""" Generate data for all methods"""
def univariate_selection_experiment_all_methods(kernel="linear"):
    univariate_selection_experiment_all_tiles("f_classif",kernel)
    univariate_selection_experiment_all_tiles("mutual_info_classif",kernel)
    #univariate_selection_experiment_all_tiles("chi2",kernel) Cannot run chi2 on negative data

""" Generate data for all kernels"""
def generate_univariate_selection_data():
    univariate_selection_experiment_all_methods("linear")
    univariate_selection_experiment_all_methods("rbf")

def generate_univariate_selection_individual_subplot(train="b234",test="b278",method="f_classif",kernel="linear"):
    
    with open(save_folder+"UnivariateFeatureSelection/"+kernel+"_"+method+"_CURVES_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
        curves = pickle.load(filename)

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)  
    nfeatures = len(X.columns)

    scores = {}

    # GET THE AUC REMOVING 1 BY 1
    for i in range(0,nfeatures):
        p,r = curves[i]
        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)     
        scores[nfeatures-i] = robust_auc

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([scores[nfeatures] for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 


    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('FS: SelectKBest. Kernel='+kernel+' Method=' +method+' test='+test+' train='+train)
    plt.savefig(save_folder+"UnivariateFeatureSelection/"+kernel+"_"+method+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
    plt.close(fig)
    return scores

def generate_univariate_selection_unified_method_subplot(method="f_classif",kernel="linear"):
    
    scores1= generate_univariate_selection_individual_subplot(train="b278",test="b234",method=method,kernel=kernel)
    scores2= generate_univariate_selection_individual_subplot(train="b278",test="b261",method=method,kernel=kernel)
    scores3= generate_univariate_selection_individual_subplot(train="b234",test="b261",method=method,kernel=kernel)
    scores4= generate_univariate_selection_individual_subplot(train="b234",test="b360",method=method,kernel=kernel)
    scores5= generate_univariate_selection_individual_subplot(train="b261",test="b360",method=method,kernel=kernel)
    scores6= generate_univariate_selection_individual_subplot(train="b261",test="b278",method=method,kernel=kernel)
    scores7= generate_univariate_selection_individual_subplot(train="b360",test="b278",method=method,kernel=kernel)
    scores8= generate_univariate_selection_individual_subplot(train="b360",test="b234",method=method,kernel=kernel)

    fig, ax = plt.subplots(figsize=(10.0, 6.66))
    ax.plot(list(scores1.keys()), list(scores1.values()),label="b278->b234") 
    ax.plot(list(scores2.keys()), list(scores2.values()),label="b278->b261") 
    ax.plot(list(scores3.keys()), list(scores3.values()),label="b234->b261") 
    ax.plot(list(scores4.keys()), list(scores4.values()),label="b234->b360") 
    ax.plot(list(scores5.keys()), list(scores5.values()),label="b261->b360") 
    ax.plot(list(scores6.keys()), list(scores6.values()),label="b261->b278") 
    ax.plot(list(scores7.keys()), list(scores7.values()),label="b360->b278") 
    ax.plot(list(scores8.keys()), list(scores8.values()),label="b360->b234") 


    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('FS: SelectKBest. Kernel='+kernel+' Method=' +method+' All tiles')
    plt.savefig(save_folder+"UnivariateFeatureSelection/"+kernel+"_"+method+"_ALL_CURVES.png",bbox_inches='tight')
    
def generate_univariate_selection_unified_tile_subplot(train="b278",test="b234",kernel="linear"):
    
    X,y = retrieve_tile(train,"full") 
    nfeatures = len(X.columns)
    
    # get f_score 
    method = "f_classif"
    scoresf =  generate_univariate_selection_individual_subplot(train=train,test=test,method=method,kernel=kernel)

    # get univariate_fs
    method = "mutual_info_classif"
    scoresm =  generate_univariate_selection_individual_subplot(train=train,test=test,method=method,kernel=kernel)

    fig, ax = plt.subplots()
    ax.plot(list(scoresf.keys()), list(scoresf.values()),label="f_score")
    ax.plot(list(scoresm.keys()), list(scoresm.values()),label="mutual_info_classif")

    horiz_line_data = np.array([scoresf[nfeatures] for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('FS: SelectKBest. Train='+train+" Test="+test+' Kernel='+kernel)
    plt.savefig(save_folder+"UnivariateFeatureSelection/"+kernel+"_ALL_METHODS"+"_train="+train+"test="+test+".png",bbox_inches='tight')

def generate_univariate_selection_unified_all_tiles(kernel="linear"):
    generate_univariate_selection_unified_tile_subplot(train="b278",test="b234",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b278",test="b261",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b234",test="b261",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b234",test="b360",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b261",test="b360",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b261",test="b278",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b360",test="b278",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b360",test="b234",kernel=kernel)

########################################### FEATURE EXTRACTION - PCA ############################################

def dimensionality_reduction_pca(train="b234",test="b278",whiten=False,kernel="linear"):
    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    curves = {}

    for i in range(0,62): 

        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_selector", PCA(n_components=62-i,whiten=whiten)),
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_selector", PCA(n_components=62-i,whiten=whiten)),
                 ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=100000))])
        
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        curves[i]=(p,r)

    with open(save_folder+"PCA/"+kernel+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)       
    
    return curves

def pca_generate_all_data(whiten=False,kernel="linear"):
    scores1= dimensionality_reduction_pca(train="b278",test="b234",whiten=whiten,kernel=kernel)
    scores2= dimensionality_reduction_pca(train="b278",test="b261",whiten=whiten,kernel=kernel)
    scores3= dimensionality_reduction_pca(train="b234",test="b261",whiten=whiten,kernel=kernel)
    scores4= dimensionality_reduction_pca(train="b234",test="b360",whiten=whiten,kernel=kernel)
    scores5= dimensionality_reduction_pca(train="b261",test="b360",whiten=whiten,kernel=kernel)
    scores6= dimensionality_reduction_pca(train="b261",test="b278",whiten=whiten,kernel=kernel)
    scores7= dimensionality_reduction_pca(train="b360",test="b278",whiten=whiten,kernel=kernel)
    scores8= dimensionality_reduction_pca(train="b360",test="b234",whiten=whiten,kernel=kernel)
    
def generate_pca_plots(train="b278",test="b234",kernel="linear"):

    with open(save_folder+"PCA/"+kernel+"_CURVES_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
        curves = pickle.load(filename)    

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)  
    nfeatures = len(X.columns)

    scores = {}

    # GET THE AUC REMOVING 1 BY 1
    for i in range(0,nfeatures):
        p,r = curves[i]
        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)     
        scores[nfeatures-i] = robust_auc

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([scores[nfeatures] for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    plt.title('PCA. Kernel='+kernel+' test='+test+' train='+train)
    plt.savefig(save_folder+"PCA/"+kernel+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
    plt.close(fig)
    
    return scores
    
def generate_pca_all_subplots(kernel="linear"):
    
    scores1= generate_pca_plots(train="b278",test="b234",kernel=kernel)
    scores2= generate_pca_plots(train="b278",test="b261",kernel=kernel)
    scores3= generate_pca_plots(train="b234",test="b261",kernel=kernel)
    scores4= generate_pca_plots(train="b234",test="b360",kernel=kernel)
    scores5= generate_pca_plots(train="b261",test="b360",kernel=kernel)
    scores6= generate_pca_plots(train="b261",test="b278",kernel=kernel)
    scores7= generate_pca_plots(train="b360",test="b278",kernel=kernel)
    scores8= generate_pca_plots(train="b360",test="b234",kernel=kernel)

    fig, ax = plt.subplots(figsize=(10.0, 6.66))
    ax.plot(list(scores1.keys()), list(scores1.values()),label="b278->b234") 
    ax.plot(list(scores2.keys()), list(scores2.values()),label="b278->b261") 
    ax.plot(list(scores3.keys()), list(scores3.values()),label="b234->b261") 
    ax.plot(list(scores4.keys()), list(scores4.values()),label="b234->b360") 
    ax.plot(list(scores5.keys()), list(scores5.values()),label="b261->b360") 
    ax.plot(list(scores6.keys()), list(scores6.values()),label="b261->b278") 
    ax.plot(list(scores7.keys()), list(scores7.values()),label="b360->b278") 
    ax.plot(list(scores8.keys()), list(scores8.values()),label="b360->b234") 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    leg = ax.legend()
    plt.title('PCA. Kernel='+kernel+' All tiles')
    plt.savefig(save_folder+"PCA/"+kernel+"_ALL_CURVES.png",bbox_inches='tight')
    




def feature_agglomeration(train="b234",test="b278",kernel="linear",linkage="ward"):

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    curves = {}

    for i in range(0,62): 

        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
                 ("ds", cluster.FeatureAgglomeration(n_clusters=62-i,linkage=linkage)),
                 ("scaler",StandardScaler()), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                 ("ds", cluster.FeatureAgglomeration(n_clusters=62-i,linkage=linkage)),
                 ("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=100000))])
        
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        curves[i]=(p,r)

    with open(save_folder+"FeatureAgglomeration/"+kernel+"_"+linkage+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)       
    
    return curves

def feature_agglomeration_full_experiment(kernel="linear",linkage="ward"):
    scores1= feature_agglomeration(train="b278",test="b234",kernel=kernel,linkage=linkage)
    scores2= feature_agglomeration(train="b278",test="b261",kernel=kernel,linkage=linkage)
    scores3= feature_agglomeration(train="b234",test="b261",kernel=kernel,linkage=linkage)
    scores4= feature_agglomeration(train="b234",test="b360",kernel=kernel,linkage=linkage)
    scores5= feature_agglomeration(train="b261",test="b360",kernel=kernel,linkage=linkage)
    scores6= feature_agglomeration(train="b261",test="b278",kernel=kernel,linkage=linkage)
    scores7= feature_agglomeration(train="b360",test="b278",kernel=kernel,linkage=linkage)
    scores8= feature_agglomeration(train="b360",test="b234",kernel=kernel,linkage=linkage)

def feature_agglomeration_all_linkages(kernel="linear"):
    #feature_agglomeration_full_experiment(kernel=kernel,linkage="ward")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="complete")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="average")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="single")

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


    ########### BASELINE
    clf = Pipeline( 
    [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
     ("scaler",StandardScaler()), 
     ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])
    X,y = retrieve_tile(train)
    Xt,yt=retrieve_tile(test) 

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    fig, ax = plt.subplots()
    ax.plot(score.keys(),score.values(), marker='o')
    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    dom = np.linspace(threshold,1)
    horiz_line_data = np.array([robust_auc for i in dom])
    ax.plot(dom, horiz_line_data, 'r--',label="Baseline")

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
