from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.cluster import FeatureAgglomeration
from sklearn import datasets, cluster
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

exec(open("/home/jere/Dropbox/University/Tesina/src/section4+5+6.py").read())
results_folder_dimensionality_reduction = "/home/jere/Desktop/section7-fs/"

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
            [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", SelectKBest(fc, k="all")),
             ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svml")["C"]))])
    elif (kernel=="rbf"):
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", SelectKBest(fc, k="all")),
             ("feature_map", Nystroem(gamma=get_optimal_parameters_p("svmk")["gamma"], n_components=300)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])
             
    clf.fit(X,y)

    feature_importance = list(clf["feature_selector"].scores_)
    feature_names = list(X.columns)

    with open(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_IMPORTANCE_SCORES_"+"train="+train+".pkl", 'wb') as output:
        pickle.dump(feature_importance,output, pickle.HIGHEST_PROTOCOL)    
        
    curves = {}

    nfeatures = len(feature_names)

    # GET THE AUC REMOVING 1 BY 1
    for i in range(0,nfeatures):
        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svml")["C"]))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(gamma=get_optimal_parameters_p("svmk")["gamma"], n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])
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
        
    with open(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
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

def calculate_aucs_univariate_selection(train="b234",test="b278",method="f_classif",kernel="linear"):
 
    with open(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_CURVES_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
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

    with open(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_aucs_"+"train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)  

    return scores


def calculate_all_univariate_selection_aucs():
    
    for kernel in ["linear","rbf"]:
        for method in ["f_classif","mutual_info_classif"]:
            calculate_aucs_univariate_selection(train="b278",test="b234",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b278",test="b261",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b234",test="b261",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b234",test="b360",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b261",test="b360",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b261",test="b278",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b360",test="b278",method=method,kernel=kernel)
            calculate_aucs_univariate_selection(train="b360",test="b234",method=method,kernel=kernel)

def get_aucs_univariate_selection(train="b234",test="b278",method="f_classif",kernel="linear"):
    try:
        with open(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_aucs_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
            scores = pickle.load(filename)
        return scores
    except:
        return calculate_aucs_univariate_selection(train,test,method,kernel)
            
def generate_univariate_selection_individual_subplot(train="b234",test="b278",method="f_classif",kernel="linear"):
   
    X,y = retrieve_tile(train,"full") 
    nfeatures = len(X.columns)

    scores = get_aucs_univariate_selection(train,test,method,kernel)
    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([ get_baseline_preprocessing_stage(train,test,kernel) for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 


    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    if (test=="b261" and train=="b234"):
        leg = ax.legend()
    plt.title('Test '+test+' - Train '+train)
    plt.savefig(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
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
    if  (method=="f_classif"):
        leg = ax.legend()
    #plt.title('FS: SelectKBest. Kernel='+kernel+' Method=' +method+' All tiles')
    plt.savefig(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_"+method+"_ALL_CURVES.png",bbox_inches='tight')
    
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
    
    ax.set_ylim([0,.55])

    ax.plot(list(scoresf.keys()), list(scoresf.values()),label="f_score")
    ax.plot(list(scoresm.keys()), list(scoresm.values()),label="mutual_info_classif")

    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in range(1,len(X.columns))])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,"rf") for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="Random Forest",color='g') 

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC')
    if (test=="b261" and train=="b234"):
        leg = ax.legend()
    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"_ALL_METHODS"+"_train="+train+"test="+test+".png",bbox_inches='tight')

def generate_univariate_selection_unified_all_tiles(kernel="linear"):
    generate_univariate_selection_unified_tile_subplot(train="b278",test="b234",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b278",test="b261",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b234",test="b261",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b234",test="b360",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b261",test="b360",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b261",test="b278",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b360",test="b278",kernel=kernel)
    generate_univariate_selection_unified_tile_subplot(train="b360",test="b234",kernel=kernel)

def univariate_selection_analyse_k_gain(kernel="linear",method="f_classif"):

    scores = {}
    scores_diff = {}

    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            try:    
                aucs = get_aucs_univariate_selection(train,test,method,kernel)
            except:
                print("faail")
                continue # We didn't calculate univ selection for that pair.
                
            for key in aucs.keys():
                
                auc_pair = aucs[key]
                auc_diff = aucs[key] - get_baseline_preprocessing_stage(train,test,kernel)  
                
                if (key in scores.keys()):
                    scores[key] = scores[key]+[auc_pair]
                    scores_diff[key] = scores_diff[key] + [auc_diff]
                else:
                    scores[key] = [auc_pair]
                    scores_diff[key] = [auc_diff]

    scores_avg = {}
    scores_min = {}
    scores_sdev = {}

    for key in scores.keys():
        scores_avg[key] = np.mean(scores_diff[key])
        scores_sdev[key] = np.std(scores_diff[key])
        scores_min[key] = np.min(scores_diff[key])

    avgs = np.asarray(list(scores_avg.values()))
    sdevs = np.asarray(list(scores_sdev.values()))


    fig, ax = plt.subplots()

    ax.set_ylim([-0.5,0.1])
    horiz_line_data = np.array([0 for i in scores_min.keys()])
    ax.plot(scores_min.keys(), horiz_line_data, 'r--',color='r') 
    ax.plot(list(scores_avg.keys()), list(scores_avg.values()),label="Mean difference")
    ax.fill_between(list(scores_avg.keys()), avgs-sdevs, avgs+sdevs,color="lightgrey")
    ax.plot(list(scores_min.keys()), list(scores_min.values()),label="Min difference")

    plt.xlabel('Number of selected features')
    plt.ylabel('Robust AUC-PRC improvement compared to baseline')
    leg = ax.legend()

    # [ print(k,"{:.3f}".format(scores_min[k]),"{:.3f}".format(scores_avg[k])) for k in scores_avg.keys() if scores_min[k]>0 and scores_avg[k]>0.015] Kernel
    # [ print(k,"{:.3f}".format(scores_min[k]),"{:.3f}".format(scores_avg[k])) for k in scores_avg.keys() if scores_min[k]>=-0.003]
    
    plt.savefig(results_folder_dimensionality_reduction+"UnivariateFeatureSelection/"+kernel+"BEST_K_"+method+".png",bbox_inches='tight')

########################################### FEATURE EXTRACTION - PCA ############################################

""" generate data for an individual pair of tiles """
def dimensionality_reduction_pca(train="b234",test="b278",whiten=False,kernel="linear"):
    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    curves = {}

    for i in range(0,62): 

        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_selector", PCA(n_components=62-i,whiten=whiten)),
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svml")["C"]))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 ("feature_selector", PCA(n_components=62-i,whiten=whiten)),
                 ("feature_map", Nystroem(gamma=get_optimal_parameters_p("svmk")["gamma"], n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])
        
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        curves[i]=(p,r)

    with open(results_folder_dimensionality_reduction+"PCA/"+kernel+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
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

    with open(results_folder_dimensionality_reduction+"PCA/"+kernel+"_CURVES_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
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
    ax.set_ylim([0,.55])

    ax.plot(list(scores.keys()), list(scores.values())) 


    # GET THE BASELINE
    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in range(1,len(X.columns))])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,"rf") for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="Random Forest",color='g') 

    plt.xlabel('Number of principal components used')
    plt.ylabel('Robust AUC-PRC')
    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_dimensionality_reduction+"PCA/"+kernel+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
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

    plt.xlabel('Number of principal components used')
    plt.ylabel('Robust AUC-PRC')
    if (kernel=="linear"):
        leg = ax.legend()
    #plt.title('PCA. Kernel='+kernel+' All tiles')
    plt.savefig(results_folder_dimensionality_reduction+"PCA/"+kernel+"_ALL_CURVES_pca.png",bbox_inches='tight')


###### FEATURE EXTRACTION - AGGLOMERATION ######
def feature_agglomeration(train="b234",test="b278",kernel="linear",linkage="ward"):

    X,y = retrieve_tile(test,"full") 
    Xt,yt=retrieve_tile(train)   

    curves = {}

    for i in range(0,62): 

        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("ds", cluster.FeatureAgglomeration(n_clusters=62-i,linkage=linkage)),
                 ("scaler",StandardScaler()), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svml")["C"]))])
        elif (kernel=="rbf"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("ds", cluster.FeatureAgglomeration(n_clusters=62-i,linkage=linkage)),
                 ("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(gamma=get_optimal_parameters_p("svmk")["gamma"], n_components=300)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"]))])
        
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        curves[i]=(p,r)

    with open(results_folder_dimensionality_reduction+"FeatureAgglomeration/"+kernel+"_"+linkage+"_CURVES_"+"train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)       
    
    return curves

def feature_agglomeration_full_experiment(kernel="rbf",linkage="ward"):
    #scores1= feature_agglomeration(train="b278",test="b234",kernel=kernel,linkage=linkage)
    #scores2= feature_agglomeration(train="b278",test="b261",kernel=kernel,linkage=linkage)
    #scores3= feature_agglomeration(train="b234",test="b261",kernel=kernel,linkage=linkage)
    scores4= feature_agglomeration(train="b234",test="b360",kernel=kernel,linkage=linkage)
    scores5= feature_agglomeration(train="b261",test="b360",kernel=kernel,linkage=linkage)
    #scores6= feature_agglomeration(train="b261",test="b278",kernel=kernel,linkage=linkage)
    scores7= feature_agglomeration(train="b360",test="b278",kernel=kernel,linkage=linkage)
    scores8= feature_agglomeration(train="b360",test="b234",kernel=kernel,linkage=linkage)

def feature_agglomeration_all_linkages(kernel="linear"):
    #feature_agglomeration_full_experiment(kernel=kernel,linkage="ward")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="complete")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="average")
    feature_agglomeration_full_experiment(kernel=kernel,linkage="single")


def generate_feature_agglomeration_individual_subplot(train="b278",test="b234",linkage="ward",kernel="linear"):

    with open(results_folder_dimensionality_reduction+"FeatureAgglomeration/"+kernel+"_"+linkage+"_CURVES_"+"train="+train+"test="+test+".pkl", 'rb') as filename:
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
    ax.set_ylim([0,.55])

    ax.plot(list(scores.keys()), list(scores.values())) 

    # GET THE BASELINE
    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in range(1,len(X.columns))])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,"rf") for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="Random Forest",color='g') 
    
    # Fill the metadata
    plt.xlabel('Number of features extracted')
    plt.ylabel('Robust AUC-PRC')
    if (test=="b261" and train=="b234"):
        leg = ax.legend()
    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_dimensionality_reduction+"FeatureAgglomeration/"+kernel+"_"+linkage+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
    plt.close(fig)
    return scores
    
def generate_feature_agglomeration_unified_tile_subplot(train="b278",test="b234",kernel="linear"):
    
    X,y = retrieve_tile(train,"full") 
    nfeatures = len(X.columns)
    
    linkage = "single"
    scoress =  generate_feature_agglomeration_individual_subplot(train=train,test=test,linkage=linkage,kernel=kernel)

    linkage = "average"
    try:
        scoresa =  generate_feature_agglomeration_individual_subplot(train=train,test=test,linkage=linkage,kernel=kernel)
    except:
        pass
        
    linkage = "complete"
    scoresc =  generate_feature_agglomeration_individual_subplot(train=train,test=test,linkage=linkage,kernel=kernel)

    linkage = "ward"
    scoresw =  generate_feature_agglomeration_individual_subplot(train=train,test=test,linkage=linkage,kernel=kernel)
    
    fig, ax = plt.subplots()
    ax.set_ylim([0,.55])

    ax.plot(list(scoress.keys()), list(scoress.values()),label="single")
    ax.plot(list(scoresc.keys()), list(scoresc.values()),label="complete")
    ax.plot(list(scoresw.keys()), list(scoresw.values()),label="ward")
    try:
        ax.plot(list(scoresa.keys()), list(scoresa.values()),label="average")
    except:
        pass
        
    # GET THE BASELINE
    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in range(1,len(X.columns))])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,"rf") for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="Random Forest",color='g') 
    
    plt.xlabel('Number of extracted features')
    plt.ylabel('Robust AUC-PRC')
    if (test=="b261" and train=="b234"):
        leg = ax.legend()
    plt.title('Train '+train+ '- Test '+test)
    plt.savefig(results_folder_dimensionality_reduction+"FeatureAgglomeration/"+kernel+"_ALL_LINKAGES"+"_train="+train+"test="+test+".png",bbox_inches='tight')

def generate_feature_agglomeration_unified_all_tiles(kernel="linear"):
    generate_feature_agglomeration_unified_tile_subplot(train="b278",test="b234",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b278",test="b261",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b234",test="b261",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b234",test="b360",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b261",test="b360",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b261",test="b278",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b360",test="b278",kernel=kernel)
    generate_feature_agglomeration_unified_tile_subplot(train="b360",test="b234",kernel=kernel)

def get_optimal_parameters_p(kernel="linear"):
    optimal = {}
    if (kernel=="linear" or kernel=="svml"):
        optimal["C"]=10
        optimal["n_bins"]=150
        optimal["k"]=48
        optimal["fc"]=mutual_info_classif
    elif (kernel=="rbf" or kernel=="svmk"):
        optimal["C"]=10000
        optimal["gamma"]=0.0001
        optimal["n_bins"]=100       
        optimal["k"]=45
        optimal["fc"]=mutual_info_classif
    
    return optimal
             
def generate_test_performance_data_fs(train_tile="b278",test_tiles=["b234","b261","b360"]):

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    # SVM
    clf2 = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", SelectKBest(get_optimal_parameters_p("svmk")["fc"], k=get_optimal_parameters_p("svmk")["k"])),
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svml")["C"]))])
            
    clf2.fit(X,y)
    
    #SVM-K
    clf3 = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", SelectKBest(get_optimal_parameters_p("svmk")["fc"], k=get_optimal_parameters_p("svmk")["k"])),
         ("feature_map", Nystroem(gamma=get_optimal_parameters_p("svmk")["gamma"], n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])

    clf3.fit(X,y)    
        
    for test in test_tiles:
        Xtest, ytest = retrieve_tile(test)
        curves = {}
        
        #RF
        #test_predictions = clf.predict_proba(Xtest)[:,1]
        #precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        #curves["rf"] = (precision,recall)
        
        # SVM-L
        test_predictions = clf2.decision_function(Xtest)
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        curves["svml"] = (precision,recall)

        # SVM-K
        test_predictions = clf3.decision_function(Xtest)
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        curves["svmk"] = (precision,recall)

        with open(results_folder_dimensionality_reduction+"best-train="+train_tile+ "test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)    

def generate_test_performance_data_fs_all():
    generate_test_performance_data_fs(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data_fs(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data_fs(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data_fs(train_tile="b360",test_tiles=["b234","b261","b278"])

def generate_test_performance_data_fs_subplots():
    
    scores = {}
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open(results_folder_dimensionality_reduction+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)

            fig, ax = plt.subplots()

            #p,r = curves["rf"]
            #precision_fold, recall_fold = p[::-1], r[::-1]
            #recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            #precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            #robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("rf",train,test)] = 0# robust_auc
            #ax.plot(r,p, label="Random Forest")

            p,r = curves["svml"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svml",train,test)] = robust_auc
            ax.plot(r,p, label="Linear SVM")
            
            p,r = curves["svmk"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svmk",train,test)] = robust_auc
            ax.plot(r,p, label="RBF SVM")
            
            plt.title('Train ' + train + "- Test" + test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            leg = ax.legend();
    
            plt.savefig(results_folder_dimensionality_reduction+"best-train="+train+ "test="+test+".png",bbox_inches='tight')

    with open(results_folder_dimensionality_reduction+"baseline_aucs.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)   
        
# generate_table_comparison(results_folder_dimensionality_reduction+ "baseline_aucs.pkl", results_folder_preproces+"baseline_aucs.pkl")

