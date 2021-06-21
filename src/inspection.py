"""
DATA VISUALIZATION AND INSPECTION EXPERIMENTS 
Chapters 6 of master's thesis.
"""
from numpy import inf
from dimensionality_reduction import *
from sklearn.inspection import permutation_importance

CARPYNCHO_LOCAL_FOLDER       = ""  # "/home/jere/carpyncho/"
EXPERIMENTS_OUTPUT_FOLDER_INSPECTION = ""

def init(carpyncho_local_folder_path, output_folder_inspecion):
    """
    Initialize this module
    
    Parameters
    ----------
    carpyncho_local_folder_path: Path in the local filesystem where VVV tiles downloaded from
      Carpyncho are stored (see common.py)

    output_folder_inspecion: Path to the folder where final and intermediate results of inspection 
      experiments  will be saved
    """

    global CARPYNCHO_LOCAL_FOLDER
    global EXPERIMENTS_OUTPUT_FOLDER_INSPECTION
    global CARPYNCHO
    
    CARPYNCHO_LOCAL_FOLDER = carpyncho_local_folder_path
    EXPERIMENTS_OUTPUT_FOLDER_INSPECTION = output_folder_inspecion
    CARPYNCHO = CarpynchoWrapper(CARPYNCHO_LOCAL_FOLDER)

    
################### VARIABLE IMPORTANCE BY STATISTICAL TESTS #####################
def calculate_univariate_importance(train="b278",method="chi2"):
    """
    Calculate the importance of each variable to predict RR-Lyrae stars by means of statistical tests.
    Persist the results in the local filesystem as pkl files.

    Parameters
    ----------
    train: id of the tile to be used as training dataset
    method: Statistical test to be used: "f_classif", "chi2" or "mutual_info_classif". 
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    X,y = CARPYNCHO.retrieve_tile(train,"full") 

    if (method=="f_classif" or method=="all"):
        selector = SelectKBest(f_classif, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", selector )])
        clf.fit(X, y)
        
        with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   
            
    
    if (method=="mutual_info_classif" or method=="all"):
        selector = SelectKBest(mutual_info_classif, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", selector )])
        clf.fit(X, y)
        
        with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/mutual_info_classif_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   
    
    if (method=="chi2" or method=="all"):
        selector = SelectKBest(chi2, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("feature_selector", selector )])
        clf.fit(X, y)   

        with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   

def calculate_all_importances():
    """
    Calculate the importance of each variable to predict RR-Lyrae stars by means of statistical tests
    in all tiles.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    for tile in ["b234","b278","b261","b360"]:
        calculate_univariate_importance(train="b278",method="all")
    
def plot_univariate_importance_pvalue(train="b278"):
    """
    Plot the p-values that measure the irrelevance of each feature when it comes to identifying
    RR-Lyrae stars.
    
    Parameters
    ----------
    train: id of the tile to be used as training dataset
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_chi = pickle.load(output)       
    
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_f = pickle.load(output)       

    X,y = CARPYNCHO.retrieve_tile(train,"full")
    X_indices = np.arange(X.shape[-1])

    ##### PLOT P VALUES
    fig, ax = plt.subplots(figsize=(19,8))

    plt.bar(X_indices , selector_f.pvalues_, width=.2,  label="ANOVA")
    plt.bar(X_indices +0.2, selector_chi.pvalues_ , width=.2,  label="Chi^2")

    plt.xlim([-1, 62])
    plt.title("P values in tile " + train)
    plt.xlabel('Feature number')
    plt.ylabel('p-value')
    plt.xticks(X_indices,X_indices)
    plt.axis('tight')
    plt.legend(loc='upper right')

    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"_variable_importance_pvalues.png",bbox_inches='tight')
    
def plot_univariate_importance_scores(train="b278"):
    """
    Plot the score that measure the importance of each feature when it comes to identifying
    RR-Lyrae stars.
    
    Parameters
    ----------
    train: id of the tile to be used as training dataset
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_chi = pickle.load(output)       

    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_f = pickle.load(output)       

    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/mutual_info_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_m = pickle.load(output)   
        
    X,y = CARPYNCHO.retrieve_tile(train,"full")
    X_indices = np.arange(X.shape[-1])

    fig, ax = plt.subplots(figsize=(19,8))

    scores_f = selector_f.scores_ / sum(selector_f.scores_)

    plt.bar(X_indices , scores_f, width=.2,  label="ANOVA")
    
    scores_chi = selector_chi.scores_ / sum(selector_chi.scores_)

    plt.bar(X_indices +0.22, scores_chi , width=.2,  label="Chi^2")

    scores_m = selector_m.scores_ / sum(selector_m.scores_)

    plt.bar(X_indices -0.22, scores_m, width=.2,  label="Mutual information")

    plt.xlim([-1, 62])
    plt.title("Scores in tile " + train)
    plt.xlabel('Feature number')
    plt.ylabel('Normalized scores')
    plt.xticks(X_indices,X_indices)
    plt.axis('tight')
    plt.legend(loc='upper right')

    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"_variable_importance_scores.png",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(19,8))

    scores_f = selector_f.scores_
    order_f = scores_f.argsort()
    ranks_f = order_f.argsort()
    plt.bar(X_indices , ranks_f, width=.2,  label="ANOVA")
    
    scores_chi = selector_chi.scores_
    order_chi = scores_chi.argsort()
    ranks_chi = order_chi.argsort()
    plt.bar(X_indices +0.22, ranks_chi , width=.2,  label="Chi^2")

    scores_m = selector_m.scores_
    order_m = scores_m.argsort()
    ranks_m = order_m.argsort()
    plt.bar(X_indices -0.22, ranks_m, width=.2,  label="Mutual information")

    plt.xlim([-1, 62])
    plt.title("Ranking in tile " + train)
    plt.xlabel('Feature number')
    plt.ylabel('Importance in ranking')
    plt.xticks(X_indices,X_indices)
    plt.axis('tight')
    plt.legend(loc='upper right')

    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"_variable_importance_ranking.png",bbox_inches='tight')
    
    
def calculate_all_plots_univariate_importance():
    """
    Generate all plots describing variable importance by means of univariate statistical tests.
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    for tile in ["b234","b278","b261","b360"]:
        plot_univariate_importance_scores(tile)
        plot_univariate_importance_pvalue(tile)


################### VARIABLE IMPORTANCE BY ML METHODS #####################
                 
def calculate_ml_importance(train="b278"):
    """
    Calculate the importance of each feature based on trained RF and linear SVM models.
    
    Parameters
    ----------
    train: id of the tile to be used as training dataset
    
    """
    X,y=CARPYNCHO.retrieve_tile(train)

    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    svc = LinearSVC(verbose=3, max_iter=10000, C=get_optimal_parameters_p("svml")["C"], dual=False)
    clf2 = Pipeline([
    ("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
    ('scaler', StandardScaler()),
    ('clf', svc) ])
    clf2.fit(X,y)
    
    persist = (svc.coef_,clf.feature_importances_)
    
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/ml_importance_"+"train="+train+".pkl", 'wb') as output:
        pickle.dump(persist,output, pickle.HIGHEST_PROTOCOL)   

def plot_ml_importance(train="b278"):
    """
    Plot the importance of each feature based on trained RF and linear SVM models.
    
    Parameters
    ----------
    train: id of the tile to be used as training dataset
    
    """
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/ml_importance_"+"train="+train+".pkl", 'rb') as output:
        persist = pickle.load(output)   
    fig, ax = plt.subplots(figsize=(19,8))

    X,y=CARPYNCHO.retrieve_tile(train)
    X_indices = np.arange(X.shape[-1])

    rf_importance = persist[1]
    rf_importance_norm = rf_importance / sum(rf_importance)

    svc_coefs = persist[0] 
    svm_importance = np.abs(svc_coefs.tolist()[0])
    svm_importance_norm = svm_importance / sum(svm_importance)
    
    plt.bar(X_indices , rf_importance_norm, width=.2,label="RF Gini")
    plt.bar(X_indices+0.2 , svm_importance_norm, width=.2,label="abs SVM weight")
    plt.xlabel('Feature number')
    plt.ylabel('Score (normalized)')
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title("Tile " + train)
    plt.xticks(X_indices,X_indices)

    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"_ML_variable_importance_scores.png",bbox_inches='tight')

def calculate_all_plots_ml_importance():
    for tile in ["b234","b278","b261","b360"]:
        plot_ml_importance(tile)


######## VARIABLE IMPORTANCE BY PERMUTATION IMPORTANCE #######

def calculate_permutation_importance(train="b234", test="b261", method="svmk"):
    """
    Calculate the permutation importance of each variable to predict RR-Lyrae stars.
    Persist the results in the local filesystem as pkl files.

    Parameters
    ----------
    train: id of the tile to be used as training dataset
    test: id of the tile to be used as test dataset
    method: either "svmk", "rf" or "linear"
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/permutation_importance.html

    """
    X,y = CARPYNCHO.retrieve_tile(train,"full") 
    Xt,yt = CARPYNCHO.retrieve_tile(test,"full")
    
    if method=="rf":
        clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
        clf.fit(X,y)

    if method=="linear":
        clf = Pipeline([
            ('disc',KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(verbose=3, max_iter=100000, C=get_optimal_parameters_p("svml")["C"], dual=False)) ])
        clf.fit(X,y)

    #SVM-K
    if method=="svmk":
        clf = Pipeline( 
            [('disc',KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_p("svmk")["gamma"],)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])

        clf.fit(X,y)    

    result = permutation_importance(clf, Xt, yt, scoring="average_precision",n_repeats=1)
        
    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Permutation_importance/"+method+"_RESULT_train="+train+"test="+test+".pkl", 'wb') as output:
        pickle.dump(result,output, pickle.HIGHEST_PROTOCOL)   


def calculate_all_permutation_data():
    """
    Calculate the permutation importance of each variable to predict RR-Lyrae stars
    in all tiles.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """
    for method in ["rf","linear","svmk"]:
        #calculate_permutation_importance("b234","b261",method)
        calculate_permutation_importance("b261","b278",method)
        calculate_permutation_importance("b278","b360",method)
        calculate_permutation_importance("b360","b234",method)

def plot_permutation_importance(train="b234",test="b261"):
    """
    Plot the permutation importance of each variable to predict RR-Lyrae stars.
    Persist the results in the local filesystem as pkl files.

    Parameters
    ----------
    train: id of the tile to be used as training dataset
    test: id of the tile to be used as test datast
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/permutation_importance.html

    """
    result = {}
    
    for method in ["rf","linear","svmk"]:
        with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Permutation_importance/"+method+"_RESULT_train="+train+"test="+test+".pkl", 'rb') as output:
            result[method] = pickle.load(output)   
    
    X,y=CARPYNCHO.retrieve_tile(train)

    fig, ax = plt.subplots(figsize=(19,3))

    X_indices = np.arange(X.shape[-1])
    
    plt.bar(X_indices , result["rf"]["importances_mean"], width=.2,label="RF")
    plt.xlabel('Feature number')
    plt.ylabel(' Importance')
    plt.axis('tight')
    plt.title("Permutation imporance - Random forest - Train " + train + " Test " + test)
    plt.xticks(X_indices,X_indices)
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"RF_permutation_variable_importance_scores.png",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(19,3))
    plt.bar(X_indices , result["linear"]["importances_mean"], width=.2,label="SVM L")
    plt.xlabel('Feature number')
    plt.ylabel(' Importance')
    plt.axis('tight')
    plt.title("Permutation imporance - Linear SVM - Train " + train + " Test " + test)
    plt.xticks(X_indices,X_indices)
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"L_permutation_variable_importance_scores.png",bbox_inches='tight')

    
    fig, ax = plt.subplots(figsize=(19,3))    
    plt.bar(X_indices , result["svmk"]["importances_mean"], width=.2,label="SVM RBF")
    plt.xlabel('Feature number')
    plt.ylabel('Importance')
    plt.axis('tight')
    plt.title("Permutation imporance - SVM RBF - Train " + train + " Test " + test)
    plt.xticks(X_indices,X_indices)
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VariableImportance/test="+train+"RBF_permutation_variable_importance_scores.png",bbox_inches='tight')
        
def plot_all_permutation():
    """
    Plot the permutation importance of each variable to predict RR-Lyrae stars
    across all tiles.
    
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/permutation_importance.html

    """
    for method in ["rf","linear","svmk"]:
        plot_permutation_importance("b234","b261")
        plot_permutation_importance("b261","b278")
        plot_permutation_importance("b278","b360")
        plot_permutation_importance("b360","b234")

############## CORRELATIONS ###############

def calculate_correlation_matrix(tile="b278",method='pearson'):
    """
    Calculate the matrix of correlations between features.

    Parameters
    ----------
    tile: id of the tile to be used for the analysis
    method: method used to calculate the correlation between two arbitrary
    features. Either "spearman", "pearson" or "kendall"
    
    References
    ----------
    [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

    """
    fig, ax = plt.subplots(figsize=(15,10))
    X,y = CARPYNCHO.retrieve_tile(tile)
    corr_df =  X.corr(method=method) 
    df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))
    X_indices = np.arange(X.shape[-1])
    
    hmap=sns.heatmap(df_lt,cmap="Spectral",xticklabels=X_indices, yticklabels=X_indices,vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.title('Coeficientes de correlacion '+ method  +' en el tile '+tile)
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/"+method+"_"+tile+"_MATRIX.png",bbox_inches='tight')

    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/train="+tile+"_"+method+"_correlation_matrix.pkl", 'wb') as output:
        pickle.dump(corr_df,output, pickle.HIGHEST_PROTOCOL)
        
def calculate_all_correlation_matrixes():
    """
    Calculate the matrix of correlations between features for all tiles and methods.    
    """
            
def calculate_correlations(threshold=0.5,train="b278",test=["b234","b278","b261","b360"],method="pearson",kernel="linear"):
    """
    Run an experiment that:

    1) Identifies the most strongly correlated pair of features in @p train
    2) Removes oen of the fatures from @p train
    3) Trains SVM with the resulting dataset
    4) Calculates the area under the curve of precision-recall for each test dataset
    5) Stores the results and repeats 1.

    Parameters
    ----------
    threshold: If the pair identified in (1) has correlation coefficient smaller than threshold,
               terminate the algorithm.
    train: tile to be used as training dataset
    test: tiles to be used as test datasets
    method: method used to calculate the correlations (either "kendall", "spearman" or "pearson")
    kernel: which kernel to use in SVM. (Either "linear" or "rbf")

    """
    X,y = CARPYNCHO.retrieve_tile(train)
    can_drop = True
    dropped_columns = []
    n_dropped_features = 0
    persist = {}

    while (can_drop):

        # Calculate correlation matrix 
        corr_df =  X.corr(method=method) 
        corr_df[corr_df==1]=0

        # Get the most correlated pair of features
        max_val = np.abs(corr_df).values.max()

        # Check if stop condition's been reached 
        if (max_val < threshold or len(X.columns)==1):
            can_drop = False
        else:
            
            # Identify the two most correlated features
            index = np.where(np.abs(corr_df)==max_val)[0]
            cols = X.columns
            feature_1 = cols[index[0]]
            feature_2 = cols[index[1]]
            n_dropped_features = n_dropped_features + 1
            print("Iter "+str(n_dropped_features)+" Features "+feature_1+" and "+feature_2+" have correlation coefficient=" + str(max_val)+". Removing "+feature_2)

            # Remove one of the pair
            dropped_columns = dropped_columns + [feature_2]
            X = X.drop([feature_2],axis=1)

            # Train a SVM
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

            # Test it on the other tiles, and save the results
            curves = {}
            for test_tile in test:
                if (test_tile=="train"):
                    continue
                Xt,yt = CARPYNCHO.retrieve_tile(test_tile)
                Xt = Xt.drop(dropped_columns,axis=1)

                decs  = clf.decision_function(Xt)
                p,r,t = metrics.precision_recall_curve(yt,decs)
                precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
                recall_interpolated    = np.linspace(MIN_RECALL_GLOBAL, 1, N_SAMPLES_PRC)
                precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
                robust_auc = auc(recall_interpolated, precision_interpolated)

                curves[test_tile] = (p,r,robust_auc)

            # Save results in persistent object:  (Correlated_feature,Removed_feature,correlation_value,
            persist[n_dropped_features] = (feature_1,feature_2,max_val,curves)

    with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/train="+train+"_"+method+"_"+kernel+"_persisted_data.pkl", 'wb') as output:
        pickle.dump(persist,output, pickle.HIGHEST_PROTOCOL)   

def generate_correlation_data(kernel="linear",method="pearson"):
    calculate_correlations(threshold=0,train="b234",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b278",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b261",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b360",test=["b234","b278","b261","b360"],method=method,kernel=kernel)    

def analyse_correlations(method="pearson",kernel="linear"):
    """
    Summarize and plot the information generated in the experiment 'calculate_correlations'

    Parameters
    ----------
    method: method used to calculate the correlations (either "kendall", "spearman" or "pearson")
    kernel: which kernel to use in SVM. (Either "linear" or "rbf")

    """
    diffs = {}

    for train in ["b234","b278","b261","b360"]:

        with open(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/train="+train+"_"+method+"_"+kernel+"_persisted_data.pkl", 'rb') as output:
            db = pickle.load(output)   

        correlations = {}
        aucs = {}
        auc_1 = {}
        auc_2 = {}
        auc_3 = {}
        auc_4 = {}

        for k in db.keys():
            correlations[k]=db[k][2]        
            auc_1[k] = db[k][3]["b234"][2]
            auc_2[k] = db[k][3]["b278"][2]
            auc_3[k] = db[k][3]["b261"][2]
            auc_4[k] = db[k][3]["b360"][2]

        scores = {}
        scores["b234"] = auc_1
        scores["b278"] = auc_2
        scores["b261"] = auc_3
        scores["b360"] = auc_4

        for test in ["b234","b278","b261","b360"]:
            if test==train:
                continue
            aucs = scores[test]
            
            fig, ax = plt.subplots()
            domain = list(aucs.keys())[::-1]
            codomain = list(aucs.values())
            horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in domain])
            if (kernel=="linear"):
                label = "SVM Lineal"
            elif (kernel=="rbf"):
                label = "SVM RBF"
            ax.plot(domain, codomain,label=label+" + Eliminar correlaciones") 
            ax.plot(domain, horiz_line_data, 'r--',label=label+"  baseline") 
            plt.xlabel('Número de atributos utilizados')
            plt.ylabel('R-AUPRC')
            plt.title('Train '+train+' - Test '+test)
            ax.set_ylim([0,.55])
            
            if train=="b234" and test=="b261":
                leg = ax.legend(loc='upper left')
                
            plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/"+method+"_"+kernel+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
            plt.close(fig)


            fig, ax = plt.subplots()
            domain = list(correlations.values())
            horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in domain])
            if (kernel=="linear"):
                label = "SVM Lineal"
            elif (kernel=="rbf"):
                label = "SVM RBF"
            ax.plot(domain, list(aucs.values()),label=label+" + Eliminar correlaciones") 



            ax.set_ylim([0,.55])
            ax.plot(domain, horiz_line_data, 'r--',label=label) 
            
            if train=="b234" and test=="b261":
                leg = ax.legend(loc='upper left')
            plt.xlabel('Correlation threshold')
            plt.ylabel('R-AUPRC')
            plt.title('Train '+train+' - Test '+test)
            plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/"+method+"_"+kernel+"_INDIVIDUAL_CURVES_CORR_"+"train="+train+"test="+test+".png",bbox_inches='tight')
            plt.close(fig)
            
            
            for k in db.keys():
                diffs[(train,test,k)]= aucs[k] - get_baseline_preprocessing_stage(train,test,kernel)
    
    # Finished main loop,let's print the overal evaluation
    grouped_diffs = {}
    for k in db.keys():
        grouped_diffs[k]=[]
        for train in ["b234","b278","b261","b360"]:
            for test in ["b234","b278","b261","b360"]:
                if test==train:
                    continue
                grouped_diffs[k] = grouped_diffs[k] + [diffs[(train,test,k)]]
        
    avg_diffs = {} 
    min_diffs = {}
    sdev_diffs =  {}
    
    for k in db.keys():
        avg_diffs[k] = np.mean(grouped_diffs[k])
        min_diffs[k] = np.min(grouped_diffs[k])
        sdev_diffs[k] = np.std(grouped_diffs[k])

    avgs = np.asarray(list(avg_diffs.values()))
    sdevs = np.asarray(list(sdev_diffs.values()))

    fig, ax = plt.subplots()

    ax.set_ylim([-0.5,0.1])
    horiz_line_data = np.array([0 for i in min_diffs.keys()])
    ax.plot(list(min_diffs.keys())[::-1], horiz_line_data, 'r--',color='r') 
    ax.plot(list(min_diffs.keys())[::-1], list(avg_diffs.values()),label="Mean difference")
    ax.fill_between(list(avg_diffs.keys())[::-1], avgs-sdevs, avgs+sdevs,color="lightgrey")
    ax.plot(list(min_diffs.keys())[::-1], list(min_diffs.values()),label="Min difference")

    plt.xlabel('Número de atributos utilizados')
    plt.ylabel('Ganancia en R-AUPRC respecto al baseline')
    if (kernel=="linear"):
        leg = ax.legend(loc='lower right')
    
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"Correlations/"+method+"_"+kernel+"_CORRELATIONS_BIG_PICTURE.png",bbox_inches='tight')
         
    
########### DATA VISUALIZATION ############

def plot_fetures_distribution(tile="b278"):
    """
    Plot frequency histograms of each feature in @p tile

    Parameters
    ----------
    tile: id of the tile to be used as dataset

    """
    X,y = CARPYNCHO.retrieve_tile(tile,"full") 
    columns = X.columns

    h = 11
    k = 6
    # Without binning
    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(20,20))
    for i in range(h):
        for j in range(k):
            index = j*h+i
            if (index < len(X.columns)):
                col = X.columns[index]
                a = (X[col]).hist(bins=100,ax=ax[i,j])  
                a.set_title(col)
                a.tick_params(axis='both', which='major', labelsize= 0)
            else:
                a = ax[i,j]
                a.set_title("")
                a.tick_params(axis='both', which='major', labelsize= 0)    
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VisualizandoVariables/allfeatures_"+tile+".png",bbox_inches='tight')
    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(15,17))


def compare_same_feature_different_tiles(f_index=0,tiles=["b234","b360","b278","b261"]):
    """
    Plot frequency histograms of feature number @p f_index for all tiles in @p tiles
    """

    fig, ax = plt.subplots(2, 2, sharey=True,sharex=True,figsize=(15.0, 10.0))

    X,y = CARPYNCHO.retrieve_tile(tiles[0],"full") 
    col = X.columns[f_index]

    a = (X[col]).hist(bins=100,ax=ax[0,0],density=True)  
    a.set_title(tiles[0])


    X,y = CARPYNCHO.retrieve_tile(tiles[1],"full") 
    a = (X[col]).hist(bins=100,ax=ax[0,1],density=True)  
    a.set_title(tiles[1])

    X,y = CARPYNCHO.retrieve_tile(tiles[2],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,0],density=True)  
    a.set_title(tiles[2])

    X,y = CARPYNCHO.retrieve_tile(tiles[3],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,1],density=True)  
    a.set_title(tiles[3])

    fig.suptitle(col)
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_INSPECTION+"VisualizandoVariables/feature_comparison_"+str(f_index)+".png",bbox_inches='tight')


def compare_all_features():
    """
    Plot frequency histograms of all features for different tiles
    """
    for i in range(62):
        compare_same_feature_different_tiles(f_index=i)
