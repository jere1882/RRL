""" VISUALIZATION AND INSPECTION """
exec(open("/home/jere/Dropbox/University/Tesina/src/section7.py").read())
from numpy import inf
results_folder_inspection = "/home/jere/Desktop/section8/"


################### VARIABLE IMPORTANCE BY STATISTICAL TESTS #####################
def calculate_univariate_importance(train="b278",method="chi2"):
    
    X,y = retrieve_tile(train,"full") 

    if (method=="f_classif" or method=="all"):
        selector = SelectKBest(f_classif, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", selector )])
        clf.fit(X, y)
        
        with open(results_folder_inspection+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   
            
    
    if (method=="mutual_info_classif" or method=="all"):
        selector = SelectKBest(mutual_info_classif, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_selector", selector )])
        clf.fit(X, y)
        
        with open(results_folder_inspection+"VariableImportance/mutual_info_classif_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   
    
    if (method=="chi2" or method=="all"):
        selector = SelectKBest(chi2, k=4)
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("feature_selector", selector )])
        clf.fit(X, y)   

        with open(results_folder_inspection+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'wb') as output:
            pickle.dump(selector,output, pickle.HIGHEST_PROTOCOL)   

def calculate_all_importances():
    for tile in ["b234","b278","b261","b360"]:
        calculate_univariate_importance(train="b278",method="all")
    
def plot_univariate_importance_pvalue(train="b278"):
        
    with open(results_folder_inspection+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_chi = pickle.load(output)       
    
    with open(results_folder_inspection+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_f = pickle.load(output)       

    X,y = retrieve_tile(train,"full")
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

    plt.savefig(results_folder_inspection+"VariableImportance/test="+train+"_variable_importance_pvalues.png",bbox_inches='tight')
    
def plot_univariate_importance_scores(train="b278"):
        
    with open(results_folder_inspection+"VariableImportance/chi2_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_chi = pickle.load(output)       

    with open(results_folder_inspection+"VariableImportance/f_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_f = pickle.load(output)       

    with open(results_folder_inspection+"VariableImportance/mutual_info_classif_SELECTOR_"+"train="+train+".pkl", 'rb') as output:
        selector_m = pickle.load(output)   
        
    X,y = retrieve_tile(train,"full")
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

    plt.savefig(results_folder_inspection+"VariableImportance/test="+train+"_variable_importance_scores.png",bbox_inches='tight')

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

    plt.savefig(results_folder_inspection+"VariableImportance/test="+train+"_variable_importance_ranking.png",bbox_inches='tight')
    
    
def calculate_all_plots_univariate_importance():
    for tile in ["b234","b278","b261","b360"]:
        plot_univariate_importance_scores(tile)
        plot_univariate_importance_pvalue(tile)


################### VARIABLE IMPORTANCE BY ML METHODS #####################
                 
def calculate_ml_importance(train="b278"):

    X,y=retrieve_tile(train)

    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    svc = LinearSVC(verbose=3, max_iter=10000, C=get_optimal_parameters_p("svml")["C"], dual=False)
    clf2 = Pipeline([
    ("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
    ('scaler', StandardScaler()),
    ('clf', svc) ])
    clf2.fit(X,y)
    
    persist = (svc.coef_,clf.feature_importances_)
    
    with open(results_folder_inspection+"VariableImportance/ml_importance_"+"train="+train+".pkl", 'wb') as output:
        pickle.dump(persist,output, pickle.HIGHEST_PROTOCOL)   

def plot_ml_importance(train="b278"):

    with open(results_folder_inspection+"VariableImportance/ml_importance_"+"train="+train+".pkl", 'rb') as output:
        persist = pickle.load(output)   
    fig, ax = plt.subplots(figsize=(19,8))

    X,y=retrieve_tile(train)
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

    plt.savefig(results_folder_inspection+"VariableImportance/test="+train+"_ML_variable_importance_scores.png",bbox_inches='tight')

def calculate_all_plots_ml_importance():
    for tile in ["b234","b278","b261","b360"]:
        plot_ml_importance(tile)
        
############################ CORRELATIONS ########################################


def calculate_correlation_matrix(tile="b278",method='pearson'):
    
    fig, ax = plt.subplots(figsize=(15,10))
    X,y = retrieve_tile(tile)
    corr_df =  X.corr(method=method) 
    df_lt = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool))
    X_indices = np.arange(X.shape[-1])
    
    hmap=sns.heatmap(df_lt,cmap="Spectral",xticklabels=X_indices, yticklabels=X_indices,vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.title('Coeficientes de correlacion '+ method  +' en el tile '+tile)
    plt.savefig(results_folder_inspection+"Correlations/"+method+"_"+tile+"_MATRIX.png",bbox_inches='tight')

    with open(results_folder_inspection+"Correlations/train="+tile+"_"+method+"_correlation_matrix.pkl", 'wb') as output:
        pickle.dump(corr_df,output, pickle.HIGHEST_PROTOCOL)
        
def calculate_all_correlation_matrixes():
    for method in ["spearman","kendall"]:
        for tile in ["b234","b278","b261","b360"]:
            calculate_correlation_matrix(tile,method)
            
def calculate_correlations(threshold=0.5,train="b278",test=["b234","b278","b261","b360"],method="pearson",kernel="linear"):

    X,y = retrieve_tile(train)
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
                Xt,yt = retrieve_tile(test_tile)
                Xt = Xt.drop(dropped_columns,axis=1)

                decs  = clf.decision_function(Xt)
                p,r,t = metrics.precision_recall_curve(yt,decs)
                precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
                recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
                precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
                robust_auc = auc(recall_interpolated, precision_interpolated)

                curves[test_tile] = (p,r,robust_auc)

            # Save results in persistent object:  (Correlated_feature,Removed_feature,correlation_value,
            persist[n_dropped_features] = (feature_1,feature_2,max_val,curves)

    with open(results_folder_inspection+"Correlations/train="+train+"_"+method+"_"+kernel+"_persisted_data.pkl", 'wb') as output:
        pickle.dump(persist,output, pickle.HIGHEST_PROTOCOL)   

def generate_correlation_data(kernel="linear",method="pearson"):
    calculate_correlations(threshold=0,train="b234",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b278",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b261",test=["b234","b278","b261","b360"],method=method,kernel=kernel)
    calculate_correlations(threshold=0,train="b360",test=["b234","b278","b261","b360"],method=method,kernel=kernel)    

def analyse_correlations(method="pearson",kernel="linear"):

    diffs = {}

    for train in ["b234","b278","b261","b360"]:

        with open(results_folder_inspection+"Correlations/train="+train+"_"+method+"_"+kernel+"_persisted_data.pkl", 'rb') as output:
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
            ax.plot(domain, codomain) 
            horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in domain])
            if (kernel=="linear"):
                label = "SVM Lineal Baseline"
            elif (kernel=="rbf"):
                label = "SVM RBF Baseline"
            ax.plot(domain, horiz_line_data, 'r--',label=label) 
            plt.xlabel('Número de atributos utilizados')
            plt.ylabel('Robust AUC-PRC')
            plt.title('Train '+train+' - Test '+test)
            ax.set_ylim([0,.55])
            plt.savefig(results_folder_inspection+"Correlations/"+method+"_"+kernel+"_INDIVIDUAL_CURVES_"+"train="+train+"test="+test+".png",bbox_inches='tight')
            plt.close(fig)


            fig, ax = plt.subplots()
            domain = list(correlations.values())
            ax.plot(domain, list(aucs.values())) 
            horiz_line_data = np.array([get_baseline_preprocessing_stage(train,test,kernel) for i in domain])
            if (kernel=="linear"):
                label = "SVM Lineal Baseline"
            elif (kernel=="rbf"):
                label = "SVM RBF Baseline"
            ax.set_ylim([0,.55])
            ax.plot(domain, horiz_line_data, 'r--',label=label) 
            plt.xlabel('Correlation threshold')
            plt.ylabel('Robust AUC-PRC')
            plt.title('Train '+train+' - Test '+test)
            plt.savefig(results_folder_inspection+"Correlations/"+method+"_"+kernel+"_INDIVIDUAL_CURVES_CORR_"+"train="+train+"test="+test+".png",bbox_inches='tight')
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
    plt.ylabel('Ganancia en R-AUC-PRC respecto al baseline')
    leg = ax.legend()
    
    plt.savefig(results_folder_inspection+"Correlations/"+method+"_"+kernel+"_CORRELATIONS_BIG_PICTURE.png",bbox_inches='tight')
         
    
########### VISUALIZATION ############

def plot_fetures_distribution(tile="b278"):

    X,y = retrieve_tile(tile,"full") 
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
    plt.savefig(results_folder_inspection+"VisualizandoVariables/allfeatures_"+tile+".png",bbox_inches='tight')
    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(15,17))


def compare_same_feature_different_tiles(f_index=0,tiles=["b234","b360","b278","b261"]):


    fig, ax = plt.subplots(2, 2, sharey=True,sharex=True,figsize=(15.0, 10.0))

    X,y = retrieve_tile(tiles[0],"full") 
    col = X.columns[f_index]

    a = (X[col]).hist(bins=100,ax=ax[0,0],density=True)  
    a.set_title(tiles[0])


    X,y = retrieve_tile(tiles[1],"full") 
    a = (X[col]).hist(bins=100,ax=ax[0,1],density=True)  
    a.set_title(tiles[1])

    X,y = retrieve_tile(tiles[2],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,0],density=True)  
    a.set_title(tiles[2])

    X,y = retrieve_tile(tiles[3],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,1],density=True)  
    a.set_title(tiles[3])

    fig.suptitle(col)
    plt.savefig(results_folder_inspection+"VisualizandoVariables/feature_comparison_"+str(f_index)+".png",bbox_inches='tight')


def compare_all_features():
    for i in range(62):
        compare_same_feature_different_tiles(f_index=i)


def get_feature_selected_tile(tile_to_retrieve="b234",kernel="linear",ranking_to_use="b234",rate="full"):
    with open(results_folder_inspection+"VariableImportance/mutual_info_classif_SELECTOR_"+"train="+ranking_to_use+".pkl", 'rb') as output:
        selector_m = pickle.load(output)
    k = get_optimal_parameters_fs(kernel)["k"]
    features = selector_m.scores_.argsort()[-k:][::-1]
    X,y = retrieve_tile(tile_to_retrieve,rate)
    return X.iloc[:,features],y
