exec(open("/home/jere/Dropbox/University/Tesina/src/src/section9.py").read())
#### EXPLORE THE EFFECT THAT DIFFERENT HYPERPARAMETERS HAVE ON SVM-RBF

results_folder_potential= "/home/jere/Desktop/section10/"
import seaborn as sb
import matplotlib.style
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


###################### PART 1: OVERFITTING HYPERPARAMETERS ##################
def get_range(param,legacy=False):
    if param=="C":
        if legacy:
            return np.logspace(-5, 12, 18)
        else:
            return np.logspace(-4, 12, 17)
    elif param=="gamma":
        if legacy:
            return np.logspace(-15, 4,20)
        else:
            return np.logspace(-11, 1,13)

def explore_rbf_potential(train="b278",test="b234",kernel="rbf"):
  
    X,y=get_feature_selected_tile(train,kernel,train,"full")   
    Xt,yt=get_feature_selected_tile(test,kernel,train,"full")  
    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    
    for c in get_range("C"):
        for gamma in get_range("gamma"):
            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                    ("scaler",StandardScaler()),
                    ("feature_map", Nystroem(gamma=gamma, n_components=300)),
                    ("svm", LinearSVC(dual=False,max_iter=100000,C=c))])

            clf.fit(X,y)
            decs  = clf.decision_function(Xt)
            
            s = stats.describe(decs)
            if ( abs(s.minmax[0] - s.minmax[1]) < 0.1 ):
                scores[(c,gamma)] = -1
                continue
            
            p,r,t = metrics.precision_recall_curve(yt,decs)

            precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            scores[(c,gamma)] = auc(recall_interpolated, precision_interpolated)

    with open(results_folder_potential+ "_svmk_train="+train+"test="+test+"_scores.pkl", 'wb') as s_file:
        pickle.dump(scores,s_file)
    return scores  

def calculate_all_potential_rbf():
    scores1= explore_rbf_potential(train="b278",test="b234")
    scores2= explore_rbf_potential(train="b278",test="b261")
    scores3= explore_rbf_potential(train="b234",test="b261")
    scores4= explore_rbf_potential(train="b234",test="b360")
    scores5= explore_rbf_potential(train="b261",test="b360")
    scores6= explore_rbf_potential(train="b261",test="b278")
    scores7= explore_rbf_potential(train="b360",test="b278")
    scores8= explore_rbf_potential(train="b360",test="b234")     
    scores8= explore_rbf_potential(train="b278",test="b360")     
    scores8= explore_rbf_potential(train="b234",test="b278")     
    scores8= explore_rbf_potential(train="b261",test="b234")     
    scores8= explore_rbf_potential(train="b360",test="b261")

def get_pr_curve(train="b278",test="b234",c=0.1,gamma=0.1,kernel="rbf"):

    try:
        with open(results_folder_potential+"curves/_svmk_train="+train+"test="+test+"C=+"+str(c)+"g="+str(gamma)+"_curves.pkl", 'rb') as s_file:
            scores = pickle.load(s_file) 
        return scores
        
    except:

        print("P-R curve has never been calculated. Calculating it now")
        
        X,y=get_feature_selected_tile(train,kernel,train,"full")   
        Xt,yt=get_feature_selected_tile(test,kernel,train,"full")  

        clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                ("scaler",StandardScaler()),
                ("feature_map", Nystroem(gamma=gamma, n_components=300)),
                ("svm", LinearSVC(dual=False,max_iter=100000,C=c))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        scores = (p,r)
        #Cache it
        with open(results_folder_potential+"curves/_svmk_train="+train+"test="+test+"C=+"+str(c)+"g="+str(gamma)+"_curves.pkl", 'wb') as s_file:
            pickle.dump(scores,s_file)

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        print("AUC=",auc(recall_interpolated, precision_interpolated))
            
        return(p,r)
    
def plot_rbf_potential(train="b278",test="b234",legacy=False):

    #try:
        
        if (legacy):
            prefix = "legacy/"
        else:
            prefix = ""

        with open(results_folder_potential+prefix+ "_svmk_train="+train+"test="+test+"_scores.pkl", 'rb') as s_file:
            scores = pickle.load(s_file)

        c_values =  get_range("C",legacy)
        g_values =  get_range("gamma",legacy)
        perf = np.zeros((len(c_values),len(g_values)))

        for i in range(0,len(c_values)-1):
            c = c_values[i]
            for j in range(0,len(g_values)-1):
                gamma = g_values[j]
                if scores[(c,gamma)]!=-1:
                    perf[i,j] = scores[(c,gamma)]
                else:
                    perf[i,j] = 0

        perf = pd.DataFrame(perf)
        rf_auc = get_baseline_preprocessing_stage(train,test,"rf")

        # PLOT THE COLORS
        fig, ax = plt.subplots(figsize=(11, 9))
        hmax = 0.6#max(rf_auc,np.max(perf.values))

        sb.heatmap(perf,square=True, cmap="magma", linewidth=.3, linecolor='w',vmax=hmax,
                    cbar_kws={ 'label': 'R-AUCPRC' } , ax=ax)

        # PLOT HATCHED CELLS
        zm = np.ma.masked_less(perf.values, rf_auc-0.05)
        x= np.arange(len(perf.columns)+1)
        y= np.arange(len(perf.index)+1)
        mpl.rcParams['hatch.color'] = "white"        
        ax.pcolor(x,y,zm, hatch='//', alpha=0)

        xlabels = [ "{:.0e}".format(x) for x in c_values ]
        ylabels = [ "{:.0e}".format(x) for x in g_values ]

        plt.yticks(np.arange(len(c_values))+.5, labels=xlabels,rotation=60)
        plt.xticks(np.arange(len(g_values))+.5, labels=ylabels, rotation=45)
        # PLOT THE RF TICK
        cax = plt.gcf().axes[-1]
        cax.hlines(y=rf_auc, xmin=0, xmax=1, colors = 'lawngreen', linewidth = 4, linestyles = 'solid',label="RF")

        cax = plt.gcf().axes[-1]
        leaked_svm = np.max(perf.values)
        cax.hlines(y=leaked_svm, xmin=0, xmax=1, colors = 'cyan', linewidth = 4, linestyles = 'solid',label="leaked_svm")
        print("SVM-OPT-AUC", leaked_svm)

        cax = plt.gcf().axes[-1]
        optimized_svmk =  get_baseline_fs_stage(train,test,"svmk")
        cax.hlines(y=optimized_svmk, xmin=0, xmax=1, colors = 'dodgerblue', linewidth = 4, linestyles = 'solid',label="optimized_svmk")
        print("SVM-noOPT-AUC", optimized_svmk)

        cax = plt.gcf().axes[-1]
        optimized_svml = get_baseline_imb_stage(train,test,"svml")
        cax.hlines(y=optimized_svml, xmin=0, xmax=1, colors = 'white', linewidth = 4, linestyles = 'solid',label="optimized_svml")

        # axis labels
        plt.ylabel('C')
        plt.xlabel('gamma')
        
        # ticks
        #if (train=="b234" and test=="b261"):
        #    cbar = ax.collections[0].colorbar
        #   cbar.set_ticks([0,optimized_svml,optimized_svmk,leaked_svm,rf_auc,hmax])
        #   cbar.set_ticklabels(['0','SVML','SVMK','SVMK-opt', 'RF', '0.6'])
        #else:
        #    cbar = ax.collections[0].colorbar
        #    cbar.set_ticks([0,hmax])
        #    cbar.set_ticklabels(['0', '0.6'])
                      
        # title
        title = ('Robust AUCPRC in test. Train: '+ train +' Test: ' + test).upper()
        plt.title(title, loc='left')

        plt.savefig(results_folder_potential+"heatmap_train="+train+"_test="+test+".png",bbox_inches='tight')
    

        plt.close("all")
        fig, ax = plt.subplots()
        
        # GET RANDOM FOREST DATA FROM PREPROCESSING STAGE
        with open(results_folder_preproces+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
            rf_curves = pickle.load(input_file)
        p,r = rf_curves["rf"]
        ax.plot(r,p, label="Random Forest",color="lawngreen")

        # NOW, PLOT THE BEST P-R CURVES OBTAINED IN THIS SECTION
        best_key = max(scores,key=scores.get)
        opt_c = best_key[0]
        opt_g = best_key[1]
        (p,r) = get_pr_curve(train,test,opt_c,opt_g,"svmk")

        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        print("SVM-OPT-AUC", auc(recall_interpolated, precision_interpolated))

        ax.plot(r,p, label="optimal SVM RBF",color="cyan")
        
        # GET SVM-K DATA FROM FEATURE SELECTION STAGE
        with open(results_folder_dimensionality_reduction+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
            curves = pickle.load(input_file)
        p,r = curves["svmk"]
        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        print("SVM-NO-OPT-AUC", auc(recall_interpolated, precision_interpolated))
        ax.plot(r,p, label="RBF SVM",color="dodgerblue")

        plt.title('Train ' + train + "- Test" + test)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        
        # GET SVM-L DATA FROM IMBLEARNING STAGE
        with open(results_folder_imbalance+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
            curves = pickle.load(input_file)
            
        p,r = curves["svml"]
        ax.plot(r,p, label="Linear SVM",color="black")
        if (train=="b234" and test=="b261"):
            leg = ax.legend();

        plt.savefig(results_folder_potential+"best-train="+train+ "test="+test+".png",bbox_inches='tight')

    #except:
      #  print("Unable to generate heatmap for ",train,test)
      #  pass
            
        
def calculate_all_grids(legacy=False):
    scores1= plot_rbf_potential(train="b278",test="b234",legacy=legacy)
    scores2= plot_rbf_potential(train="b278",test="b261",legacy=legacy)
    scores3= plot_rbf_potential(train="b234",test="b261",legacy=legacy)
    scores4= plot_rbf_potential(train="b234",test="b360",legacy=legacy)
    scores5= plot_rbf_potential(train="b261",test="b360",legacy=legacy)
    scores6= plot_rbf_potential(train="b261",test="b278",legacy=legacy)
    scores7= plot_rbf_potential(train="b360",test="b278",legacy=legacy)
    scores8= plot_rbf_potential(train="b360",test="b234",legacy=legacy)    
    scores8= plot_rbf_potential(train="b278",test="b360",legacy=legacy)      
    scores8= plot_rbf_potential(train="b234",test="b278",legacy=legacy)      
    scores8= plot_rbf_potential(train="b261",test="b234",legacy=legacy)      
    scores8= plot_rbf_potential(train="b360",test="b261",legacy=legacy)  

###################### PART 2: IDENTIFYING COVARIATE DRIFT ######################

###  IDENTIFY DRIFT USING HIGH DIMENSIONAL CLASSIFIER
def calculate_covariate_shift(tile1="b234",tile2="b360"):
    kernel="rbf" #This only matters for getting the optimal feature selection subset

    # Let's use tile1 feature selection for this.
    X1,y1=get_feature_selected_tile(tile1,kernel,tile1,"full")   
    X2,y2=get_feature_selected_tile(tile2,kernel,tile1,"full")  
    
    data = calculate_covariate_shift_internal(X1,y1,X2,y2)
    
    with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'wb') as s_file:
        pickle.dump(data,s_file)

def calculate_covariate_shift_internal(X1,y1,X2,y2,classifiers=["rf","lr"]):

    # Let's combine both datasets. Lets set class 1 for those who come from tile 1 ; class 2 for those who come from tile 2
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n  = min(n1,n2)

    X = pd.concat([X1.sample(n=n),X2.sample(n=n)])
    y = [1 for y in range(0,n)] + [0 for y in range(0,n)]

    # Use 75% of data to fit a RF classifier and 25% of data to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    data = {}
    ######### RANDOM FOREST #########
    if "rf" in classifiers:
        model = RandomForestClassifier(n_estimators = 50, max_depth = 5 , min_samples_leaf = 5)
        model.fit(X_train,y_train)

        y_scores = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test,y_scores)

        y_pred = model.predict(X_test)
        print("RANDOM FOREST RESULTS")
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        data["rf_predictions"] = y_pred
        data["rf_scores"] = y_scores
        data["rf_auc"] = roc
        data["rf_accuracy"] = metrics.accuracy_score(y_test, y_pred)
        data["rf_importance"] = model.feature_importances_
    
    ########## LINEAR REGRESSION ##########
    if "lr" in classifiers:
        m = LogisticRegression(max_iter=100000, dual=False)

        clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
            ("scaler",StandardScaler()),
            ("lr", m)])
                        
        clf.fit(X_train, y_train)
        
        y_scores = clf.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test,y_scores)

        y_pred = clf.predict(X_test)
        print("LOGISTIC REGRESSION RESULTS")
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        data["lr_predictions"] = y_pred
        data["lr_scores"] = y_scores
        data["lr_auc"] = roc
        data["lr_accuracy"] = metrics.accuracy_score(y_test, y_pred)
        data["lr_importance"] = m.coef_
    # THE CALLER IS RESPONSIBLE FOR PERSISTING THE DATA
    return data
        

def calculate_covariate_shift_all_tiles():
    scores1= calculate_covariate_shift(tile1="b278",tile2="b234")
    scores2= calculate_covariate_shift(tile1="b278",tile2="b261")
    scores3= calculate_covariate_shift(tile1="b234",tile2="b261")
    scores4= calculate_covariate_shift(tile1="b234",tile2="b360")
    scores5= calculate_covariate_shift(tile1="b261",tile2="b360")
    scores6= calculate_covariate_shift(tile1="b261",tile2="b278")
    scores7= calculate_covariate_shift(tile1="b360",tile2="b278")
    scores8= calculate_covariate_shift(tile1="b360",tile2="b234")    
    scores8= calculate_covariate_shift(tile1="b278",tile2="b360")      
    scores8= calculate_covariate_shift(tile1="b234",tile2="b278")      
    scores8= calculate_covariate_shift(tile1="b261",tile2="b234")      
    scores8= calculate_covariate_shift(tile1="b360",tile2="b261")  
    scores2= calculate_covariate_shift(tile1="b278",tile2="b278")
    scores2= calculate_covariate_shift(tile1="b234",tile2="b234")
    scores2= calculate_covariate_shift(tile1="b261",tile2="b261")
    scores2= calculate_covariate_shift(tile1="b360",tile2="b360")
    
def get_covariate_shift_auc(tile1,tile2,method="rf",metric="accuracy"):
    with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'rb') as s_file:
        data = pickle.load(s_file)
    key = method+"_"+metric
    return (data[key])
    
def print_covariate_shift_matrix(method="rf",metric="accuracy"):
    
    tiles = ["b234","b261","b278","b360"]

    perf = np.zeros((len(tiles),len(tiles)))

    for i in range(0,4):
        for j in range(0,4):
            perf[i,j] = get_covariate_shift_auc(tiles[i],tiles[j],method,metric)
            
    fig, ax = plt.subplots()

    vmin=0.45


    if (metric=="accuracy"):
        metric_string = metric
    else:
        metric_string="Area under ROC curve"
    sb.heatmap(perf,square=True, cmap="magma", linewidth=.3,annot=True,linecolor='w',vmin=vmin,vmax=1,
            cbar_kws={ 'label': metric_string } , ax=ax)

    plt.yticks(np.arange(len(tiles))+.5, labels=tiles)
    plt.xticks(np.arange(len(tiles))+.5, labels=tiles)
    
    if (method=="rf"):
        method="Random Forest"
    else:
        method="Linear Regression"
        
    plt.title("Presencia de covariate shift. Clasificador "+method)
    
    plt.savefig(results_folder_potential+"CS-Method+"+method+"metric"+metric+".png",bbox_inches='tight')

########################## CORRECTING COVARIATE DRIFT ########################

# Print the importance based on high-dimensional classifier
def get_covariate_shift_importance(tile1,tile2,method="rf"):

    X1,y1=get_feature_selected_tile(tile1,"rbf",tile1,"full")   

    with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'rb') as s_file:
        data = pickle.load(s_file)

    key = method+"_importance"

    if (method=="rf"):
        scores = data[key]
    else:
        scores = np.abs(data[key][0])
    temp = (-scores).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))

    print("Ranking in contribution:")
    for i in temp:
        print(X1.columns[i])
        
# Plot the importance obtained using high dimensional classifier
def plot_covariance_shift_scores(tile1="b360",tile2="b234"):

    X1,y1=get_feature_selected_tile(tile1,"rbf",tile1,"full")   

    with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'rb') as s_file:
        data = pickle.load(s_file)
        
    scores_rf = data["rf_importance"]
    temp = (-scores_rf).argsort()
    ranks_rf = np.empty_like(temp)
    ranks_rf[temp] = np.arange(len(scores_rf))

    scores_lr = np.abs(data["lr_importance"][0])
    scores_lr_adj = [ np.e**x for x in scores_lr ]
    temp = (-scores_lr).argsort()
    ranks_lr = np.empty_like(temp)
    ranks_lr[temp] = np.arange(len(scores_lr))
            

    X_indices = np.arange(X1.shape[-1])
    fig, ax = plt.subplots(figsize=(15,15))

    scores_rf_scaled = [ x/sum(scores_rf) for x in scores_rf]
    plt.bar(X_indices , scores_rf_scaled, width=.2,  label="RF Gini")
    scores_lr_scaled = [ x/sum(scores_lr) for x in scores_lr]
    plt.bar(X_indices +0.22, scores_lr_scaled , width=.2,  label="LR")
    plt.xlim([-1, 50])

    plt.xlabel('Feature')
    plt.ylabel('Normalized importance score')
    plt.xticks(X_indices,X1.columns,rotation=90)
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()
    
    
#### Calculate feature importance using 1-dimensional classifier
def calculate_covariate_shift_single(tile1="b360",tile2="b234",classifiers=["rf","lr"]):
    kernel="rbf" #This only matters for getting the optimal feature selection subset

    # Let's use tile1 feature selection for this.
    X1,y1=get_feature_selected_tile(tile1,kernel,tile1,"full")   
    X2,y2=get_feature_selected_tile(tile2,kernel,tile1,"full")  
        
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n  = min(n1,n2)

    X = pd.concat([X1.sample(n=n),X2.sample(n=n)])
    y = [1 for y in range(0,n)] + [0 for y in range(0,n)]

    # Use 75% of data to fit a RF classifier and 25% of data to test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    data_auc_l = {}
    data_acc_l = {}
    data_auc_r = {}
    data_acc_r = {}
    
    for feature in X.columns:
        X_tr_tmp =   np.array(X_train[feature]).reshape(-1, 1)
        X_t_tmp  =    np.array(X_test[feature]).reshape(-1, 1)
    
        ######### RANDOM FOREST #########
        if "rf" in classifiers:
            model = RandomForestClassifier(n_estimators = 50, max_depth = 5 , min_samples_leaf = 5)
            model.fit(X_tr_tmp.reshape(-1, 1),y_train)


            y_scores = model.predict_proba(X_t_tmp)[:,1]
            roc = roc_auc_score(y_test,y_scores)

            y_pred = model.predict(X_t_tmp)

            data_auc_l[feature] = roc
            data_acc_l[feature] = metrics.accuracy_score(y_test, y_pred)
    
        ########## LINEAR REGRESSION ##########
        if "lr" in classifiers:
            m = LogisticRegression(max_iter=100000, dual=False)

            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                ("scaler",StandardScaler()),
                ("lr", m)])
                            
            clf.fit(X_tr_tmp, y_train)
            
            y_scores = clf.predict_proba(X_t_tmp)[:,1]
            roc = roc_auc_score(y_test,y_scores)

            y_pred = clf.predict(X_t_tmp)

            data_auc_r[feature] = roc
            data_acc_r[feature] = metrics.accuracy_score(y_test, y_pred)

    ans = (data_auc_l,data_acc_l,data_auc_r,data_acc_r)
    
    with open(results_folder_potential+"shift_importance_individual="+tile1+"test="+tile2+".pkl", 'wb') as s_file:
        pickle.dump(ans,s_file)
        
    return ans
    

# Plot the importance obtained using 1-dimensional classifier
def plot_calculate_covariate_shift_single(tile1="b360",tile2="b234"):
    with open(results_folder_potential+"shift_importance_individual="+tile1+"test="+tile2+".pkl", 'rb') as s_file:
        ans = pickle.load(s_file)
            
    fig, ax = plt.subplots(figsize=(15,15))
    X_indices = np.arange(len(ans[0].keys()))

    plt.bar(X_indices , ans[0].values(), width=.2,  label="LR AUC")
    plt.bar(X_indices+0.22 , ans[2].values(), width=.2,  label="RF AUC")

    plt.xlabel('Feature')
    plt.ylabel('Area under ROC in test')
    plt.xticks(X_indices,ans[0].keys(),rotation=90)
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

    fig, ax = plt.subplots(figsize=(15,15))
    X_indices = np.arange(len(ans[0].keys()))

    plt.bar(X_indices , ans[1].values(), width=.2,  label="LR Accuracy")
    plt.bar(X_indices+0.22 , ans[3].values(), width=.2,  label="RF Accuracy")

    plt.xlabel('Feature')
    plt.ylabel('Accuracy in test')
    plt.xticks(X_indices,ans[0].keys(),rotation=90)
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

######################## CORRECT DRIFT BY REMOVING FEATURES: Estimating decrease in shift   ######################

# Number of features from the top features for distinguishing RRL and no-RRL
# that won't be allowed to be removed
def get_number_protected_features():
    return 5
    
#### Remove features using high-dimensional classifier && recalculate whether drift decreased
def remove_covariate_shift_top_and_recalculate_shift(tile1="b360",tile2="b234",method="rf",single=False):
    kernel="rbf" #This only matters for getting the optimal feature selection subset

    X1,y1=get_feature_selected_tile(tile1,kernel,tile1,"full")   
    Xt,yt=get_feature_selected_tile(tile2,kernel,tile1,"full")   


    if (single==False):  # Use high dimensional classifier data
        with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'rb') as s_file:
            data = pickle.load(s_file)

        key = method+"_importance"

        if (method=="rf"):
            scores = data[key]
        else:
            scores = np.abs(data[key][0])
            
    else:   # Use 1-dimentional classifier data
        with open(results_folder_potential+"shift_importance_individual="+tile1+"test="+tile2+".pkl", 'rb') as s_file:
            ans = pickle.load(s_file)

        if (method=="rf"):
            scores = ans[2].values()
        else:
            scores = ans[0].values()
        
        scores = np.array(list(scores))
    temp = (-scores).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))

    accum_acc = {}
    accum_auc = {}

    for j in range(0,40):
            dropped = []
            for i in temp:
                if (len(dropped)<j):
                    if i>get_number_protected_features():
                        print("Removing ",X1.columns[i])
                        dropped = dropped + [X1.columns[i]]
                    else:
                        print("Ommitting the removal of ",X1.columns[i]," because it is in the top-10")
                        continue
                
            Xtr_temp = X1.drop(columns=dropped)
            Xt_temp = Xt.drop(columns=dropped)
        
            data2 = calculate_covariate_shift_internal(Xtr_temp,y1,Xt_temp,yt,[method])
            
            key1 = method+"_accuracy"
            key2 = method+"_auc"
            accum_acc[j] = data2[key1]
            accum_auc[j] = data2[key2]

    
    ret = (accum_acc, accum_auc,get_number_protected_features())
    
    with open(results_folder_potential+"shift_decrease="+tile1+"test="+tile2+"method"+method+"_single="+str(single)+".pkl", 'wb') as s_file:
        pickle.dump(ret,s_file)
    return(ret)


### Plot the reduction in drift after removing drifting features
def plot_remove_covariate_shift_top_and_recalculate_shift(tile1="b360",tile2="b234",single=False):
    
    with open(results_folder_potential+"shift_decrease="+tile1+"test="+tile2+"methodrf_single="+str(single)+".pkl", 'rb') as s_file:
        ret_rf = pickle.load(s_file)
    with open(results_folder_potential+"shift_decrease="+tile1+"test="+tile2+"methodlr_single="+str(single)+".pkl", 'rb') as s_file:
        ret_lr = pickle.load(s_file)
    fig, ax = plt.subplots()

    ax.plot(ret_rf[0].keys(),ret_rf[0].values(),label="RF, Accuracy")
    ax.plot(ret_rf[1].keys(),ret_rf[1].values(),label="RF, AUC-ROC")
    ax.plot(ret_lr[0].keys(),ret_lr[0].values(),label="LR, Accuracy")
    ax.plot(ret_lr[1].keys(),ret_lr[1].values(),label="LR, AUC-ROC")

    plt.xlabel('Número de atributos eliminados')
    #plt.xticks(np.arange(len(ret_rf[0].keys())), labels=ret_rf[0].keys())
    plt.ylabel('Score del clasificador en test')
    leg = ax.legend();

    plt.title('Covariate shift. Tiles ' + tile1 + " y " + tile2)

    plt.savefig(results_folder_potential+"CS-Reduction-tile1"+tile1+"tile2"+tile2+"_single="+str(single)+".png",bbox_inches='tight')


################## CORRECT DRIFT BY REMOVING FEATURES: Checking if original problem improved   ##############

def remove_covariate_shift_top_and_recalculate_rrl_classification_internal(tile1,tile2,scores,label):
    
    kernel="rbf" #This only matters for getting the optimal feature selection subset
    X1,ytr=get_feature_selected_tile(tile1,kernel,tile1,"full")   
    Xt,yt=get_feature_selected_tile(tile2,kernel,tile1,"full")   

    temp = (-scores).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))

    results = {}

    for j in range(0,20):
            dropped = []
            for i in temp:
                if (len(dropped)<j):
                    if i>get_number_protected_features():
                        print("Removing ",X1.columns[i])
                        dropped = dropped + [X1.columns[i]]
                    else:
                        print("Ommitting the removal of ",X1.columns[i]," because it is in the top-10")
                        continue
                
            Xtr_temp = X1.drop(columns=dropped)
            Xt_temp = Xt.drop(columns=dropped)
        
            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                    ("scaler",StandardScaler()),
                    ("feature_map", Nystroem(gamma=get_optimal_parameters_fs("svmk")["gamma"], n_components=300)),
                    ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svmk")["C"]))])
                    
            clf.fit(Xtr_temp,ytr)
            decs  = clf.decision_function(Xt_temp)
            p,r,t = metrics.precision_recall_curve(yt,decs)            
            results[j] = (p,r)
    
    with open(results_folder_potential+"shift_decrease_curves="+tile1+"test="+tile2+"method"+label+".pkl", 'wb') as s_file:
        pickle.dump(results,s_file)
    return(results)


#### Remove features using high-dimensional classifier && recalculate R-AUCPRC in RRL classification
def remove_covariate_shift_top_and_recalculate_rrl_classification(tile1="b360",tile2="b234",method="rf"):
    with open(results_folder_potential+ "train="+tile1+"test="+tile2+"_cov_shift.pkl", 'rb') as s_file:
        data = pickle.load(s_file)

    key = method+"_importance"

    if (method=="rf"):
        scores = data[key]
    else:
        scores = np.abs(data[key][0])
        
    data = remove_covariate_shift_top_and_recalculate_rrl_classification_internal(tile1,tile2,scores,method)
    
### Remove drifting features using 1-dimensional classifier ranking
def remove_covariate_shift_top_and_recalculate_rrl_classification_single(tile1="b360",tile2="b234",method="rf"):
    with open(results_folder_potential+"shift_importance_individual="+tile1+"test="+tile2+".pkl", 'rb') as s_file:
        ans = pickle.load(s_file)

    key = method+"_importance"

    if (method=="rf"):
        scores = ans[2].values()
    else:
        scores = ans[0].values()
        
    data = remove_covariate_shift_top_and_recalculate_rrl_classification_internal(tile1,tile2,np.array(list(scores)),method+"_single") 
    

### Plotting the change in performance
def plot_remove_covariate_shift_top_and_recalculate_rrl_classification(tile1="b360",tile2="b234",method="rf",single=False):
    
    if single:
        label = method + "_single"
    else:
        label = method
    
    with open(results_folder_potential+"shift_decrease_curves="+tile1+"test="+tile2+"method"+label+".pkl", 'rb') as s_file:
        results = pickle.load(s_file)

    aucs = {}
    aucs[0] = get_baseline_fs_stage(tile1,tile2,"rbf")
    for i in results.keys():
        (p,r) = results[i]
        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        aucs[i+1] = auc(recall_interpolated, precision_interpolated)
        
    fig, ax = plt.subplots()
    

    ax.plot(aucs.keys(),aucs.values())

    horiz_line_data = np.array([get_baseline_fs_stage(tile1,tile2,"rbf") for i in aucs.keys()])
    ax.plot(aucs.keys(), horiz_line_data, 'r--',label="Baseline SVM",color='g') 

    plt.xlabel('Número de atributos eliminados')
    plt.xticks(np.arange(len(aucs.keys())), labels=aucs.keys())
    plt.ylabel('R-AUCPRC')
    plt.title(" Tiles " + tile1 + " y " + tile2)
    plt.savefig(results_folder_potential+"ROC-Increase+"+label+"tile1"+tile1+"tile2"+tile2+".png",bbox_inches='tight')


    """  #In case you wanna inspect some of the P-R curves
    fig, ax = plt.subplots()

    with open(results_folder_dimensionality_reduction+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
        curves = pickle.load(input_file)
    p,r = curves["svmk"]
    ax.plot(r,p, label="RBF SVM",color="dodgerblue")


    (p,r) = results[4]
    ax.plot(r,p, label="RBF SVM not shifted",color="red")

    plt.title('Train ' + train + "- Test" + test)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    """



def calculate_all_data(tile1,tile2,method="rf"):
    calculate_covariate_shift_single(tile1=tile1,tile2=tile2,classifiers=[method])
    remove_covariate_shift_top_and_recalculate_rrl_classification_single(tile1=tile1,tile2=tile2,method=method)
    plot_remove_covariate_shift_top_and_recalculate_rrl_classification(tile1=tile1,tile2=tile2,method=method,single=True)
