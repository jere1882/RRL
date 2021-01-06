############### This code re-does the EXPERIMENT 2: Model selection ################
"""
    Description: Optimize each classifier parameters by doing k-fold-cross-validation
                 on a given tile.


    DIRECTORY LIST:
    experiments/
        |- rf/  => RANDOM FOREST RESULTS
        |- svm/ => LINEAR SVM RESULTS
        |- svm-k/ => SVM KERNEL APPROXIMATION RESULTS
    
    
    experiments/*/optimize_hyperparameters/ -> Here is where the results of cross validation parameters optimization are saved.
        |- cvobject_train={tile}{_rate_suffix}.pkl    -> Indivitual results
        |- train={tile}{_rate_suffix}test={tile}.png  -> Precision-recall curves in test
        |- testscores_train={tile}{_rate_suffix}.pkl  -> Scores on test tiles

        THE MOST IMPORTANT THING OF THIS FOLDER IS THE CROSS VALIDATION OBJECT

"""
####################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import auc
import pickle
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,MaxAbsScaler,MinMaxScaler, PowerTransformer, Normalizer, QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import tempfile
import shutil 
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from matplotlib.colors import Normalize
from sklearn.preprocessing import KBinsDiscretizer

n_jobs_global_param = 7 ## NUMBER OF PROCESSORS TO BE USED

exec(open("/home/jere/Dropbox/University/Tesina/src/common.py").read())

"""
Define custom metrics to be used as cross validation scores
"""

n_samples_prc = 1000 # Parameter for interpolated metrics

def auc_prc(y, y_pred):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    pr_auc  = auc(recall_fold, precision_fold)
    return pr_auc

def auc_interpolated_prc(y, y_pred):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    recall_interpolated    = np.linspace(0, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    pr_auc2 = auc(recall_interpolated, precision_interpolated)
    return pr_auc2

def precision_at_a_fixed_recall(y,y_pred,fixed_recall):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    pafr = max(precision_fold[recall_fold >= fixed_recall])
    return pafr
    
def precision_at_a_fixed_recall_interpolated(y,y_pred,fixed_recall):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    recall_interpolated    = np.linspace(0, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
    pafr = max(precision_interpolated[recall_interpolated >= fixed_recall])
    return pafr    
    
min_recall_global=0.3
def auc_interpolated_prc_robust(y, y_pred,min_recall=min_recall_global):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    recall_interpolated    = np.linspace(min_recall, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    pr_auc2 = auc(recall_interpolated, precision_interpolated)
    return pr_auc2


""" Get scorers to be used in cross validation to optimize hyperparameters.
- Scorers make use of need_treshold for SVM and needs_proba for RF
"""
def get_scorers(need_threshold = True):
    if (need_threshold):
        auc_prc_scorer  = make_scorer(auc_prc,needs_threshold=True)
        auc_iprc_scorer = make_scorer(auc_interpolated_prc,needs_threshold=True)
        pafr5_scorer    = make_scorer(precision_at_a_fixed_recall,fixed_recall=0.5,needs_threshold=True)
        pafr5_i_scorer  = make_scorer(precision_at_a_fixed_recall_interpolated,fixed_recall=0.5,needs_threshold=True)
        pafr9_scorer    = make_scorer(precision_at_a_fixed_recall,fixed_recall=0.9,needs_threshold=True)
        pafr9_i_scorer  = make_scorer(precision_at_a_fixed_recall_interpolated,fixed_recall=0.9,needs_threshold=True)
        auc_prc_robust  = make_scorer(auc_interpolated_prc_robust,needs_threshold=True)
    else:
        auc_prc_scorer  = make_scorer(auc_prc,needs_proba=True)
        auc_iprc_scorer = make_scorer(auc_interpolated_prc,needs_proba=True)
        pafr5_scorer    = make_scorer(precision_at_a_fixed_recall,fixed_recall=0.5,needs_proba=True)
        pafr5_i_scorer  = make_scorer(precision_at_a_fixed_recall_interpolated,fixed_recall=0.5,needs_proba=True)
        pafr9_scorer    = make_scorer(precision_at_a_fixed_recall,fixed_recall=0.9,needs_proba=True)
        pafr9_i_scorer  = make_scorer(precision_at_a_fixed_recall_interpolated,fixed_recall=0.9,needs_proba=True)
        auc_prc_robust  = make_scorer(auc_interpolated_prc_robust,needs_threshold=True)
        

    scorers = { #"auc_prc"    : auc_prc_scorer, DISABLED SINCE I'VE EMPIRICALLY VERIFIED IT IS EQUIVALENT TO aps
                #"auc_prci"   : auc_iprc_scorer, DISABLED SINCE I'VE EMPIRICALLY VERIFIED IT IS EQUIVALENT TO aps
                #"pafr5"      : pafr5_scorer,
                "pafr5i"     : pafr5_i_scorer,
                #"pafr9"      : pafr9_scorer,
                "pafr9i"     : pafr9_i_scorer,
                #"aps"        : 'average_precision'
                'auc_prc_r'   : auc_prc_robust
              }
              
    # !If you want to print the full precision-recall curve, add a scorer for each recall point you are interested in
    # This computationally expensive! GridSearchCV does not accept scorers returning non-scalar values.
    return scorers

test_scorers_names = ["auc_prc","auc_prci", "pafr5","pafr5i","pafr9","pafr9i","aps","auc_prc_r"]


""" ############################################# SHARED CODE ################################################## """

""" 
    Given an already trained GridSearch object, evaluates the best classifier in a set of tiles
    
    returns a dictionary with the resulting scores for each tile
    
    classifier: object implementing predic_proba or decision_func
    tiletest: may be either a real tile "b234" or a vtile in string "2" format
    
    SAVES THE SCORES TO experiments/"+folder+"/optimize_hyperparameters/testscores...
    
"""
def test_classifier(classifier,tilestest,folder,tiletrain="",trainrate="full",optional_suffix=""):
        
    scores_out = {}
    
    for tiletest in tilestest:
    
        plt.clf()

        # 0) Retrieve data
        Xtest, ytest = retrieve_tile(tiletest)

        # 1) Predict
        Xtest = Xtest.to_numpy()
        ytest = ytest.to_numpy()
        
        # Use predict_proba if available, else use decision function
        try:
            test_predictions = classifier.predict_proba(Xtest)[:,1]
        except:
            test_predictions = classifier.decision_function(Xtest)

        # 2) Calculate metrics
        pr_auc  = auc_prc(ytest,test_predictions)
        pr_auci = auc_interpolated_prc(ytest,test_predictions)
        pafr5   = precision_at_a_fixed_recall(ytest,test_predictions,0.5)
        pafr5i  = precision_at_a_fixed_recall_interpolated(ytest,test_predictions,0.5)
        pafr9   = precision_at_a_fixed_recall(ytest,test_predictions,0.9)
        pafr9i  = precision_at_a_fixed_recall_interpolated(ytest,test_predictions,0.9)
        aps     = metrics.average_precision_score(ytest,test_predictions)
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        prc = (precision,recall,thresh)
        scores_out[tiletest] = (pafr9, pafr9i, pafr5, pafr5i, aps, pr_auc, pr_auci,prc)
        
        # 3) [Optional] Plot P-R Curve
    
        precision_fold, recall_fold, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]     # reverse order of results
        plt.plot(recall_fold, precision_fold)
                
        recall_interpolated    = np.linspace(0, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        plt.plot(recall_interpolated, precision_interpolated)
        
        plt.savefig("experiments/"+folder+"/optimize_hyperparameters/train="+ tiletrain + suffix(trainrate)+"test="+tiletest+optional_suffix+'.png') # Esto pisa todo cuando usas SVM!

        # 4) Print a message
        print("Testing on", tiletest, "resulted in", pafr9, pafr9i, pafr5, pafr5i, aps, pr_auc, pr_auci)
        
        
    with open("experiments/"+folder+"/optimize_hyperparameters/testscores_train="+ tiletrain + suffix(trainrate)+optional_suffix+".pkl", 'wb') as s_file:
        pickle.dump(scores_out,s_file)
        
    return scores_out
    
def get_params(method):
    if method=="rf":
        return rf_param_grid
    elif method=="svm": #Linear
        return svm_param_grid
    elif method=="svm-k":
        return asvm_rbf_param_grid


""" Display the results of HYPERPARAMETER OPTIMIZATION 

- Reads the cv-object from experiments/'+folder+'/optimize_hyperparameters/cvobject_train
- Reads the scores in test from experiments/"+folder+"/optimize_hyperparameters/testscores

"""
def display_cv_results(train_tile="b278",rate="full",optional_suffix="",folder="rf"):
        
    path_cv = 'experiments/'+folder+'/optimize_hyperparameters/cvobject_train='+ train_tile + suffix(rate)+ optional_suffix +'.pkl'
    path_sc = "experiments/"+folder+"/optimize_hyperparameters/testscores_train="+ train_tile + suffix(rate) +optional_suffix+".pkl"
    param_grid = get_params(folder)

    with open(path_cv,"rb") as gs_file:
        gs_rf = pickle.load(gs_file)
        
    results = pd.DataFrame(gs_rf.cv_results_)
    results_c = results
    
    selected_features = []
    selected_params = []
    selected_scores = []

    colnames = []
    for param in param_grid[0].keys():
        temp = [str(p) for p in gs_rf.cv_results_["param_"+param].data ]
        if len(np.unique(temp))>1:
            selected_params = selected_params + ["param_"+param]
            colnames = colnames + [param]
            
    for score in get_scorers().keys():
        selected_scores = selected_scores + ["mean_test_"+score]
        colnames = colnames + [score]
            
    selected_features = selected_params + selected_scores
    results = results[selected_features]
    results.columns = colnames

    print("\n")
    print("***** Hyperparameter optimisation results in tile ",train_tile,"*****")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(results)
    print("\n")
    for score in get_scorers().keys():
        max_value = max(results[score])
        max_index = np.argmax(results[score])
        print("Score",score.ljust(10),"Optimal value is ",str(max_value).ljust(20)," found in combination",results_c['params'][max_index])
        print 
    
    print("\n")
    
    print("***** Optimal classifier performance in test tiles *****")

    with open(path_sc, 'rb') as s_file:
        scores = pickle.load(s_file)
    
    names = ["tile","pafr9", "pafr9i", "pafr5", "pafr5i", "aps", "pr_auc", "pr_auci"]
    for name in names:
        print(name.ljust(7),end="")
    print("\n")        

    for tile in scores.keys():
        values = [tile] + list(scores[tile])[0:len(test_scorers_names)]
        for v in values:
            print(str(v)[:5].ljust(7),end="")     
        print("\n")
    print("Done")
        
""" ############################################# RANDOM FOREST CODE ################################################## """
#This title corresponds to section 1 of the thesis report

rf_param_grid = [
    {'n_estimators': [10],            #[10,50,100,150,200,250,300,350,400,450,500,750], 
     'criterion': ["entropy"],       # gini has been empirically demonstrated to be worse
     #'max_depth': [10,20,30,40,50,None],
     #'min_samples_split' : [2,5,10,15,20],
     'min_samples_leaf' : [2],       #[1,2,5,10,15],
     'max_features' : ["sqrt"]       #['auto','sqrt','log2'],
    }
]
"""
Run cross validation on the given tile, optimizing random forest parameters.
tile may be either a real tile "b234" or a vtile in string "2" format
"""
def optimize_random_forest_hyperparameters(tile, rate="full", n_folds=10,optional_suffix=""):

    X,y = retrieve_tile(tile,rate)
                                              
    # TODO: Use random search 
    gs_rf = GridSearchCV(
        RandomForestClassifier(), 
        param_grid=rf_param_grid, 
        scoring=get_scorers(), 
        cv=n_folds,
        n_jobs = n_jobs_global_param,
        verbose=2,
        refit="aps",  # Use aps as the metric to actually decide which classifier is the best
    )

    gs_rf.fit(X,y)
    
    with open('experiments/rf/optimize_hyperparameters/cvobject_train='+ tile + suffix(rate)+ optional_suffix +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
        
    return gs_rf

""" 
    This function reproduces experiment 2 carried out in the paper (only Random Forest)
"""
def cv_experiment_random_forest(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    
    gs_rf = optimize_random_forest_hyperparameters(train_tile,rate)
    
    scores = test_classifier(gs_rf.best_estimator_,test_tiles,train_tile,rate,folder="rf")
    display_cv_results(train_tile,rate,folder="rf")
    return scores
   
""" ############################################# SVM-LINEAR CODE ############################################ """
# Code corresponding to section 2 of the thesis
svm_param_grid = [
    { 'clf__C': np.logspace(-5, 15, 21),
      'clf__dual' : [False],
      #'clf__penalty' : ['l1','l2'], # Default l2
      #'clf__loss' : ['hinge','squared_hinge'], # squared hinge is the default
      'clf__class_weight' : [None] #[None, 'balanced']      
    }
]

def optimize_svm_hyperparameters(tile, rate="full", n_folds=10,optional_suffix=""):

    X,y = retrieve_tile(tile,rate)
                 
    
    cachedir = tempfile.mkdtemp()
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=10000)) ])
        
    # TODO: Use random search 
    gs_rf = GridSearchCV(
        pipe, 
        param_grid=svm_param_grid, 
        scoring=get_scorers(True), 
        cv=n_folds,
        n_jobs = n_jobs_global_param,
        verbose=3,
        refit="aps",  # Use aps as the metric to actually decide which classifier is the best
    )
    
    gs_rf.fit(X,y)
    
    shutil.rmtree(cachedir)
    
    with open('experiments/svm/optimize_hyperparameters/cvobject_train='+ tile + suffix(rate)+ optional_suffix +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
        
    return gs_rf

def cv_experiment_svm(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    gs_rf = optimize_svm_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,trainrate=rate,folder="svm")
    display_cv_results(train_tile,rate,folder="svm")
    return scores


""" ############################################# SVM-kernel CODE ############################################ """

asvm_rbf_param_grid = [
    { 'svm__C': np.logspace(-5, 12, 18),
      'feature_map__gamma' : np.logspace(-15, 4,20)
    }
]

# Optimal values: {'feature_map__gamma': 0.0001, 'svm__C': 10000.0}
def optimize_svmk_hyperparameters(tile="b278", rate="full", n_folds=10,optional_suffix=""):


    nystroem_approx_svm = Pipeline( [("scaler",StandardScaler()), ("feature_map", Nystroem()), ("svm", LinearSVC(dual=False,max_iter=100000))])


    X,y = retrieve_tile(tile,rate) 

    nystroem_approx_svm.set_params(feature_map__n_components=150)


    gs_rf = GridSearchCV(
        nystroem_approx_svm, 
        param_grid=asvm_rbf_param_grid, 
        scoring=get_scorers(True), 
        cv=10,
        n_jobs = 1,
        verbose=3,
        refit="aps",  # Use aps as the metric to actually decide which classifier is the best
    )

    gs_rf.fit(X,y)

    with open('experiments/svm-k/optimize_hyperparameters/cvobject_train='+ tile + suffix(rate)+ optional_suffix +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
            
    return gs_rf

def cv_experiment_svmk(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    gs_rf = optimize_svmk_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,trainrate=rate,folder="svm-k")
    display_cv_results(train_tile,rate,folder="svm-k")
    return scores


""" ############################################# Sections 1-2-3 comparison ############################################ """

# This function plots the performance of SVM-L, SVM-RBF and RF with the optimal parameters found in sections 1, 2 and 3
def compare_best_hyperparameters(train_tile="b278",test_tiles=["b234","b261","b360"]):
    rate = "full"

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    scores = test_classifier(clf,test_tiles,"rf",train_tile,rate)

    # SVM
    clf2 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=10000, C=0.1, dual=False)) ])
            
    clf2.fit(X,y)

    scores2 = test_classifier(clf2,test_tiles,"svm",train_tile,rate)
    
    #SVM-K

    nystroem_approx_svm = Pipeline( 
        [("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])

    nystroem_approx_svm.fit(X,y)    
        
    scores3 = test_classifier(nystroem_approx_svm,test_tiles,"svm-k",train_tile,rate)

    fig, ax = plt.subplots()

    plt.title('Training in tile' + train_tile + ' using optimal hyperparameters')
    plt.xlabel('recall')
    plt.ylabel('precision')


    for tt,col in zip(test_tiles,["blue","red","green"]):
                
        (p,r,t) = (scores[tt])[7]
        ax.plot(r,p,label=tt+' rf',color=col)

        (p,r,t) = (scores2[tt])[7]
        ax.plot(r,p,linestyle='--', dashes=(5, 5),label=tt+' svm-l',color=col)

        (p,r,t) = (scores3[tt])[7]
        ax.plot(r,p,linestyle='dotted',label=tt+' svm-k',color=col)

    leg = ax.legend();

def generate_figure_1():
    with open('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/cv_results.csv', newline='') as csvfile:
        dataset = pd.read_csv(csvfile, delimiter=' ')

    ntrees = dataset['n_estimators'][:12]

    auc_values_1 = dataset['auc_prc'][0:12]
    auc_values_2 = dataset['auc_prc'][12:24]
    auc_values_3 = dataset['auc_prc'][24:36]
    auc_values_4 = dataset['auc_prc'][36:48]

    fig, ax = plt.subplots()

    plt.title('Random Forest: Cross validation grid search results in tile b278')
    plt.xlabel('number of trees')
    plt.ylabel('AUC-PRC')
    ax.plot(ntrees,auc_values_1,label="criterion = gini ; max_features = log2", color="blue")
    ax.plot(ntrees,auc_values_2,label="criterion = gini ; max_features = sqrt",color="blue",linestyle='dashed')
    ax.plot(ntrees,auc_values_3,label="criterion = entropy ; max_features = log2",color="red")
    ax.plot(ntrees,auc_values_4,label="criterion = entropy ; max_features = sqrt",color="red",linestyle='dashed')
    leg = ax.legend();
    
def generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"]):

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    # SVM
    clf2 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=100000, C=0.1, dual=False)) ])
            
    clf2.fit(X,y)
    
    #SVM-K
    nystroem_approx_svm = Pipeline( 
        [("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])

    nystroem_approx_svm.fit(X,y)    
        
        
    for test in test_tiles:
        Xtest, ytest = retrieve_tile(test)
        curves = {}
        
        #RF
        test_predictions = clf.predict_proba(Xtest)[:,1]
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        curves["rf"] = (precision,recall)
        
        # SVM-L
        test_predictions = clf2.decision_function(Xtest)
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        curves["svml"] = (precision,recall)

        # SVM-K
        test_predictions = nystroem_approx_svm.decision_function(Xtest)
        precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        curves["svmk"] = (precision,recall)

        with open('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/test_results_train='+train_tile+ "Test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

def generate_figure_2_data():
    generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data(train_tile="b360",test_tiles=["b234","b261","b278"])


def generate_figure_4_subplots():
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/test_results_train='+train+ "Test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)

            fig, ax = plt.subplots()

            p,r = curves["rf"]
            ax.plot(r,p)

            #p,r = curves["svml"]
            #ax.plot(r,p, label="Linear SVM")
            
            #p,r = curves["svmk"]
            #ax.plot(r,p, label="RBF SVM")
            
            plt.title('Train ' + train + "- Test" + test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            #leg = ax.legend();
    
            plt.savefig('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/rf_test_results_train='+train+ "Test="+test+".png")
        

def generate_figure_2_subplots():
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/test_results_train='+train+ "Test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)

            fig, ax = plt.subplots()

            p,r = curves["rf"]
            ax.plot(r,p, label="Random Forest")

            p,r = curves["svml"]
            ax.plot(r,p, label="Linear SVM")
            
            p,r = curves["svmk"]
            ax.plot(r,p, label="RBF SVM")
            
            plt.title('Train ' + train + "- Test" + test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            leg = ax.legend();
    
            plt.savefig('/home/jere/carpyncho/experiments/rf/optimize_hyperparameters/test_results_train='+train+ "Test="+test+".png")

def generate_figure_3():
    
    with open('/home/jere/carpyncho/experiments/svm/optimize_hyperparameters/cv_result.txt', newline='') as csvfile:
        dataset = pd.read_csv(csvfile, delimiter=' ')

    C = dataset['C']
    pafr5i = dataset['pafr5i']
    pafr9i = dataset['pafr9i']
    aps = dataset['aps']

    fig, ax = plt.subplots()

    plt.title('SVM Lineal - Cross validation grid search in tile b278')
    plt.xlabel('log(C)')
    plt.ylabel('score')
    ax.plot(np.log(C),aps,label="Area under precision-recall curve",marker='.')
    ax.plot(np.log(C),pafr5i,label="Precision at a filxed recall of 0.5",marker='.')
    ax.plot(np.log(C),pafr9i,label="Precision at a fixed recall 0.9",marker='.')

    leg = ax.legend();
        
    
