"""
MODEL SELECTION EXPERIMENTS 
Chapters 2 and 3 of the master's thesis

Description: Optimize each classifier's parameters by doing k-fold-cross-validation
             on a given tile.
Usage:

    model_selection.py carpyncho_path output_path
"""
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
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from common import CarpynchoWrapper
import sys
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS
N_JOBS_GLOBAL_PARAM = 7            # NUMBER OF PROCESSORS TO BE USED
CARPYNCHO_LOCAL_FOLDER    = ""     # "/home/jere/carpyncho/"
EXPERIMENTS_OUTPUT_FOLDER_MS = ""  # "/home/jere/Desktop/ms/"

def init(carpyncho_local_folder_path, output_folder):
	"""
	Initialize this module
	
	Parameters
    ----------
    carpyncho_local_folder_path: Path in the local filesystem where VVV tiles downloaded from
	  Carpyncho are stored (see common.py)
	
    output_folder: Path where final and intermediate results of model selection experiments 
      will be saved
	"""

	global CARPYNCHO_LOCAL_FOLDER
	global EXPERIMENTS_OUTPUT_FOLDER_MS
	global CARPYNCHO
	
	CARPYNCHO_LOCAL_FOLDER = carpyncho_local_folder_path
	EXPERIMENTS_OUTPUT_FOLDER_MS = output_folder
	CARPYNCHO = CarpynchoWrapper(CARPYNCHO_LOCAL_FOLDER)

"""
Define custom performance metrics, to be used as cross validation scores
"""
N_SAMPLES_PRC = 1000 # Parameter for interpolated metrics
def auc_prc(y, y_pred):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    pr_auc  = auc(recall_fold, precision_fold)
    return pr_auc

def auc_interpolated_prc(y, y_pred):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    recall_interpolated    = np.linspace(0, 1, N_SAMPLES_PRC)
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
    recall_interpolated    = np.linspace(0, 1, N_SAMPLES_PRC)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
    pafr = max(precision_interpolated[recall_interpolated >= fixed_recall])
    return pafr    
    
MIN_RECALL_GLOBAL=0.3
def auc_interpolated_prc_robust(y, y_pred,min_recall=MIN_RECALL_GLOBAL):
    precision_fold, recall_fold, thresh = metrics.precision_recall_curve(y, y_pred)
    precision_fold, recall_fold, thresh = precision_fold[::-1], recall_fold[::-1], thresh[::-1]
    recall_interpolated    = np.linspace(min_recall, 1, N_SAMPLES_PRC)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    pr_auc2 = auc(recall_interpolated, precision_interpolated)
    return pr_auc2


def get_scorers(svm = True):

    """ 
    Get scorers to be used in cross validation to optimize hyperparameters.

    Parameters
    ----------
    svm: whether the scorer is going to be used for SVM or not.
    """
    if (svm):
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
        auc_prc_robust  = make_scorer(auc_interpolated_prc_robust,needs_proba=True)
        

    scorers = { #"auc_prc"    : auc_prc_scorer,  DISABLED SINCE I'VE EMPIRICALLY VERIFIED IT IS EQUIVALENT TO aps
                #"auc_prci"   : auc_iprc_scorer, DISABLED SINCE I'VE EMPIRICALLY VERIFIED IT IS EQUIVALENT TO aps
                #"pafr5"      : pafr5_scorer,
                "pafr5i"     : pafr5_i_scorer,
                #"pafr9"      : pafr9_scorer,
                "pafr9i"     : pafr9_i_scorer,
                "aps"        : 'average_precision',
                'auc_prc_r'   : auc_prc_robust
              }
              
    # Tip: If you want to print the full precision-recall curve, add a scorer for each recall point you are interested in
    # This computationally expensive! GridSearchCV does not accept scorers returning non-scalar values.
    return scorers

TEST_SCORER_NAMES = ["auc_prc","auc_prci", "pafr5","pafr5i","pafr9","pafr9i","aps","auc_prc_r"]

def test_classifier(classifier,tilestest,folder,tiletrain=""):
    """ 
    Given an already trained classifier, estimate its performance in a list of test 
    datasets.

    Parameters
    ----------  
    classifier: object implementing predic_proba or decision_func
    tilestest: a list of tile ids to be used, one at a time, as the test dataset of the classifier
    folder: where to save the results

    Returns
    ----------  
    returns a dictionary with the test scores for each tile in tilestest
            
    """
    scores_out = {}
    
    for tiletest in tilestest:
    
        plt.clf()

        # 0) Retrieve data
        Xtest, ytest = CARPYNCHO.retrieve_tile(tiletest)

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
                
        recall_interpolated    = np.linspace(0, 1, N_SAMPLES_PRC)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        plt.plot(recall_interpolated, precision_interpolated)
        
        plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_MS+"/"+folder+"/optimize_hyperparameters/train="+ tiletrain +"test="+tiletest+'.png') # Esto pisa todo cuando usas SVM!

        # 4) Print a message
        print("Testing on", tiletest, "resulted in", pafr9, pafr9i, pafr5, pafr5i, aps, pr_auc, pr_auci)
        
        
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+"/"+folder+"/optimize_hyperparameters/testscores_train="+ tiletrain +".pkl", 'wb') as s_file:
        pickle.dump(scores_out,s_file)
        
    return scores_out
    

def display_cv_results(train_tile="b278",method="rf"):
    """ 
    Display the results of HYPERPARAMETER OPTIMIZATION

    Parameters
    ----------  
    train_tile: the tile that was used for training in the results that are to be 
        displayed

    Returns
    ----------  
    returns a dictionary with the test scores for each tile in tilestest
            
    """
    rate = "full" # use full tiles

    path_cv = EXPERIMENTS_OUTPUT_FOLDER_MS+"/"+method+'/optimize_hyperparameters/cvobject_train='+ train_tile  +'.pkl'
    path_sc = EXPERIMENTS_OUTPUT_FOLDER_MS+"/"+method+"/optimize_hyperparameters/testscores_train="+ train_tile +".pkl"
    param_grid = get_params(method)

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
        values = [tile] + list(scores[tile])[0:len(TEST_SCORER_NAMES)]
        for v in values:
            print(str(v)[:5].ljust(7),end="")     
        print("\n")
    print("Done")
        
####################### RANDOM FOREST EXPERIMENTS ########################

rf_param_grid = [
    {'n_estimators': [10],           #[10,50,100,150,200,250,300,350,400,450,500,750], 
     'criterion': ["entropy"],       # gini has been empirically demonstrated to be worse
     #'max_depth': [10,20,30,40,50,None],
     #'min_samples_split' : [2,5,10,15,20],
     'min_samples_leaf' : [2],       #[1,2,5,10,15],
     'max_features' : ["sqrt"]       #['auto','sqrt','log2'],
    }
]

def optimize_random_forest_hyperparameters(tile, n_folds=10):
    """
    Run grid search cross validation on the given tile, optimizing random forest parameters.

    Parameters
    ----------  
    tile: Dataset to be used for the grid seach cross validation
    n_folds: number of folds to be used in cross validation

    Returns
    ----------  
    returns the resulting GridSearchCV object

    """
    X,y = CARPYNCHO.retrieve_tile(tile)
                                              
    # TODO: Use random search 
    gs_rf = GridSearchCV(
        RandomForestClassifier(), 
        param_grid=rf_param_grid, 
        scoring=get_scorers(), 
        cv=n_folds,
        n_jobs = N_JOBS_GLOBAL_PARAM,
        verbose=2,
        refit="aps",  # Use aps as the metric to actually decide which classifier is the best
    )

    gs_rf.fit(X,y)
    
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+"/rf/optimize_hyperparameters/cvobject_train="+ tile +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
        
    return gs_rf

def cv_experiment_random_forest(train_tile="b278", test_tiles=["b234","b261","b360"]):
    """ 
    Find optimal parameters for random forest doing grid search cross validation, and calculate
    performance in test using the optimal hyperparameters.

    Parameters
    ----------  
    train_tile: Tile to be used for the hyperparameter optimization using grid search
                cross validation. 
    test_tiles: List of tiles to be used as test datasets for the optimal classifier
                found using grid search cross validation

    Returns
    ----------  
    a dictionary containing performance scores in test

    """    
    gs_rf = optimize_random_forest_hyperparameters(train_tile)
    
    scores = test_classifier(gs_rf.best_estimator_,test_tiles,train_tile,folder="rf")
    display_cv_results(train_tile,rate,method="rf")
    return scores
   

####################### SVM-LINEAR EXPERIMENTS ########################

svm_param_grid = [
    { 'clf__C': np.logspace(-5, 15, 21),
      'clf__dual' : [False],
      #'clf__penalty' : ['l1','l2'], # Default l2
      #'clf__loss' : ['hinge','squared_hinge'], # squared hinge is the default
      'clf__class_weight' : [None] #[None, 'balanced']      
    }
]

def optimize_svm_hyperparameters(tile, n_folds=10):
    """
    Run grid search cross validation on a given tile, optimizing linear svm parameters.

    Parameters
    ----------  
    tile: Dataset to be used for the grid seach cross validation
    n_folds: number of folds to be used in cross validation

    Returns
    ----------  
    returns the resulting GridSearchCV object

    """
    X,y = CARPYNCHO.retrieve_tile(tile)
                 
    
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
        n_jobs = N_JOBS_GLOBAL_PARAM,
        verbose=3,
        refit="auc_prc_r",
    )
    
    gs_rf.fit(X,y)
    
    shutil.rmtree(cachedir)
    
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+'/svm/optimize_hyperparameters/cvobject_train-nopreproces='+ tile +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
        
    return gs_rf

def cv_experiment_svm(train_tile="b278", test_tiles=["b234","b261","b360"]):
    """ 
    Find optimal parameters for svml doing grid search cross validation, and calculate
    performance in test using the optimal hyperparameters.

    Parameters
    ----------  
    train_tile: Tile to be used for the hyperparameter optimization using grid search
                cross validation. 
    test_tiles: List of tiles to be used as test datasets for the optimal classifier
                found using grid search cross validation

    Returns
    ----------  
    a dictionary containing performance scores in test
    """

    gs_rf = optimize_svm_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,folder="svm")
    display_cv_results(train_tile,method="svm")
    return scores


####################### SVM-RBF EXPERIMENTS ########################

asvm_rbf_param_grid = [
    { 'svm__C': np.logspace(-5, 12, 18),
      'feature_map__gamma' : np.logspace(-15, 4,20)
    }
]

def optimize_svmk_hyperparameters(tile="b278",n_folds=10):
    """
    Run grid search cross validation on a given tile, optimizing svm RBF parameters.

    Parameters
    ----------  
    tile: Dataset to be used for the grid seach cross validation
    n_folds: number of folds to be used in cross validation

    Returns
    ----------  
    returns the resulting GridSearchCV object

    """

    nystroem_approx_svm = Pipeline( [("scaler",StandardScaler()), ("feature_map", Nystroem()), ("svm", LinearSVC(dual=False,max_iter=100000))])


    X,y = CARPYNCHO.retrieve_tile(tile,rate) 

    nystroem_approx_svm.set_params(feature_map__n_components=150)


    gs_rf = GridSearchCV(
        nystroem_approx_svm, 
        param_grid=asvm_rbf_param_grid, 
        scoring=get_scorers(True), 
        cv=10,
        n_jobs = 1,
        verbose=3,
        refit="auc_prc_r",  # Use aps as the metric to actually decide which classifier is the best
    )

    gs_rf.fit(X,y)

    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+'/svm-k/optimize_hyperparameters/cvobject_train-nopreproces='+ tile +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
            
    return gs_rf

def cv_experiment_svmk(train_tile="b278", test_tiles=["b234","b261","b360"]):
    """ 
    Find optimal parameters for svm RBF doing grid search cross validation, and calculate
    performance in test using the optimal hyperparameters.

    Parameters
    ----------  
    train_tile: Tile to be used for the hyperparameter optimization using grid search
                cross validation. 
    test_tiles: List of tiles to be used as test datasets for the optimal classifier
                found using grid search cross validation

    Returns
    ----------  
    a dictionary containing performance scores in test
    """
    gs_rf = optimize_svmk_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,folder="svm-k")
    display_cv_results(train_tile,method="svm-k")
    return scores

def get_params(method):
    """ 
    Get the grid of hyperparameters explored in cross validation grid search

    Parameters
    ----------  
    method: One of "rf", "svm" (for linear SVM), "svm-k" (for SVM RBF)

    Returns
    ----------  
    a dictionary with a list of values for each hyperparameter
    """
    if method=="rf":
        return rf_param_grid
    elif method=="svm": #Linear
        return svm_param_grid
    elif method=="svm-k":
        return asvm_rbf_param_grid

def get_optimal_parameters_i(kernel="linear"):
    """ 
    Get the optimal hyperparameters found using cross validation grid search un b278.

    Parameters
    ----------  
    kernel: either "linear" or "rbf"

    Returns
    ----------  
    a dictionary with the optimal values found for each hyperparameter
    """
    optimal = {}
    if (kernel=="linear" or kernel=="svml"):
        optimal["C"]=0.1
    elif (kernel=="rbf" or kernel=="svmk"):
        optimal["C"]=10000
        optimal["gamma"]=0.0001
    return optimal
    
####################### ANALYSIS AND PLOTS FOR THESIS DOCUMENT ########################

def generate_figure_1():
    """ 
    Plot cross validation results for RF
    """
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+'/rf/optimize_hyperparameters/cv_results.csv', newline='') as csvfile:
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
    leg = ax.legend()
    
    plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_MS+'/optimize_hyperparameters/test_results_train='+train+ "Test="+test+".png",bbox_inches='tight')

def generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"]):
    """ 
    Train a classifier and calculate the perfmance in test using the hyperparameters
    estimated in this section
    """
    # RF
    X,y=CARPYNCHO.retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    # SVM
    clf2 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=100000, C=get_optimal_parameters_i("svml")["C"], dual=False)) ])
            
    clf2.fit(X,y)
    
    #SVM-K
    nystroem_approx_svm = Pipeline( 
        [("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_i("svml")["gamma"])), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_i("svmK")["C"]))])

    nystroem_approx_svm.fit(X,y)    
        
    for test in test_tiles:
        Xtest, ytest = CARPYNCHO.retrieve_tile(test)
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

        with open(EXPERIMENTS_OUTPUT_FOLDER_MS+ '/optimize_hyperparameters/test_results_train='+train_tile+ "Test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

def generate_figure_2_data():
    """ 
    For each pair of tiles t1 and t2 in ["b234","b261","b278","b360"], train a classifier
    using t1 and test it using t2. Register the performance in test in the local filesystem.
    """
    generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data(train_tile="b360",test_tiles=["b234","b261","b278"])

def generate_figure_2_subplots():
    """ 
    For each pair of tiles t1 and t2 in ["b234","b261","b278","b360"], plot the precision-recall
    curves of classifiers trained with t1 and tested in t2.
    """
    scores = {}

    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open(EXPERIMENTS_OUTPUT_FOLDER_MS+"/rf/optimize_hyperparameters/test_results_train="+train+ "Test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)

            fig, ax = plt.subplots()

            p,r = curves["rf"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(MIN_RECALL_GLOBAL, 1, N_SAMPLES_PRC)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("rf",train,test)] = robust_auc
            ax.plot(r,p, label="Random Forest")

            p,r = curves["svml"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(MIN_RECALL_GLOBAL, 1, N_SAMPLES_PRC)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svml",train,test)] = robust_auc
            ax.plot(r,p, label="Linear SVM")
            
            p,r = curves["svmk"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(MIN_RECALL_GLOBAL, 1, N_SAMPLES_PRC)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svmk",train,test)] = robust_auc
            ax.plot(r,p, label="RBF SVM")
            
            plt.title('Train ' + train + "- Test" + test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            if (train=="b278" and test=="b234"): # Only plot a single legend
                leg = ax.legend();
    
            plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_MS+'/optimize_hyperparameters/test_results_train='+train+ "Test="+test+".png",bbox_inches='tight')
            
    with open(results_folder_initial_estimation+"baseline_aucs.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)   
        
    return scores
    
def generate_figure_3(train_tile="b278"):
    """ 
    Plot grid search cross validation scores for linear svm
    """

    # Read cross validation objects
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+'/svm/optimize_hyperparameters/cvobject_train-nopreproces='+train_tile+'.pkl', 'rb') as output:
        gs_rf= pickle.load(output)
    aucr = gs_rf.cv_results_['mean_test_auc_prc_r']
    pafr5i = gs_rf.cv_results_['mean_test_pafr5i']
    pafr9i = gs_rf.cv_results_['mean_test_pafr9i']
    C = svm_param_grid[0]['clf__C']

    fig, ax = plt.subplots()

    plt.title('SVM Lineal - Cross validation grid search in tile b278')
    plt.xlabel('log(C)')
    plt.ylabel('score')
    ax.set_ylim([0,1])
    ax.plot(np.log(C),aucr,label="R-AUCPRC",marker='.')
    ax.plot(np.log(C),pafr5i,label="Precision at recall of 0.5",marker='.')
    ax.plot(np.log(C),pafr9i,label="Precision at recall of 0.9",marker='.')

    leg = ax.legend(bbox_to_anchor=(0.5,0.5));

def generate_svm_heatmap(train_tile="b278"):
    """ 
    Plot grid search cross validation scores for svm RBF as a heatmap
    """
    # Read cross validation objects
    with open(EXPERIMENTS_OUTPUT_FOLDER_MS+'/svm-k/optimize_hyperparameters/cvobject_train-nopreproces='+train_tile+'.pkl', 'rb') as output:
        gs_rf= pickle.load(output)

    metric = 'mean_test_auc_prc_r' # 'mean_test_pafr5i'

    scores_svmk = gs_rf.cv_results_[metric].reshape(len(asvm_rbf_param_grid[0]['feature_map__gamma']),len(asvm_rbf_param_grid[0]['svm__C']))


    cmap = "magma"

    ###### PRINT THE SVM-K HEATMAP######
    df_m = scores_svmk

    fig, ax = plt.subplots(figsize=(11, 9))
    sb.heatmap(df_m,square=True, cmap=cmap, linewidth=.3, linecolor='w')

    xlabels = [ "{:.0e}".format(x) for x in asvm_rbf_param_grid[0]['svm__C'] ]
    ylabels = [ "{:.0e}".format(x) for x in asvm_rbf_param_grid[0]['feature_map__gamma'] ]

    plt.xticks(np.arange(len(asvm_rbf_param_grid[0]['svm__C']))+.5, labels=xlabels,rotation=60)
    plt.yticks(np.arange(len(asvm_rbf_param_grid[0]['feature_map__gamma']))+.5, labels=ylabels, rotation=45)

    # axis labels
    plt.xlabel('C')
    plt.ylabel('gamma')
    # title
    title = 'Robust AUC-PRC'.upper()#'Average Precision at a fixed recall of 0.5'.upper()
    plt.title(title, loc='left')

    if cmap==None:
        plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_MS+"/heatmapk"+"NONE.png")
    else:
        plt.savefig(EXPERIMENTS_OUTPUT_FOLDER_MS+"/heatmapk"+cmap+".png")
