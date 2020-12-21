""" HANDLING IMBALANCE OF CLASSES """
exec(open("/home/jere/Dropbox/University/Tesina/src/section8.py").read())
from imblearn.combine import SMOTEENN
from scipy import stats


def class_weight(test="b278",train="b234"):
    
    
    save_folder = "/home/jere/Desktop/section9/"

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)   

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    scores_pafr = {}

    ############ USING CW
    for i in [1.5,2,5,10,25,50,75,100]:#np.logspace(1, 5, 10): 

        clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0,class_weight={0:1,1:i}))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label= "CW= "+str(i)) 

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
    
        recall_interpolated    = np.linspace(0, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
        pafr = max(precision_interpolated[recall_interpolated >= 0.9])   

        scores[i]=robust_auc
        scores_pafr[i]=pafr
        
    clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=4,label="class_weight=None")

    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc = auc(recall_interpolated, precision_interpolated)
    recall_interpolated    = np.linspace(0, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)    
    pafr = max(precision_interpolated[recall_interpolated >= 0.9])  

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0,class_weight="balanced"))])

    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="class_weight=Auto")


    ########## PLOT ALL SCORES 

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Effect of class weight. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"cw_train="+train+"test="+test+"_curves.png")

    fig, ax = plt.subplots()
    ax.plot(list(scores.keys()), list(scores.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('CW')
    plt.ylabel('Robust AUC-PRC')
    plt.title('cw ; svm-rbg. ' + str(train) + ' testing in '+str(test))

    fig, ax = plt.subplots()
    ax.plot(list(scores_pafr.keys()), list(scores_pafr.values())) 

    horiz_line_data = np.array([robust_auc for i in range(1,len(X.columns))])
    ax.plot(range(1,len(X.columns)), horiz_line_data, 'r--',label="No feature selection") 

    plt.xlabel('CW')
    plt.ylabel('pafr5i')
    plt.title('cw ; svm-rbg. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"cw_train="+train+"test="+test+"_scores_pafri.png")

# Override svm-l param grid

svm_param_grid_hist = [
    { 'clf__C': np.logspace(-5, 10, 16),
      'clf__dual' : [False],
      'clf__class_weight' : [ {0:1,1:x} for x in np.logspace(1, 4, 10) ],
      'discretizer__n_bins' : [150]
    }
]

#cv_experiment_svm_hist()

def compare_gain_linear(test="b278",train="b234"):

    save_folder = "/home/jere/Desktop/section9/"

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)   

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    scores_pafr = {}
    
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=10.0))])

    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="older best params")
    
    
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=100.0,class_weight={0:1,1:10}))])
        
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="cw=1:10; C=10")
    
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=1.0,class_weight={0:1,1:10}))])
        
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="cw=1:10; C=100")   

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=1.0,class_weight={0:1,1:46.41}))])
        
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="cw=1:46.41; C=1")  

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Effect of class weight. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"EVAL_train="+train+"test="+test+".png")


#compare_gain_linear()
#compare_gain_linear(test="b234",train="b278")
#compare_gain_linear(test="b234",train="b360")
#compare_gain_linear(test="b360",train="b261")
#compare_gain_linear(test="b261",train="b234")
###########################################  OVERSAMPLING IN SVM-L #######################################################
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

def random_oversampling(test="b278",train="b234",method="naive"):
    
    save_folder = "/home/jere/Desktop/section9/"

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)   

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    scores_pafr = {}
    
    min_prop = (sum(y)/len(y))*1.01
    max_prop =  min_prop * 10
    
    for i in np.linspace(min_prop, max_prop, 10): 
        try:
            if (method=="naive"):
                ros = RandomOverSampler(sampling_strategy=i)
                X_resampled, y_resampled = ros.fit_resample(X, y)
            elif (method=="SMOTE"):
                X_resampled, y_resampled = SMOTE(sampling_strategy=i).fit_resample(X, y)
            elif (method=="ADASYN"):
                X_resampled, y_resampled = ADASYN(sampling_strategy=i).fit_resample(X, y)
            elif (method=="SMOTEENN"):
                X_resampled, y_resampled = SMOTEENN(sampling_strategy=i).fit_resample(X, y)
                
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
                ("scaler",StandardScaler()), 
                ("svm", LinearSVC(dual=False,max_iter=100000,C=10.0))])

            clf.fit(X_resampled, y_resampled)
            decs  = clf.decision_function(Xt)
            p,r,t = metrics.precision_recall_curve(yt,decs)
            
            p_rrl = sum(y_resampled)/len(y_resampled)  # how many RRLs per no-RRL
            ax.plot(r,p,linewidth=1,label="ratio 1 RLL :"+str(1/p_rrl)+"RRL")
        except:
            print("Failure at prop=",str(i))
            continue
        
    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=10.0))])

    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=3,label="no oversampling")
    
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Effect of '+method +'  oversampling. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+method+"_oversampling_train="+train+"test="+test+"_curves.png")
    
    
def random_oversampling_all_tiles(method="naive"):
    scores1= random_oversampling(train="b278",test="b234",method=method)
    scores2= random_oversampling(train="b278",test="b261",method=method)
    scores3= random_oversampling(train="b234",test="b261",method=method)
    scores4= random_oversampling(train="b234",test="b360",method=method)
    scores5= random_oversampling(train="b261",test="b360",method=method)
    scores6= random_oversampling(train="b261",test="b278",method=method)
    scores7= random_oversampling(train="b360",test="b278",method=method)
    scores8= random_oversampling(train="b360",test="b234",method=method)    
    
def random_oversampling_all_methods():
    #random_oversampling_all_tiles("naive")
    random_oversampling_all_tiles("SMOTE")
    random_oversampling_all_tiles("ADASYN")
    
    
### OVERSAMPLING IN SVM-KERNEL

def random_oversampling_k(test="b278",train="b234",method="naive"):
    
    save_folder = "/home/jere/Desktop/section9/"

    X,y = retrieve_tile(train,"full") 
    Xt,yt=retrieve_tile(test)   

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    scores_pafr = {}
    
    min_prop = (sum(y)/len(y))*1.01
    max_prop =  min_prop * 10
    
    for i in np.linspace(min_prop, max_prop, 10): 
        try:
            if (method=="naive"):
                ros = RandomOverSampler(sampling_strategy=i)
                X_resampled, y_resampled = ros.fit_resample(X, y)
            elif (method=="SMOTE"):
                X_resampled, y_resampled = SMOTE(sampling_strategy=i).fit_resample(X, y)
            elif (method=="ADASYN"):
                X_resampled, y_resampled = ADASYN(sampling_strategy=i).fit_resample(X, y)
            
            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                ("scaler",StandardScaler()), 
                ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
                ("svm", LinearSVC(dual=False,max_iter=100000,C=100000.0))])

            clf.fit(X_resampled, y_resampled)
            decs  = clf.decision_function(Xt)
            p,r,t = metrics.precision_recall_curve(yt,decs)
            
            p_rrl = sum(y_resampled)/len(y_resampled)  # how many RRLs per no-RRL
            ax.plot(r,p,linewidth=1,label="ratio 1 RLL :"+str(1/p_rrl)+"RRL")
        except:
            print("Failure at prop=",str(i))
            continue
        
    clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
        ("scaler",StandardScaler()), 
        ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
        ("svm", LinearSVC(dual=False,max_iter=100000,C=100000.0))])

    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=3,label="no oversampling")
    
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[SVM-RBF] Effect of '+method +'  oversampling. ' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+method+"_svmk_oversampling_train="+train+"test="+test+"_curves.png")

def random_oversampling_all_tiles_k(method="naive"):
    scores1= random_oversampling_k(train="b278",test="b234",method=method)
    scores2= random_oversampling_k(train="b278",test="b261",method=method)
    scores3= random_oversampling_k(train="b234",test="b261",method=method)
    scores4= random_oversampling_k(train="b234",test="b360",method=method)
    scores5= random_oversampling_k(train="b261",test="b360",method=method)
    scores6= random_oversampling_k(train="b261",test="b278",method=method)
    scores7= random_oversampling_k(train="b360",test="b278",method=method)
    scores8= random_oversampling_k(train="b360",test="b234",method=method)    


def random_oversampling_all_methods_k():
    random_oversampling_all_tiles_k("naive")
    random_oversampling_all_tiles_k("SMOTE")
    random_oversampling_all_tiles_k("ADASYN")


##### MOVE ME TO SECTION 10

def explore_rbf_potential(train="b234",test="b278"):
    
    save_folder = "/home/jere/Desktop/section9/"

    X,y=retrieve_tile(train)
    Xt,yt=retrieve_tile(test)   
    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    
    # Step 1: train RF, test it.
    #         - get a robust-auc
    #         - plot the line in the p-r curve, dashed
    
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)
    decs  = clf.predict_proba(Xt)[:,1]
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=3,label="Random Forest")
    
    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc_rf = auc(recall_interpolated, precision_interpolated)
    
    max_auc = 0
    
    for c in np.logspace(-5, 12, 18):
        for gamma in np.logspace(-15, 4,20):
            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                    ("scaler",StandardScaler()),
                    ("feature_map", Nystroem(gamma=gamma, n_components=300)),
                    ("svm", LinearSVC(dual=False,max_iter=100000,C=c))])

            clf.fit(X,y)
            decs  = clf.decision_function(Xt)
            
            s = stats.describe(decs)
            if ( abs(s.minmax[0] - s.minmax[1]) < 0.1 ):
                scores[(c,gamma)] = 0
                continue
            
            p,r,t = metrics.precision_recall_curve(yt,decs)

            precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            scores[(c,gamma)] = auc(recall_interpolated, precision_interpolated)
            
            if (scores[(c,gamma)] > 0.6 * robust_auc_rf and scores[(c,gamma)] > max_auc):
                max_auc = scores[(c,gamma)]
                ax.plot(r,p,linewidth=3,label="c="+str(c)+" g="+str(gamma))

    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[SVM-RBF] Perfomance train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_svmk_train="+train+"test="+test+"_curves.png")
    
    with open(save_folder+ "_svmk_train="+train+"test="+test+"_scores.pkl", 'wb') as s_file:
        pickle.dump(scores,s_file)
    return scores  

scores1= explore_rbf_potential(train="b278",test="b234")
scores2= explore_rbf_potential(train="b278",test="b261")
scores3= explore_rbf_potential(train="b234",test="b261")
scores4= explore_rbf_potential(train="b234",test="b360")
scores5= explore_rbf_potential(train="b261",test="b360")
scores6= explore_rbf_potential(train="b261",test="b278")
scores7= explore_rbf_potential(train="b360",test="b278")
scores8= explore_rbf_potential(train="b360",test="b234")     



