exec(open("/home/jere/Dropbox/University/Tesina/src/section1+2+3.py").read())

#### Experiments related to preprocessing - SVM Linear (Section 4 report)####

def scales_svm(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = LinearSVC(C=0.01,verbose=3,dual=False, max_iter=10000)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="No scaling")

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler")

    clf = make_pipeline(MinMaxScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="MinMaxScaler")

    clf = make_pipeline(MaxAbsScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="MaxAbsScaler") 

    clf = make_pipeline(RobustScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="RobustScaler")

    clf = make_pipeline(Normalizer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Normalizer")
        
    clf = make_pipeline(PowerTransformer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer")
               
    leg = ax.legend();
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Effect of different scaling techniques, svm-linear. Training in tile ' + str(train) + ' testing in '+str(test))

    plt.show()
    
    
def kbins_discretizers_linear(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    
    # Bins quantile
    for bins in [10,50,100,500]:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile'),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer quantile nbins="+str(bins))

    # Bins uniformes
    for bins in [10,50,100,500]:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform'),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer unfirom nbins="+str(bins))

    # Bins kmeans
    for bins in [5,10]:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans'),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer kmeans nbins="+str(bins))
                   
    leg = ax.legend();
    
    
    plt.title('svm-l. KBinsDiscretizer. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()

def quantile_discretizers_linear(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    
    # uniforme
    for n_quantiles in [10,100,500,1000]:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer uniform n_quantiles="+str(n_quantiles))

    # Bins normal
    for n_quantiles in [10,100,500,1000]:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer normal n_quantiles="+str(n_quantiles))
                   
    leg = ax.legend();
    
    plt.title('svm-l. QuantileTransformer. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()



def quantile_discretizers_zoom_linear(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    
    # uniforme
    for n_quantiles in [5,10,25,50]:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer uniform n_quantiles="+str(n_quantiles))

    # Bins normal
    for n_quantiles in [5,10,25,50]:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer normal n_quantiles="+str(n_quantiles))
                   
    leg = ax.legend();
    
    plt.title('svm-l. QuantileTransformer. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()

def best_preprocessing_linear(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))

    clf = make_pipeline(PowerTransformer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer",linestyle='--', dashes=(5, 5))    

    bins = 100
    clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile'),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="KBinsDiscretizer solo nbins="+str(bins))

    clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile'),StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="KBinsDiscretizer+StandardScaler nbins="+str(bins))

    clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile'),PowerTransformer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="KBinsDiscretizer+PowerTransformer nbins="+str(bins))
      
    n_quantiles = 10

    clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="QuantileTransformer normal n_quantiles="+str(n_quantiles))

    clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="QuantileTransformer+StandardScaler nbins="+str(bins))

    clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),PowerTransformer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="QuantileTransformer+PowerTransformer nbins="+str(bins))


    leg = ax.legend();


    plt.title('svm-l. Combinations of best preprocessings. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()


######### Reoptimize SVM-L hyperparameters after an improved preprocessing (Section 6.1) #########

svm_param_grid_hist = [
    { 'clf__C': np.logspace(-5, 15, 21),
      'clf__dual' : [False],
      'clf__class_weight' : [None],
      'discretizer__n_bins' : [10,50,100,150]
    }
]

def get_params(method):
    if method=="rf":
        return rf_param_grid
    elif method=="svm": #Linear
        return svm_param_grid_hist
    elif method=="svm-k":
        return asvm_rbf_param_grid
        
def optimize_svml_hist_hyperparameters(tile, rate="full", n_folds=10,optional_suffix=""):

    X,y = retrieve_tile(tile,rate)
                 
    cachedir = tempfile.mkdtemp()
    
    pipe = Pipeline([
        ("discretizer",KBinsDiscretizer(encode='ordinal', strategy='quantile')),
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=10000)) ])
        
    # TODO: Use random search 
    gs_rf = GridSearchCV(
        pipe, 
        param_grid=svm_param_grid_hist, 
        scoring=get_scorers(True), 
        cv=n_folds,
        n_jobs = n_jobs_global_param,
        verbose=1,
        refit="auc_prc_r",  # Use aps as the metric to actually decide which classifier is the best
    )
    
    gs_rf.fit(X,y)
    
    shutil.rmtree(cachedir)
    
    with open('experiments/svm/optimize_hyperparameters/cvobject_train='+ tile + suffix(rate)+ optional_suffix +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
        
    return gs_rf


def cv_experiment_svm_hist(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    gs_rf = optimize_svml_hist_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,trainrate=rate,folder="svm")
    display_cv_results(train_tile,rate,folder="svm")
    return scores
    
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def print_svm_grid_svm_l_hist(train_tile="b278",rate="full"):
    with open('experiments/svm/optimize_hyperparameters/cvobject_train='+train_tile+'.pkl', 'rb') as output:
        gs_rf= pickle.load(output)

    scores = gs_rf.cv_results_['mean_test_aps'].reshape(len(svm_param_grid_hist[0]['clf__C']),len(svm_param_grid_hist[0]['discretizer__n_bins']))
                    
    plt.figure(figsize=(6, 8))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.6))
    plt.ylabel('nbins')
    plt.xlabel('gamma')
    plt.colorbar()
    plt.yticks(np.arange(len(svm_param_grid_hist[0]['clf__C'])), svm_param_grid_hist[0]['clf__C'], rotation=45)
    plt.xticks(np.arange(len(svm_param_grid_hist[0]['discretizer__n_bins'])), svm_param_grid_hist[0]['discretizer__n_bins'])
    plt.title('Average validation AUCPRC (SVM-L)')
    plt.show()

                    
    scores = gs_rf.cv_results_['mean_test_pafr5i'].reshape(len(svm_param_grid_hist[0]['clf__C']),len(svm_param_grid_hist[0]['discretizer__n_bins']))
                    
    plt.figure(figsize=(6, 8))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.6))
    plt.ylabel('nbins')
    plt.xlabel('gamma')
    plt.colorbar()
    plt.yticks(np.arange(len(svm_param_grid_hist[0]['clf__C'])), svm_param_grid_hist[0]['clf__C'], rotation=45)
    plt.xticks(np.arange(len(svm_param_grid_hist[0]['discretizer__n_bins'])), svm_param_grid_hist[0]['discretizer__n_bins'])
    plt.title('Average validation precision at a fixed recall of 0.5 (SVM-L)')
    plt.show()

    scores = gs_rf.cv_results_['mean_test_pafr9i'].reshape(len(svm_param_grid_hist[0]['clf__C']),len(svm_param_grid_hist[0]['discretizer__n_bins']))
                    
    plt.figure(figsize=(6, 8))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.6))
    plt.ylabel('nbins')
    plt.xlabel('gamma')
    plt.colorbar()
    plt.yticks(np.arange(len(svm_param_grid_hist[0]['clf__C'])), svm_param_grid_hist[0]['clf__C'], rotation=45)
    plt.xticks(np.arange(len(svm_param_grid_hist[0]['discretizer__n_bins'])), svm_param_grid_hist[0]['discretizer__n_bins'])
    plt.title('Average validation precision at a fixed recall of 0.9 (SVM-L)')
    plt.show()
    
################# Experiments related to preprocessing - SVM-rbf (SEction 5 report)##############

def scales_svmk(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    print("go")

    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler")
    print("BASELINE")


    clf =  Pipeline( 
                [("scaler",MinMaxScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="MinMaxScaler")
    print("BASELINE")


    clf =  Pipeline( 
                [("scaler",MaxAbsScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])    
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="MaxAbsScaler") 

    print("BASELINE")

    #clf =  Pipeline( 
    #            [("scaler",RobustScaler()), 
    #             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
    #             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))]) 
    #clf.fit(X, y)
    #decs  = clf.decision_function(Xt)
    #p,r,t = metrics.precision_recall_curve(yt,decs)
    #ax.plot(r,p,label="RobustScaler")

    print("BASELINE")


    clf =  Pipeline( 
                [("scaler",Normalizer()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))]) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Normalizer")
        
    print("BASELINE")

    clf =  Pipeline( 
        [("scaler",PowerTransformer()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))]) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer")
               
    leg = ax.legend();
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Effect of different scaling techniques, svm-rbf. Training in tile ' + str(train) + ' testing in '+str(test))

    plt.show()


def kbins_discretizers_rbf(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()


    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    print("BASELINE")
    # Bins quantile
    for bins in [10,50,100,500]:    
        clf =  Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')), 
                     ("scaler",StandardScaler()),
                     ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer quantile nbins="+str(bins))

    print("Q")
    # Bins uniformes
    for bins in [10,50,100,500]:
        clf =  Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')), 
                     ("scaler",StandardScaler()),
                     ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])  
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer unfirom nbins="+str(bins))
    print("U")

    # Bins kmeans
    for bins in [5,10,15]:
        print("K")

        clf =  Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')), 
                     ("scaler",StandardScaler()),
                     ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])    
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer kmeans nbins="+str(bins))

    leg = ax.legend();
    
    plt.title('svm-k. KBinsDiscretizer. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()
    
     
def quantile_discretizers_rbf(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()


    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    
    # uniforme
    for n_quantiles in [10,100,500,1000]:
        print("K")
        clf =  Pipeline( 
            [("transf",QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform")), ("scaler",StandardScaler()),
             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer uniform n_quantiles="+str(n_quantiles))

    # Bins normal
    for n_quantiles in [10,100,500,1000]:
        print("K")
        clf =  Pipeline( 
            [("tra",QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")), ("scaler",StandardScaler()),
             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer normal n_quantiles="+str(n_quantiles))
                   
    leg = ax.legend();
    
    plt.title('svm-k. QuantileTransformer. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()
    
def kbins_discretizers_rbf_zoom(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()


    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    print("BASELINE")
    # Bins quantile
    for bins in [50,75,100,150,200]:    
        clf =  Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')), 
                     ("scaler",StandardScaler()),
                     ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer+StandardScaler quantile nbins="+str(bins))

        clf =  Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')), 
                     ("scaler",PowerTransformer()),
                     ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer+PowerTransformer quantile nbins="+str(bins))
        

    leg = ax.legend();
    
    plt.title('svm-k. KBinsDiscretizer zoom. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()
    

def quantile_discretizers_rbf_zoom(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()


    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    
    # Bins normal
    for n_quantiles in [5,10,20,30]:
        print("K")
        clf =  Pipeline( 
            [("tra",QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")), ("scaler",StandardScaler()),
             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer+StandardScaler normal n_quantiles="+str(n_quantiles))

        clf =  Pipeline( 
            [("tra",QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")), ("scaler",PowerTransformer()),
             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="QuantileTransformer+PowerTransformer normal n_quantiles="+str(n_quantiles))
                           
    leg = ax.legend();
    
    plt.title('svm-k. QuantileTransformer zoom. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()
    
    
def best_preprocessing_rbf(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf =  Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))

    bins = 100
    clf =  Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')), 
                 ("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="KBinsDiscretizer+StandardScaler quantile nbins="+str(bins))


    bins = 150
    clf =  Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')), 
                 ("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="KBinsDiscretizer+StandardScaler quantile nbins="+str(bins))
        
        
    n_quantiles = 5
    clf =  Pipeline( 
        [("tra",QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")), ("scaler",StandardScaler()),
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="QuantileTransformer+StandardScaler normal n_quantiles="+str(n_quantiles))

    leg = ax.legend();


    plt.title('svm-rbf. Combinations of best preprocessings. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.show()
    
    
########### Reoptimise svm rbf parameters (sect 6.2) ########################
    
def optimize_svmk_hist_hyperparameters(tile="b278", rate="full", n_folds=10,optional_suffix=""):

    nystroem_approx_svm = Pipeline( [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')), ("scaler",StandardScaler()), ("feature_map", Nystroem()), ("svm", LinearSVC(dual=False,max_iter=100000))])


    X,y = retrieve_tile(tile,rate) 

    nystroem_approx_svm.set_params(feature_map__n_components=150)


    gs_rf = GridSearchCV(
        nystroem_approx_svm, 
        param_grid=asvm_rbf_param_grid, 
        scoring=get_scorers(True), 
        cv=10,
        n_jobs = 1,
        verbose=3,
        refit='auc_prc_r',  # Use aps as the metric to actually decide which classifier is the best
    )

    gs_rf.fit(X,y)

    with open('experiments/svm-k/optimize_hyperparameters/cvobject_train='+ tile + suffix(rate)+ optional_suffix +'.pkl', 'wb') as output:
        pickle.dump(gs_rf, output, pickle.HIGHEST_PROTOCOL)
            
    return gs_rf

def svm_param_grid_show():
    with open('experiments/svm-k/optimize_hyperparameters/cvobject_train=b278.pkl', 'rb') as output:
        gs_rf= pickle.load(output)

    scores = gs_rf.cv_results_['mean_test_auc_prc_r'].reshape(len(asvm_rbf_param_grid[0]['feature_map__gamma']),len(asvm_rbf_param_grid[0]['svm__C']))
                    
    plt.figure(figsize=(6, 8))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.37))
    plt.ylabel('gamma')
    plt.xlabel('C')
    plt.colorbar()
    plt.yticks(np.arange(len(asvm_rbf_param_grid[0]['feature_map__gamma'])), asvm_rbf_param_grid[0]['feature_map__gamma'], rotation=45)
    plt.xticks(np.arange(len(asvm_rbf_param_grid[0]['svm__C'])), asvm_rbf_param_grid[0]['svm__C'], rotation=60)
    plt.title('Average validation  Robust AUCPRC')
    plt.show()


    scores = gs_rf.cv_results_['mean_test_pafr5i'].reshape(len(asvm_rbf_param_grid[0]['feature_map__gamma']),len(asvm_rbf_param_grid[0]['svm__C']))
                    
    plt.figure(figsize=(6, 8))
    #plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.9))
    plt.ylabel('gamma')
    plt.xlabel('C')
    plt.colorbar()
    plt.yticks(np.arange(len(asvm_rbf_param_grid[0]['feature_map__gamma'])), asvm_rbf_param_grid[0]['feature_map__gamma'], rotation=45)
    plt.xticks(np.arange(len(asvm_rbf_param_grid[0]['svm__C'])), asvm_rbf_param_grid[0]['svm__C'], rotation=60)
    plt.title('Average validation precision at a fixed recall of 0.5')
    plt.show()

def cv_experiment_svmk_hist(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    gs_rf = optimize_svmk_hist_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,trainrate=rate,folder="svm-k")
    display_cv_results(train_tile,rate,folder="svm-k")
    return scores


############ Section 6.3 - Performance omparison ###############

# This function plots the performance of SVM-L, SVM-RBF and RF with the optimal parameters found in sections 4-5-6:

def compare_best_hyperparameters_s6(train_tile="b278",test_tiles=["b234","b261","b360"]):
    rate = "full"

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    scores = test_classifier(clf,test_tiles,"rf",train_tile,rate)

    # SVM-linear
    clf2 = Pipeline([
        ("discretizer",KBinsDiscretizer(encode='ordinal', strategy='quantile',n_bins=150)),
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=10000,C=1, dual=False)) ])
        
    clf2.fit(X,y)

    scores2 = test_classifier(clf2,test_tiles,"svm",train_tile,rate)
    
    #SVM-K

    nystroem_approx_svm = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')), 
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=100000.0))])

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



######################################################################
########################## Bonus experiments ######################### 

# Effect of C in svm-linear
def comparar_svml_c():
    Xt,yt=retrieve_tile("b234")   
    X,y = retrieve_tile("b278","full")

    fig, ax = plt.subplots()

    # EXPLRA EL RANGO DE C  
    for c in np.logspace(-5, 15, 21):
        clf = make_pipeline( StandardScaler(), LinearSVC(C=c,verbose=3,dual=False, max_iter=100000))
        clf.fit(X, y)
        preds = clf.predict(Xt)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="C="+str(c))
    leg = ax.legend();
    plt.title('Effect of C, svm-linear. Training in tile b278/ testing in b234')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

# Effect of undersample in svm-l
def comparar_undersample(train="1",test="2"):
    Xt,yt=retrieve_tile(test)
    fig, ax = plt.subplots()

    for rate in ["1:1","1:10","1:100","1:500","1:1000","full"]:
        X,y = retrieve_tile(train,rate) 
        clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
        clf.fit(X, y)
        preds = clf.predict(Xt)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        plt.plot(r,p,label=rate)

    leg = ax.legend();
    plt.title('Effect of undersampling in training dataset, svm-linear. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

# Effect of undersample in svm-kl
def comparar_undersample_k(train="1",test="2"):
    Xt,yt=retrieve_tile(test)
    fig, ax = plt.subplots()

    for rate in ["1:1","1:10","1:100","1:500","1:1000","full"]:
        X,y = retrieve_tile(train,rate) 
        clf = Pipeline( 
                [("scaler",StandardScaler()), 
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
        clf.fit(X, y)
        preds = clf.predict(Xt)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        plt.plot(r,p,label=rate)

    leg = ax.legend();
    plt.title('Effect of undersampling in training dataset, svm-linear. Training in tile ' + str(train) + ' testing in '+str(test))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()


#######################################################################################
################# DEBUGGING RBF: WHY DOES IT WORK SO POORLY?? #########################

#### Analisis 0: El C elegido generaliza muy mal

# Number of kernel approximate components
def effect_of_C_svm_rbf(train="b278",test="b234"):

    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    for val_c in np.logspace(-5, 15, 21):
    
        clf = Pipeline( 
            [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
             ("scaler",StandardScaler()), 
             ("feature_map", Nystroem(gamma=0.0001,n_components=300)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=val_c))])

        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="C="+str(val_c))  

    leg = ax.legend(ncol=3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('100Bins+StandardScaler+NystroemKernelAprox+SVM-Linear. Effect of C' + str(train) + ' testing in '+str(test))
    plt.show()  
    
#### Analysis 1: Is the approximator to blame? #####

# Number of kernel approximate components
def number_of_approximate_components_svmk(train="b278",test="b234"):

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])


    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 

    fig, ax = plt.subplots()
        
    grid = 30 * np.arange(1, 13)
            
    for d in grid[:len(grid)//2]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label=str(d)+" features")  
        
    for d in grid[len(grid)//2+1:]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,linestyle="dashed",label=str(d)+" features")  
        
    leg = ax.legend(ncol=3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('100Bins+StandardScaler+NystroemKernelAprox+SVM-Linear. Effect of # components in Nystroem ' + str(train) + ' testing in '+str(test))

    plt.show()  

def number_of_approximate_components_svmk_old_hyp(train="b278",test="b234"):

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])


    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 

    fig, ax = plt.subplots()
        
    grid = 30 * np.arange(1, 13)
            
    for d in grid[:len(grid)//2]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label=str(d)+" features")  
        
    for d in grid[len(grid)//2+1:]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,linestyle="dashed",label=str(d)+" features")  
        
    leg = ax.legend(ncol=3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('100Bins+StandardScaler+NystroemKernelAprox+SVM-Linear. Effect of # components in Nystroem ' + str(train) + ' testing in '+str(test))

    plt.show()  
    
def number_of_approximate_components_svmk_no_bins(train="b278",test="b234"):

    clf = Pipeline( 
        [("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])


    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 

    fig, ax = plt.subplots()
        
    grid = 30 * np.arange(1, 13)
            
    for d in grid[:len(grid)//2]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label=str(d)+" features")  
        
    for d in grid[len(grid)//2+1:]:
        clf.set_params(feature_map__n_components=d)
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,linestyle="dashed",label=str(d)+" features")  
        
    leg = ax.legend(ncol=3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('100Bins+StandardScaler+NystroemKernelAprox+SVM-Linear. Effect of # components in Nystroem ' + str(train) + ' testing in '+str(test))

    plt.show()      
    
# Looks like... the more componentes, the more performance, as expected. With a notable exception when training with b234
# Let's compare with the real RBF
    
# Svm-kernel real vs svm-kernel aprox with optimal parameters
def svmkreal_vs_aproximado_bins(train="b278",test="b234"):
    
    fig, ax = plt.subplots()

    rate = "1:100"
    X,y = retrieve_tile(train,rate)
    Xt,yt=retrieve_tile(test)   
    
    clf = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("svm", SVC(kernel="rbf",max_iter=100000,C=1000000000000,gamma=0.0001))])
    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm rbf exacto rate 1:100") 

    print("Finished real SVM")
    
    X,y = retrieve_tile(train,"full")

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=150)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 150 componentes") 

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 300 componentes") 

    X,y = retrieve_tile(train,rate)
    Xt,yt=retrieve_tile(test)   

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=150)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 150 componentes 1:100") 

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=1000000000000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 300 componentes 1:100") 
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Nystroem rbf approximator vs SVC-rbf exacto' + str(train) + ' testing in '+str(test))
    
    plt.show()  

# Svm-kernel real vs svm-kernel aprox with sub-optimal parameters and higher rate
def svmkreal_vs_aproximado_bins_subopt(train="b278",test="b234"):
    
    Xt,yt=retrieve_tile(test)   
    fig, ax = plt.subplots()
    
    rate = "1:500"
    X,y = retrieve_tile(train,rate)
    clf = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("svm", SVC(kernel="rbf",max_iter=100000,C=10000,gamma=0.0001))])
    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm rbf exacto rate 1:500") 
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Nystroem rbf approximator vs SVC-rbf exacto' + str(train) + ' testing in '+str(test))
    
    print("Finished real SVM")
    
    X,y = retrieve_tile(train,"full")



    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=150)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 150 componentes") 

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 300 componentes") 

    X,y = retrieve_tile(train,rate)
    Xt,yt=retrieve_tile(test)   
    fig, ax = plt.subplots()


    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=150)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 150 componentes 1:500") 

    clf = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(gamma=0.0001, n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000.0))])

    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 300 componentes 1:500") 
    

    plt.show()  
    
    


def svmkreal_vs_aproximado_nobins(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test)   
    fig, ax = plt.subplots()

    clf = Pipeline(
        [("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
         
    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm aproximado Nystroem 300 componentes") 

    X,y = retrieve_tile(train,"1:100")
    clf = Pipeline(
        [("scaler",StandardScaler()), 
         ("svm", SVC(kernel="rbf",max_iter=100000,C=10000,gamma=0.00017782794100389227))])
    clf.fit(X,y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="100bins+Scaler+ svm rbf exacto rate 1:100") 
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Nystroem rbf approximator vs SVC-rbf exacto' + str(train) + ' testing in '+str(test))

    plt.show()  
    
    
    


#######################################################################################
############################### RULING OUT ERRORS ##################################### 

#  decisionfunc vs predict_proba
def probabilities_svm():
    X,y = retrieve_tile("b278","full") 
    Xt,yt=retrieve_tile(test)
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="decision_func")

    clf2 = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf = CalibratedClassifierCV(clf2) 
    clf.fit(X, y)
    y_proba = clf.predict_proba(Xt)
    p,r,t = metrics.precision_recall_curve(yt, y_proba[:,1])
    ax.plot(r,p,label="predict_proba")  

    leg = ax.legend();
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

#tolerance parameter in svm
def tol_experiment():
    
    X,y = retrieve_tile("b278","full") 
    Xt,yt=retrieve_tile("b234")  
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.1,verbose=3,dual=False, max_iter=100000,tol=1e-12))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-12")


    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.1,verbose=3,dual=False, max_iter=100000,tol=1e-4))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-4")

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.1,verbose=3,dual=True, max_iter=100000,tol=1e-12))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-12 noD")


    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.1,verbose=3,dual=True, max_iter=100000,tol=1e-4))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-4 NoD")

    clf = make_pipeline(StandardScaler(),LinearSVC(C=10,verbose=3,dual=False, max_iter=100000,tol=1e-12))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-12 10")


    clf = make_pipeline(StandardScaler(),LinearSVC(C=10,verbose=3,dual=False, max_iter=100000,tol=1e-4))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-4 10")

    clf = make_pipeline(StandardScaler(),LinearSVC(C=10,verbose=3,dual=True, max_iter=100000,tol=1e-12))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-12 noD 10")


    clf = make_pipeline(StandardScaler(),LinearSVC(C=10,verbose=3,dual=True, max_iter=100000,tol=1e-4))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="tol=1e-4 NoD 10")

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    
    
#Figure 14
def compare_best_hyperparameters_bins(train_tile="b278",test_tile="b234"):
    rate = "full"

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    scores = test_classifier(clf,[test_tile],"rf",train_tile,rate)

    # SVM
    clf2 = Pipeline([
        ("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=10000, C=0.1, dual=False)) ])
        
    clf2.fit(X,y)

    scores2 = test_classifier(clf2,[test_tile],"svm",train_tile,rate)
    
    #SVM-K
    nystroem_approx_svm = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])

    nystroem_approx_svm.fit(X,y)    
        
    scores3 = test_classifier(nystroem_approx_svm,[test_tile],"svm-k",train_tile,rate)

    fig, ax = plt.subplots()

    plt.title('Training in tile' + train_tile + 'testing in tile' + test_tile +' using optimal hyperparameters')
    plt.xlabel('recall')
    plt.ylabel('precision')
      
    (p,r,t) = (scores[test_tile])[7]
    ax.plot(r,p,label=test_tile+' rf')

    (p,r,t) = (scores2[test_tile])[7]
    ax.plot(r,p,label=test_tile+' 100kbins + scaling + svm-l')

    (p,r,t) = (scores3[test_tile])[7]
    ax.plot(r,p,label=test_tile+' 100kbins + scaling + svm-k')

    leg = ax.legend();




def compare_svm_kernel_best_hyp(train_tile="b278",test_tile="b234"):
    X,y = retrieve_tile(train_tile,"full") 
    Xt,yt=retrieve_tile(test_tile)   
    fig, ax = plt.subplots()

    # FIRST
    clf = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.0001)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=100000000000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Optimized using hbins + pafr5i",linestyle='--', dashes=(5, 5))
    
    # SECOND 
    clf = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=1)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=100000000000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Optimized using hbins + pafr9i",linestyle='--', dashes=(5, 5))
        
    # THIRD
    clf = Pipeline(
        [("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Old hyperparameters",linestyle='--', dashes=(5, 5))
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
  
    plt.show()
    

##################################3

