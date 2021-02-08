exec(open("/home/jere/Dropbox/University/Tesina/src/section1+2+3.py").read())

results_folder = "/home/jere/Desktop/preprocessing/"

################################################### ESTANDARIZACION ################################################################

def generate_scales_svm_data(train="b278",test="b234",kernel="linear"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    curves = {}
    
    if (kernel=="linear"):
        svc = LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])    

    print(".")
    
    clf = make_pipeline(StandardScaler(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    curves["StandardScaler"] = (p,r)
    ax.plot(r,p,label="StandardScaler")

    print(".")

    clf = make_pipeline(MinMaxScaler(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    curves["MinMaxScaler"] = (p,r)
    ax.plot(r,p,label="MinMaxScaler")

    print(".")

    clf = make_pipeline(MaxAbsScaler(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    curves["MaxAbsScaler"] = (p,r)
    ax.plot(r,p,label="MaxAbsScaler") 

    print(".")

    clf = make_pipeline(RobustScaler(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    curves["RobustScaler"] = (p,r)
    ax.plot(r,p,label="RobustScaler")

    print(".")

    clf = make_pipeline(Normalizer(),StandardScaler(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    curves["Normalizer"] = (p,r)
    ax.plot(r,p,label="Normalizer + StandardScaler")
                 
    if (test=="b234" and train=="b278"):
        leg = ax.legend();
        
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('train ' + str(train) + '- test '+str(test))

    with open(results_folder+kernel+'-Estandarizaciones-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.savefig(results_folder+kernel+'-Estandarizaciones-train='+train+ "test="+test+".png",bbox_inches='tight')


def generate_figure_5_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder+kernel+'-Estandarizaciones-train='+train+ "test="+test+".pkl", 'rb') as output:
        curves = pickle.load(output)      

    fig, ax = plt.subplots()
    
    (p,r) = curves["StandardScaler"]
    ax.plot(r,p,label="StandardScaler")

    (p,r) = curves["MinMaxScaler"]
    ax.plot(r,p,label="MinMaxScaler")
    
    (p,r) = curves["MaxAbsScaler"]
    ax.plot(r,p,label="MaxAbsScaler")

    (p,r) = curves["RobustScaler"]
    ax.plot(r,p,label="RobustScaler")

    (p,r) = curves["Normalizer"]
    ax.plot(r,p,label="Normalizer + StandardScaler")
    
    plt.xlabel('recall')
    plt.ylabel('precision')

    if (train=="b278" and test=="b234"):
        leg = ax.legend()
        
    plt.title('train ' + str(train) + '- test '+str(test))
    plt.savefig(results_folder+kernel+'-Estandarizaciones-train='+train+ "test="+test+".png",bbox_inches='tight')


def generate_figure_5_data(kernel):
    generate_scales_svm_data(kernel=kernel)
    generate_scales_svm_data("b234","b261",kernel)
    generate_scales_svm_data("b261","b360",kernel)
    generate_scales_svm_data("b360","b278",kernel)
    #generate_scales_svm_data("b278","b261",kernel)
    #generate_scales_svm_data("b234","b360",kernel)
    #generate_scales_svm_data("b261","b278",kernel)
    #generate_scales_svm_data("b360","b234",kernel)
        
def generate_figure_5_plots(kernel):
    generate_figure_5_subplot(kernel=kernel)
    generate_figure_5_subplot("b234","b261",kernel)
    generate_figure_5_subplot("b261","b360",kernel)
    generate_figure_5_subplot("b360","b278",kernel)
    #generate_figure_5_subplot("b278","b261",kernel)
    #generate_figure_5_subplot("b234","b360",kernel)
    #generate_figure_5_subplot("b261","b278",kernel)
    #generate_figure_5_subplot("b360","b234",kernel)
        
        
#################################################### BINNING #######################################################

def get_robust_auc_from_p_r(p,r):
        p, r  = p[::-1], r[::-1],
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, r, p)
        return auc(recall_interpolated, precision_interpolated)

bins_range = [10,50,100,150,200,300,500]
kmeans_bins_range = [5,10]

def kbins_discretizers(train="b278",test="b234",kernel="linear"):
    
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    curves = {}
    
    if (kernel=="linear"):
        svc = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
                 
    svc.fit(X, y)
    decs  = svc.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    curves["baseline"]=(p,r,get_robust_auc_from_p_r(p,r))
    
    # Bins quantile
    for bins in bins_range:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile'),svc) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer quantile nbins="+str(bins))
        curves[("quantile",bins)]=(p,r,get_robust_auc_from_p_r(p,r))

    # Bins uniformes
    for bins in bins_range:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform'),svc) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer unfirom nbins="+str(bins))
        curves[("uniform",bins)]=(p,r,get_robust_auc_from_p_r(p,r))

    # Bins kmeans
    for bins in kmeans_bins_range:
        clf = make_pipeline(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans'),svc) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="KBinsDiscretizer kmeans nbins="+str(bins))
        curves[("kmeans",bins)]=(p,r,get_robust_auc_from_p_r(p,r))

    leg = ax.legend();
    
    with open(results_folder+kernel+'-bins-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder+kernel+'-KBinsDiscretizer-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_6_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder+kernel+'-bins-train='+train+ "test="+test+".pkl", 'rb') as output:
        curves = pickle.load(output)      


    (p,r,auc) = curves["baseline"]

    fig, ax = plt.subplots()

    y = [auc for x in bins_range] 

    ax.plot(bins_range, y ,linewidth=1,label="baseline")

    ##### UNIFORM BINS
    aucs = []
    for nbins in bins_range:
        aucs = [curves[("uniform",nbins)][2]] + aucs 

    ax.plot(bins_range,aucs,label="Uniform",marker='.')

    ##### QUANTILE BINS
    aucs = []
    for nbins in bins_range:
        aucs = [curves[("quantile",nbins)][2]] + aucs 

    ax.plot(bins_range,aucs,label="Quantile",marker='.')


    ##### KMEANS BINS
    aucs = []
    for nbins in kmeans_bins_range:
        aucs = [curves[("kmeans",nbins)][2]] + aucs 

    ax.plot(kmeans_bins_range,aucs,label="KMeans",marker='.')

    plt.xlabel('Number of bins')
    plt.ylabel('Robust AUC-PRC')
    if (train=="b360" and test=="b278"):
        leg = ax.legend();
    #plt.show()
    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder+kernel+'-KBinsDiscretizerAUC-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_6_data(kernel):
    #kbins_discretizers(kernel=kernel)
    #kbins_discretizers("b234","b261",kernel)
    #kbins_discretizers("b261","b360",kernel)
    #kbins_discretizers("b360","b278",kernel)
    kbins_discretizers("b278","b261",kernel)
    kbins_discretizers("b234","b360",kernel)
    kbins_discretizers("b261","b278",kernel)
    kbins_discretizers("b360","b234",kernel)
    
def generate_figure_6_subplots(kernel):
    generate_figure_6_subplot(kernel=kernel)
    generate_figure_6_subplot("b234","b261",kernel)
    generate_figure_6_subplot("b261","b360",kernel)
    generate_figure_6_subplot("b360","b278",kernel)

######################################################  QUANTILE  TRANSFORMER ##########################################


n_quantiles_values = [5,10,25,50,100,250,500,1000]

def quantile_transformer(train="b278",test="b234",kernel="linear"):
    
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    curves = {}
    
    if (kernel=="linear"):
        svc = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
                 
    svc.fit(X, y)
    decs  = svc.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--', dashes=(5, 5))
    curves["baseline"]=(p,r,get_robust_auc_from_p_r(p,r))
    
    # uniforme
    for n_quantiles in n_quantiles_values:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform"),svc) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="uniform ; n_quantiles="+str(n_quantiles))
        curves[("uniform",n_quantiles)]=(p,r,get_robust_auc_from_p_r(p,r))

    # normal
    for n_quantiles in n_quantiles_values:
        clf = make_pipeline(QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform"),svc) 
        clf.fit(X, y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        ax.plot(r,p,label="normal ; n_quantiles="+str(n_quantiles))
        curves[("normal",n_quantiles)]=(p,r,get_robust_auc_from_p_r(p,r))

    leg = ax.legend();
    
    with open(results_folder+kernel+'-quantile-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder+kernel+'-quantile-train='+train+ "test="+test+".png",bbox_inches='tight')


def generate_figure_7_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder+kernel+'-quantile-train='+train+ "test="+test+".pkl", 'rb') as output:
        curves = pickle.load(output)      


    (p,r,auc) = curves["baseline"]

    fig, ax = plt.subplots()

    y = [auc for x in n_quantiles_values] 

    ax.plot(n_quantiles_values, y ,linewidth=1,label="baseline")

    ##### UNIFORM BINS
    aucs = []
    for q in n_quantiles_values:
        aucs = [curves[("uniform",q)][2]] + aucs 

    ax.plot(n_quantiles_values,aucs,label="Uniform",marker='.')

    ##### QUANTILE BINS
    aucs = []
    for q in n_quantiles_values:
        aucs = [curves[("normal",q)][2]] + aucs 

    ax.plot(n_quantiles_values,aucs,label="Normal",marker='.')


    plt.xlabel('Number of quantiles')
    plt.ylabel('Robust AUC-PRC')
    if (train=="b360" and test=="b278"):
        leg = ax.legend();
    
    #plt.show()
    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder+kernel+'-quantiles-AUC-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_7_data(kernel):
    quantile_transformer("b278","b234",kernel)
    quantile_transformer("b234","b261",kernel)
    quantile_transformer("b261","b360",kernel)
    quantile_transformer("b360","b278",kernel)
    quantile_transformer("b278","b261",kernel)
    quantile_transformer("b234","b360",kernel)
    quantile_transformer("b261","b278",kernel)
    quantile_transformer("b360","b234",kernel)

def generate_figure_7_subplots(kernel):
    generate_figure_7_subplot(kernel=kernel)
    generate_figure_7_subplot("b234","b261",kernel)
    generate_figure_7_subplot("b261","b360",kernel)
    generate_figure_7_subplot("b360","b278",kernel)


########################################################## OVERALL COMPARISON ###############################################################

def best_preprocessing_linear(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    clf = make_pipeline(StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--')

    clf = make_pipeline(PowerTransformer(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer")    

    clf = make_pipeline(KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile'),StandardScaler(),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Quantile Binning, 100 bins")

    clf = make_pipeline(QuantileTransformer(n_quantiles=1000, output_distribution="normal"),LinearSVC(C=0.01,verbose=3,dual=False, max_iter=100000)) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Normal QuantileTransformer, 1000 quantiles")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    if (train=="b261" and test=="b360"):
        leg = ax.legend();

    plt.title('train ' + str(train) + ' - test '+str(test))
    plt.savefig(results_folder+"linear"+'best-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_8_subplots():
    best_preprocessing_linear("b278","b234")
    best_preprocessing_linear("b234","b261")
    best_preprocessing_linear("b261","b360")
    best_preprocessing_linear("b360","b278")
    best_preprocessing_linear("b278","b261")
    best_preprocessing_linear("b234","b360")
    best_preprocessing_linear("b261","b278")
    best_preprocessing_linear("b360","b234")

def best_preprocessing_rbf(train="b278",test="b234"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test)
    fig, ax = plt.subplots()

    svc = Pipeline(
            [("scaler",StandardScaler()),
             ("feature_map", Nystroem(n_components=300,gamma=0.00017782794100389227)), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=10000))])
    svc.fit(X, y)
    decs  = svc.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--')

    clf = make_pipeline(KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile'),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Quantile Binning, 100 bins")


    clf = make_pipeline(PowerTransformer(),svc)
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer")    

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if (train=="b261" and test=="278"):
        leg = ax.legend();

    plt.title('train ' + str(train) + ' - test '+str(test))
    plt.savefig(results_folder+"rbf"+'best-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_9_subplots():
    best_preprocessing_rbf("b278","b234")
    best_preprocessing_rbf("b234","b261")
    best_preprocessing_rbf("b261","b360")
    best_preprocessing_rbf("b360","b278")
    best_preprocessing_rbf("b278","b261")
    best_preprocessing_rbf("b234","b360")
    best_preprocessing_rbf("b261","b278")
    best_preprocessing_rbf("b360","b234")
    
    
######### Reoptimize SVM-L hyperparameters after an improved preprocessing #########

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

    
########### Reoptimise svm rbf parameters  #######################
    
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
    
def cv_experiment_svmk_hist(train_tile="b278", test_tiles=["b234","b261","b360"],rate="full"):
    gs_rf = optimize_svmk_hist_hyperparameters(train_tile,rate)
    scores = test_classifier(gs_rf.best_estimator_,tilestest=test_tiles,tiletrain=train_tile,trainrate=rate,folder="svm-k")
    display_cv_results(train_tile,rate,folder="svm-k")
    return scores
    

########### Create heatmaps with the same scale showing parameter optimization results ########

def plot_heatmap(train_tile="b278"):

    # Read cross validation objects
    with open('experiments/svm/optimize_hyperparameters/cvobject_train='+train_tile+'.pkl', 'rb') as output:
        gs_rf= pickle.load(output)
    scores_l = gs_rf.cv_results_['mean_test_auc_prc_r'].reshape(len(svm_param_grid_hist[0]['clf__C']),len(svm_param_grid_hist[0]['discretizer__n_bins']))

    with open('experiments/svm-k/optimize_hyperparameters/cvobject_train=b278.pkl', 'rb') as output:
        gs_rf= pickle.load(output)
    scores_svmk = gs_rf.cv_results_['mean_test_auc_prc_r'].reshape(len(asvm_rbf_param_grid[0]['feature_map__gamma']),len(asvm_rbf_param_grid[0]['svm__C']))

    # Get min and max values
    min_s = min(scores_svmk.min() , scores_l.min())
    max_s = max(scores_svmk.max() , scores_l.max())


    center=None

    for cmap in ["viridis","magma","inferno","cividis",None]:
        ###### PRINT THE SVM-K HEATMAP######
        df_m = scores_svmk

        fig, ax = plt.subplots(figsize=(11, 9))
        sb.heatmap(df_m,square=True,vmin= min_s, vmax=max_s, cmap=cmap, center=center, linewidth=.3, linecolor='w')

        xlabels = [ "{:.0e}".format(x) for x in asvm_rbf_param_grid[0]['svm__C'] ]
        ylabels = [ "{:.0e}".format(x) for x in asvm_rbf_param_grid[0]['feature_map__gamma'] ]

        plt.xticks(np.arange(len(asvm_rbf_param_grid[0]['svm__C']))+.5, labels=xlabels,rotation=60)
        plt.yticks(np.arange(len(asvm_rbf_param_grid[0]['feature_map__gamma']))+.5, labels=ylabels, rotation=45)

        # axis labels
        plt.xlabel('C')
        plt.ylabel('gamma')
        # title
        title = 'Average Robust AUCPRC'.upper()
        plt.title(title, loc='left')

        if cmap==None:
            plt.savefig(results_folder+"heatmapk"+"NONE.png")
        else:
            plt.savefig(results_folder+"heatmapk"+cmap+".png")

        ###### PRINT THE SVM-L HEATMAP######
        
        df_m = scores_l   
        fig, ax = plt.subplots(figsize=(11, 9))
        sb.heatmap(df_m,square=True,vmin= min_s, vmax=max_s, cmap=cmap, center=center, linewidth=.3, linecolor='w')

        xlabels = [ "{:.0e}".format(x) for x in svm_param_grid_hist[0]['clf__C'] ]
        ylabels = svm_param_grid_hist[0]['discretizer__n_bins']

        plt.xticks(np.arange(len(svm_param_grid_hist[0]['discretizer__n_bins']))+.5, labels=ylabels,rotation=60)
        plt.yticks(np.arange(len(svm_param_grid_hist[0]['clf__C']))+.5, labels=xlabels, rotation=45)

        # axis labels
        plt.ylabel('C')
        plt.xlabel('nbins')
        # title
        title = 'Average Robust AUCPRC'.upper()
        plt.title(title, loc='left')

        if cmap==None:
            plt.savefig(results_folder+"heatmapl"+"NONE.png")
        else:
            plt.savefig(results_folder+"heatmapl"+cmap+".png")

################ End of section Performance comparison ###############

def generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"]):

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    # SVM
    clf2 = Pipeline([
        ('disc',KBinsDiscretizer(n_bins=150, encode='ordinal', strategy='quantile')),
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=100000, C=10, dual=False)) ])
            
    clf2.fit(X,y)
    
    #SVM-K
    nystroem_approx_svm = Pipeline( 
        [('disc',KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
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

        with open(results_folder+"best-train="+train_tile+ "test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      
    
    
def generate_figure_10_data():
    generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data(train_tile="b360",test_tiles=["b234","b261","b278"])
    

def generate_figure_10_subplots():
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open(results_folder+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
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
    
            plt.savefig(results_folder+"best-train="+train+ "test="+test+".png")

def run_all_figure_10():
    generate_figure_10_data()
    generate_figure_10_subplots()

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

