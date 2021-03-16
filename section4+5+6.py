exec(open("/home/jere/Dropbox/University/Tesina/src/section1+2+3.py").read())

results_folder_preproces = "/home/jere/Desktop/preprocessing/"

################################################### ESTANDARIZACION ################################################################

def generate_scales_svm_data(train="b278",test="b234",kernel="linear"):
    X,y = retrieve_tile(train,"full")
    Xt,yt=retrieve_tile(test) 
    fig, ax = plt.subplots()

    curves = {}
    
    if (kernel=="linear"):
        svc = LinearSVC(C=get_optimal_parameters_i(kernel)["C"],verbose=3,dual=False, max_iter=100000)
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_i(kernel)["gamma"])), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_i(kernel)["C"]))])    

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

    with open(results_folder_preproces+kernel+'-Estandarizaciones-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.savefig(results_folder_preproces+kernel+'-Estandarizaciones-train='+train+ "test="+test+".png",bbox_inches='tight')


def generate_figure_5_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder_preproces+kernel+'-Estandarizaciones-train='+train+ "test="+test+".pkl", 'rb') as output:
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

    #(p,r) = curves["Normalizer"]
    #ax.plot(r,p,label="Normalizer + StandardScaler")
    
    plt.xlabel('recall')
    plt.ylabel('precision')

    if (train=="b278" and test=="b234"):
        leg = ax.legend()
        
    plt.title('train ' + str(train) + '- test '+str(test))
    plt.savefig(results_folder_preproces+kernel+'-Estandarizaciones-train='+train+ "test="+test+".png",bbox_inches='tight')


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
        svc = make_pipeline(StandardScaler(),LinearSVC(C=get_optimal_parameters_i(kernel)["C"],verbose=3,dual=False, max_iter=100000))
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_i(kernel)["gamma"])), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_i(kernel)["c"]))])
                 
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
    
    with open(results_folder_preproces+kernel+'-bins-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder_preproces+kernel+'-KBinsDiscretizer-train='+train+ "test="+test+".png",bbox_inches='tight')

def generate_figure_6_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder_preproces+kernel+'-bins-train='+train+ "test="+test+".pkl", 'rb') as output:
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

    plt.savefig(results_folder_preproces+kernel+'-KBinsDiscretizerAUC-train='+train+ "test="+test+".png",bbox_inches='tight')

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
        svc = make_pipeline(StandardScaler(),LinearSVC(C=get_optimal_parameters_i(kernel)["C"],verbose=3,dual=False, max_iter=100000))
    elif (kernel=="rbf"):
        svc = Pipeline( 
                [("scaler",StandardScaler()),
                 ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_i(kernel)["gamma"])), 
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_i(kernel)["C"]))])
                 
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
    
    with open(results_folder_preproces+kernel+'-quantile-train='+train+ "test="+test+".pkl", 'wb') as output:
        pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      

    plt.title('train ' + str(train) + ' - test '+str(test))

    plt.savefig(results_folder_preproces+kernel+'-quantile-train='+train+ "test="+test+".png",bbox_inches='tight')


def generate_figure_7_subplot(train="b278",test="b234",kernel="linear"):
        
    with open(results_folder_preproces+kernel+'-quantile-train='+train+ "test="+test+".pkl", 'rb') as output:
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

    plt.savefig(results_folder_preproces+kernel+'-quantiles-AUC-train='+train+ "test="+test+".png",bbox_inches='tight')

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

    clf = make_pipeline(StandardScaler(),LinearSVC(C=get_optimal_parameters_i("svml")["C"],verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="StandardScaler",linestyle='--')

    clf = make_pipeline(PowerTransformer(),LinearSVC(C=get_optimal_parameters_i("svml")["C"],verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="PowerTransformer")    

    clf = make_pipeline(KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile'),StandardScaler(),LinearSVC(C=get_optimal_parameters_i("svml")["C"],verbose=3,dual=False, max_iter=100000))
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Quantile Binning, 100 bins")

    clf = make_pipeline(QuantileTransformer(n_quantiles=1000, output_distribution="normal"),LinearSVC(C=get_optimal_parameters_i("svml")["C"],verbose=3,dual=False, max_iter=100000)) 
    clf.fit(X, y)
    decs  = clf.decision_function(Xt)
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,label="Normal QuantileTransformer, 1000 quantiles")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    if (train=="b261" and test=="b360"):
        leg = ax.legend();

    plt.title('train ' + str(train) + ' - test '+str(test))
    plt.savefig(results_folder_preproces+"linear"+'best-train='+train+ "test="+test+".png",bbox_inches='tight')

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
             ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_i("svmk")["gamma"])), 
             ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_i("svmk")["C"]))])
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

    if (train=="b261" and test=="b278"):
        leg = ax.legend();

    plt.title('train ' + str(train) + ' - test '+str(test))
    plt.savefig(results_folder_preproces+"rbf"+'best-train='+train+ "test="+test+".png",bbox_inches='tight')

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

def plot_heatmaps_preproc(train_tile="b278"):

    # Read cross validation objects
    with open('experiments/svm/optimize_hyperparameters/cvobject_train='+train_tile+'.pkl', 'rb') as output:
        gs_rf= pickle.load(output)

    scores_l = gs_rf.cv_results_['mean_test_auc_prc_r'].reshape(len(svm_param_grid_hist[0]['clf__C']),len(svm_param_grid_hist[0]['discretizer__n_bins']))

    with open('experiments/svm-k/optimize_hyperparameters/cvobject_train='+train_tile+'.pkl', 'rb') as output:
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
            plt.savefig(results_folder_preproces+"heatmapk"+"NONE.png")
        else:
            plt.savefig(results_folder_preproces+"heatmapk"+cmap+".png")

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
            plt.savefig(results_folder_preproces+"heatmapl"+"NONE.png")
        else:
            plt.savefig(results_folder_preproces+"heatmapl"+cmap+".png")

################ End of section Performance comparison ###############

def get_optimal_parameters_p(kernel="linear"):
    optimal = {}
    if (kernel=="linear" or kernel=="svml"):
        optimal["C"]=10
        optimal["n_bins"]=150
    elif (kernel=="rbf" or kernel=="svmk"):
        optimal["C"]= 10000
        optimal["gamma"]=0.0001
        optimal["n_bins"]=100
    return optimal

def generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"]):

    # RF
    X,y=retrieve_tile(train_tile)
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)

    # SVM
    clf2 = Pipeline([
        ('disc',KBinsDiscretizer(n_bins=get_optimal_parameters_p("svml")["n_bins"], encode='ordinal', strategy='quantile')),
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(verbose=3, max_iter=100000, C=get_optimal_parameters_p("svml")["C"], dual=False)) ])
            
    clf2.fit(X,y)
    
    #SVM-K
    nystroem_approx_svm = Pipeline( 
        [('disc',KBinsDiscretizer(n_bins=get_optimal_parameters_p("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_map", Nystroem(n_components=300,gamma=get_optimal_parameters_p("svmk")["gamma"],)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_p("svmk")["C"],))])

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

        with open(results_folder_preproces+"best-train="+train_tile+ "test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)      
    
    
def generate_figure_10_data():
    generate_test_performance_data(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data(train_tile="b360",test_tiles=["b234","b261","b278"])
    

def generate_figure_10_subplots():
    
    scores = {}
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            with open(results_folder_preproces+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)

            fig, ax = plt.subplots()

            p,r = curves["rf"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("rf",train,test)] = robust_auc
            ax.plot(r,p, label="Random Forest")

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
    
            plt.savefig(results_folder_preproces+"best-train="+train+ "test="+test+".png")

    with open(results_folder_preproces+"baseline_aucs.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)   

def run_all_figure_10():
    generate_figure_10_data()
    generate_figure_10_subplots()

def get_baseline_preprocessing_stage(train,test,method):
    with open(results_folder_preproces+"baseline_aucs.pkl", 'rb') as output:
        scores = pickle.load(output)
        
    if (method=="linear" or method=="lineal"):
        method = "svml"
    elif (method=="rbf"):
        method = "svmk"
        
    return scores[(method,train,test)]


def generate_table_comparison(scores_after, scores_before):
    
	with open(scores_after, 'rb') as output:
		scores = pickle.load(output)

	with open(scores_before, 'rb') as output:
		scoreso = pickle.load(output)
		
	for train in ["b278","b234","b261","b360"]:
		for test in ["b278","b234","b261","b360"]:
			if (train==test):
				continue
			print(train,test,"{:.3f}".format(scoreso[("rf",train,test)]),
				"{:.3f}".format(scoreso[("svml",train,test)]),"{:.3f}".format(scores[("svml",train,test)]), "{:.3f}".format(scores[("svml",train,test)]-scoreso[("svml",train,test)]),
				"{:.3f}".format(scoreso[("svmk",train,test)]),"{:.3f}".format(scores[("svmk",train,test)]), "{:.3f}".format(scores[("svmk",train,test)]-scoreso[("svmk",train,test)]))

	return((scores,scoreso))
   
# generate_table_comparison(results_folder_preproces+"baseline_aucs.pkl",results_folder_initial_estimation+"baseline_aucs.pkl")
