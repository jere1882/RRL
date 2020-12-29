exec(open("/home/jere/Dropbox/University/Tesina/src/section9.py").read())
#### EXPLORE THE EFFECT THAT DIFFERENT HYPERPARAMETERS HAVE ON SVM-RBF

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

def calculate_all_potential():
    scores1= explore_rbf_potential(train="b278",test="b234")
    scores2= explore_rbf_potential(train="b278",test="b261")
    scores3= explore_rbf_potential(train="b234",test="b261")
    scores4= explore_rbf_potential(train="b234",test="b360")
    scores5= explore_rbf_potential(train="b261",test="b360")
    scores6= explore_rbf_potential(train="b261",test="b278")
    scores7= explore_rbf_potential(train="b360",test="b278")
    scores8= explore_rbf_potential(train="b360",test="b234")     

def plot_rbf_potential(train="b234",test="b278"):

    save_folder = "/home/jere/Desktop/section9/"

    with open(save_folder+ "_svmk_train="+train+"test="+test+"_scores.pkl", 'rb') as s_file:
        scores = pickle.load(s_file)

    c_values = np.logspace(-5, 12, 18)
    g_values = np.logspace(-15, 4,20)

    perf = np.zeros((18,20))

    for i in range(0,len(c_values)-1):
        c = c_values[i]
        for j in range(0,len(g_values)-1):
            gamma = g_values[j]
            perf[i,j] = scores[(c,gamma)]

    plt.figure(figsize=(6, 8))

    plt.imshow(perf, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0, midpoint=np.max(perf)*0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(g_values)), g_values, rotation=45)
    plt.yticks(np.arange(len(c_values)), c_values, rotation=0)
    plt.title('Average validation  Robust AUCPRC')
    plt.savefig(save_folder+"_svmk_train="+train+"test="+test+"_grid.png")

        
def calculate_all_grids():
    scores1= plot_rbf_potential(train="b278",test="b234")
    scores2= plot_rbf_potential(train="b278",test="b261")
    scores3= plot_rbf_potential(train="b234",test="b261")
    scores4= plot_rbf_potential(train="b234",test="b360")
    scores5= plot_rbf_potential(train="b261",test="b360")
    scores6= plot_rbf_potential(train="b261",test="b278")
    scores7= plot_rbf_potential(train="b360",test="b278")
    scores8= plot_rbf_potential(train="b360",test="b234")    
