exec(open("/home/jere/Dropbox/University/Tesina/src/section10.py").read())
### EXTRA EXPERIMENTS WITH RANDOM FORESTS


def random_forests_bins(train,test):
    
    save_folder = "/home/jere/Desktop/section11/"

    X,y=retrieve_tile(train)
    Xt,yt=retrieve_tile(test)   
    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    
    # Step 1: train RF, test it.
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)
    decs  = clf.predict_proba(Xt)[:,1]
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=1,label="Random Forest")

    # Step 2: train RF + previous standard scaling
    
    clf = Pipeline([("scaler",StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7))])
    clf.fit(X,y)
    decs  = clf.predict_proba(Xt)[:,1]
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=1,label="Standard scaler + Random Forest")    
    
    # Step 3: train RF + standard scaling + kbins
    clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
            ("scaler",StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7))])
    clf.fit(X,y)
    decs  = clf.predict_proba(Xt)[:,1]
    p,r,t = metrics.precision_recall_curve(yt,decs)
    ax.plot(r,p,linewidth=1,label="Standard scaler + 100bins quantile + Random Forest")             
    
    
    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[RF] Perfomance train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_rf_preproc_train="+train+"test="+test+"_curves.png")
    return 1
    
def rf_preproc_all():
    scores1= random_forests_bins(train="b278",test="b234")
    scores2= random_forests_bins(train="b278",test="b261")
    scores3= random_forests_bins(train="b234",test="b261")
    scores4= random_forests_bins(train="b234",test="b360")
    scores5= random_forests_bins(train="b261",test="b360")
    scores6= random_forests_bins(train="b261",test="b278")
    scores7= random_forests_bins(train="b360",test="b278")
    scores8= random_forests_bins(train="b360",test="b234")    
