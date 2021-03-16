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




#######################################################################################
################# DEBUGGING RBF: WHY DOES IT WORK SO POORLY?? #########################



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
