#NU-SVM and oneclass svm
exec(open("/home/jere/Dropbox/University/Tesina/src/section11.py").read())

from sklearn.svm import NuSVC

nu_range = list(np.linspace(0.0000001,0.001,100)) #+ [0.01,0.1,0.25,0.5,0.75,0.99]
c_range  = np.logspace(-5, 12, 18)

def explore_nu_svc(train="b234",test="b278"):
    
    ######################################
    save_folder = "/home/jere/Desktop/section12/"

    X,y=retrieve_tile(train,"1:500")
    Xt,yt=retrieve_tile(test,"1:500")   
    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    max_auc = 0


    ## RF utopy ################
    print("Starting RF")
    clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    clf.fit(X,y)
    decs  = clf.predict_proba(Xt)[:,1]
    p,r,t = metrics.precision_recall_curve(yt,decs)
    pf, rf = p, r
    ax.plot(r,p,linewidth=3,label="Random Forest")
    
    precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
    recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
    precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
    robust_auc_rf = auc(recall_interpolated, precision_interpolated)
    ############################
    
    print("Done RF")

    for nu in nu_range:
        print("Trying out nu",nu)

        try:
            clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                    ("scaler",StandardScaler()),
                    ("svm", NuSVC(nu=nu,max_iter=100000))])

            clf.fit(X, y)

            decs  = clf.decision_function(Xt)
            
            s = stats.describe(decs)
            if ( abs(s.minmax[0] - s.minmax[1]) < 0.1 ):
                scores[nu] = 0
                continue
            
            p,r,t = metrics.precision_recall_curve(yt,decs)

            precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            scores[nu] = auc(recall_interpolated, precision_interpolated)
            print("auc=",scores[nu])
            if (scores[nu] > max_auc and scores[nu] > 0.6 * robust_auc_rf):
                max_auc = scores[nu]
                ax.plot(r,p,linewidth=1,label="nu="+str(nu))
                best_prc_nu = (p,r)
        except:
            scores[nu]=-1

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[NU-SVM] Perfomance train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_nusvm_train="+train+"test="+test+"_curves.png")
    
    with open(save_folder+ "_nusvm_train="+train+"test="+test+"_scores.pkl", 'wb') as s_file:
        pickle.dump(scores,s_file)
    
    ########################################

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    max_auc = 0
    
    for c in c_range:
    
        clf = Pipeline([("discretizer",KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
                ("scaler",StandardScaler()),
                ("svm", LinearSVC(C=c,dual=True,max_iter=100000))])

        clf.fit(X, y)

        decs  = clf.decision_function(Xt)
        
        s = stats.describe(decs)
        if ( abs(s.minmax[0] - s.minmax[1]) < 0.1 ):
            scores[c] = 0
            continue
        
        p,r,t = metrics.precision_recall_curve(yt,decs)

        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        scores[c] = auc(recall_interpolated, precision_interpolated)
        
        if (scores[c] > max_auc):
            max_auc = scores[c]
            ax.plot(r,p,linewidth=1,label="c="+str(c))
            best_prc_svc = (p,r)
            

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[SVC] Perfomance train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_svc_train="+train+"test="+test+"_curves.png")
    
    with open(save_folder+ "_svc_train="+train+"test="+test+"_scores.pkl", 'wb') as s_file:
        pickle.dump(scores,s_file)

    ########################################################
    
    fig, ax = plt.subplots(figsize=(20,10))
    
    ax.plot(best_prc_svc[1],best_prc_svc[0],linewidth=1,label="BEST SVC")
    ax.plot(best_prc_nu[1],best_prc_nu[0],linewidth=1,label="BEST NU-SVC")
    ax.plot(rf,pf,linewidth=3,label="Random Forest")

    leg = ax.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('[SVM-l vs RF] Perfomance train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_overall_train="+train+"test="+test+"_curves.png")
    return 1


def plot_svml_potential(train="b234",test="b278"):

    save_folder = "/home/jere/Desktop/section12/"

    with open(save_folder+ "_nusvm_train="+train+"test="+test+"_scores.pkl", 'rb') as s_file:
        scores = pickle.load(s_file)

    fig, ax = plt.subplots(figsize=(20,10))

    y = [scores[nu] for nu in nu_range] 

    ax.plot(nu_range, y ,linewidth=1,marker='.')

    plt.xlabel('nu')
    plt.ylabel('robust AUC-PRC')
    plt.title('[nu-svm] Effect of nu. train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_nuoverall_train="+train+"test="+test+"_curves.png")


    with open(save_folder+ "_svc_train="+train+"test="+test+"_scores.pkl", 'rb') as s_file:
        scores = pickle.load(s_file)


    fig, ax = plt.subplots(figsize=(20,10))

    y = [scores[c] for c in c_range] 

    ax.plot(list(np.log( list(scores.keys()) )), y ,linewidth=1,marker='.')
    plt.xlabel('log(C)')
    plt.ylabel('robust AUC-PRC')
    plt.title('[C-svm] Effect of C. train=' + str(train) + ' testing in '+str(test))

    plt.savefig(save_folder+"_Coverall_train="+train+"test="+test+"_curves.png")

      
def full_experiment():
    scores1= explore_nu_svc(train="b278",test="b234")
    scores2= explore_nu_svc(train="b278",test="b261")
    scores3= explore_nu_svc(train="b234",test="b261")
    scores4= explore_nu_svc(train="b234",test="b360")
    scores5= explore_nu_svc(train="b261",test="b360")
    scores6= explore_nu_svc(train="b261",test="b278")
    scores7= explore_nu_svc(train="b360",test="b278")
    scores8= explore_nu_svc(train="b360",test="b234")  
