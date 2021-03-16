""" HANDLING IMBALANCE OF CLASSES """
exec(open("/home/jere/Dropbox/University/Tesina/src/section8.py").read())
from imblearn.combine import SMOTEENN
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


##### DEALING WITH CLASS IMBALANCE ####

results_folder_imbalance= "/home/jere/Desktop/section9/"

################################ Random Undersampling ###################################

# Effect of undersample in svm-l
def calculate_undersampling_performance(train="b278",test="b234",kernel="linear",random_id=""):
    
    Xt,yt=get_feature_selected_tile(test,kernel,train,"full")   

    scores = {}
    
    for rate in get_supported_rates(train) + ["full"]:
        X_res,y_res = get_feature_selected_tile(train,kernel,train,rate) 
        
        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 #("feature_selector", SelectKBest(get_optimal_parameters_fs("svml")["fc"], k=get_optimal_parameters_fs("svml")["k"])),
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svml")["C"]))])
    
        elif (kernel=="rbf"):
            clf = Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                     ("scaler",StandardScaler()), 
                   #  ("feature_selector", SelectKBest(get_optimal_parameters_fs("svmk")["fc"], k=get_optimal_parameters_fs("svmk")["k"])),
                     ("feature_map", Nystroem(gamma=get_optimal_parameters_fs("svmk")["gamma"], n_components=300)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svmk")["C"],))])
            

        clf.fit(X_res, y_res)
        
        
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        precision_fold, recall_fold = p[::-1], r[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)

        scores[str(rate)]=(p,r,robust_auc)

    with open(results_folder_imbalance+"undersampling/train="+train+"_test="+test+"_"+kernel+"_curves"+str(random_id)+".pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)   
         

def generate_undersampling_data(kernel="linear",random_id=""):
    generate_balanced_subtiles() # Randomize the indexes again!
    scores1= calculate_undersampling_performance(train="b278",test="b234",kernel=kernel,random_id=random_id)
    scores2= calculate_undersampling_performance(train="b278",test="b261",kernel=kernel,random_id=random_id)
    scores3= calculate_undersampling_performance(train="b234",test="b261",kernel=kernel,random_id=random_id)
    scores4= calculate_undersampling_performance(train="b234",test="b360",kernel=kernel,random_id=random_id)
    scores5= calculate_undersampling_performance(train="b261",test="b360",kernel=kernel,random_id=random_id)
    scores6= calculate_undersampling_performance(train="b261",test="b278",kernel=kernel,random_id=random_id)
    scores7= calculate_undersampling_performance(train="b360",test="b278",kernel=kernel,random_id=random_id)
    scores8= calculate_undersampling_performance(train="b360",test="b234",kernel=kernel,random_id=random_id)

def generate_underampling_data_mitigate_randomness(kernel="linear"):
    for r in ["","1","2"]:
        generate_undersampling_data(kernel,r)
        
def plot_undersampling_performance(train="b278",test="b234",kernel="linear",random_ids=["","1","2"]):
    
    X,y = retrieve_tile(train) 
    fig, ax = plt.subplots()

    all_aucs = {}
    i = 0
    for r in random_ids:
        
        # Read experiment i data
        with open(results_folder_imbalance+"undersampling/train="+train+"_test="+test+"_"+kernel+"_curves"+str(r)+".pkl", 'rb') as output:
            scores = pickle.load(output)    
              
        # Print experiment i AUCPRC
        aucs = {}
        for k in scores.keys(): 
            if (k!="full" and int(k) in get_supported_rates(train)):   
                (p,r,a) = scores[k]
                aucs[int(k)] = a
            elif (k=="full"):
                aucs[len(y)/sum(y)] = a
        ax.plot(list(aucs.keys()), list(aucs.values()),linestyle='-',linewidth=1.0,label="Experimento "+str(i)) 
        i=i+1   
        
        # Save data to calculate the avg of all experiments
        for k in aucs.keys(): 
            if k in all_aucs.keys():
                all_aucs[k] = all_aucs[k]+[aucs[k]]
            else:
                all_aucs[k]= [aucs[k]]

    # PLOT THE AVERAGE AUC
    avg_aucs = {}
    for k in all_aucs.keys():
        avg_aucs[k]= sum(all_aucs[k])/len(all_aucs[k]) 
        
    ax.plot(list(avg_aucs.keys()), list(avg_aucs.values()), marker='.',label="Promedio",color="black") 
    
    # GET THE BASELINES
    xdom = [0,250,500,750,1000,1250,1500,1750,2000,2500]
    horiz_line_data = np.array([get_baseline_fs_stage(train,test,kernel) for i in xdom])
    if (kernel=="linear"):
        label = "SVM-L Baseline"
    elif (kernel=="rbf"):
        label = "SVM-RBF Baseline"
    ax.plot(xdom, horiz_line_data, linestyle='--',linewidth=1.0,label=label,color='red') 

    horiz_line_data = np.array([get_baseline_fs_stage(train,test,"rf") for i in xdom])
    ax.plot(xdom, horiz_line_data, linestyle='--',linewidth=1.0,label="RF Baseline",color='g') 
    
    ax.set_ylim([0,.55])
    ax.set_xlim([-50,2500])

    plt.xlabel('Número de no-RRLs por cada RRL') 
    plt.ylabel('R-AUC-PRC')
    
    if (train=="b234" and test=="b261"):
        leg = ax.legend(loc="lower right")

    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_imbalance+"undersampling/train="+train+"_test="+test+"_"+kernel+"_individual_curves.png",bbox_inches='tight')
    plt.close(fig)
    return avg_aucs

def plot_undersampling_performance_all(kernel="linear"):
    scores1= plot_undersampling_performance(train="b278",test="b234",kernel=kernel)
    scores2= plot_undersampling_performance(train="b278",test="b261",kernel=kernel)
    scores3= plot_undersampling_performance(train="b234",test="b261",kernel=kernel)
    scores4= plot_undersampling_performance(train="b234",test="b360",kernel=kernel)
    scores5= plot_undersampling_performance(train="b261",test="b360",kernel=kernel)
    scores6= plot_undersampling_performance(train="b261",test="b278",kernel=kernel)
    scores7= plot_undersampling_performance(train="b360",test="b278",kernel=kernel)
    scores8= plot_undersampling_performance(train="b360",test="b234",kernel=kernel)

def undersampling_analyse_gain(kernel="linear"):

    scores = {}
    scores_diff = {}

    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            try:    
                aucs = plot_undersampling_performance(train,test,kernel)
            except:
                print("plot_undersampling_performance ",train,test,"failed")
                continue # We didn't calculate undersampling performance for that pair.
                
            for key in aucs.keys():
                if not(key in get_supported_rates(train)):
                    continue
                auc_pair = aucs[key]
                auc_diff = aucs[key] - get_baseline_fs_stage(train,test,kernel)  
                
                if (key in scores.keys()):
                    scores[key] = scores[key]+[auc_pair]
                    scores_diff[key] = scores_diff[key] + [auc_diff]
                else:
                    scores[key] = [auc_pair]
                    scores_diff[key] = [auc_diff]

    scores_avg = {}
    scores_min = {}
    scores_sdev = {}

    for key in scores.keys():
        if len(scores[key])==8:
            scores_avg[key] = np.mean(scores_diff[key])
            scores_sdev[key] = np.std(scores_diff[key])
            scores_min[key] = np.min(scores_diff[key])

    domain = scores_avg.keys()

    avgs = np.asarray(list(scores_avg.values()))
    sdevs = np.asarray(list(scores_sdev.values()))

    fig, ax = plt.subplots()

    ax.set_ylim([-0.4,0.05])
    #ax.set_xlim([0,1500])

    domain = [int(x) for x in domain]

    horiz_line_data = np.array([0 for i in domain])
    ax.plot(domain, horiz_line_data, 'r--',color='r') 
    ax.plot(domain, list(scores_avg.values()),label="Ganancia promedio",marker=".")
    ax.fill_between(domain, avgs-sdevs, avgs+sdevs,color="lightgrey")
    ax.plot(domain, list(scores_min.values()),label="Ganancia mínima",marker=".")
    #plt.xticks(xticks,xticks)

    plt.xlabel('Número de no-RRLs por cada RRL')
    plt.ylabel('Ganancia en R-AUCPRC respecto al baseline')
    
    if (kernel=="linear"):
        leg = ax.legend(loc="lower right")

    plt.savefig(results_folder_imbalance+"undersampling/"+kernel+"BEST.png",bbox_inches='tight')
    return (scores_avg,scores_min,scores_diff)

def get_optimal_hyp_undersampling(kernel="linear"):
    
    (scores_avg,scores_min,scores_diff)= undersampling_analyse_gain(kernel)
    [ print(k,"{:.3f}".format(scores_avg[k]),"{:.3f}".format(scores_min[k])) for k in scores_min.keys() if scores_min[k]>=-0.05 ]
###### 

def oversampling_generate_data(train="b278",test="b234",method="naive",kernel="linear"):
    
    X,y=get_feature_selected_tile(train,kernel,train,"full")   
    Xt,yt=get_feature_selected_tile(test,kernel,train,"full")   

    fig, ax = plt.subplots(figsize=(20,10))
    scores = {}
    scores_pafr = {}
    
    #min_prop = (sum(y)/len(y))*1.01
   # max_prop =  min_prop * 10 np.linspace(min_prop, max_prop, 10)
   
    
    proportions = [ 1.0/x for x in get_supported_rates(train) ]  + [(sum(y)/len(y))*1.01]

    scores = {}

    for i in proportions: 
        print("PROPORTION",i)
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
                
            if (kernel=="linear"):
                clf = Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                     ("scaler",StandardScaler()), 
                     #("feature_selector", SelectKBest(get_optimal_parameters_fs("svml")["fc"], k=get_optimal_parameters_fs("svml")["k"])),
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svml")["C"]))])
                
            elif (kernel=="rbf"):
                clf = Pipeline( 
                        [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                         ("scaler",StandardScaler()), 
                       #  ("feature_selector", SelectKBest(get_optimal_parameters_fs("svmk")["fc"], k=get_optimal_parameters_fs("svmk")["k"])),
                         ("feature_map", Nystroem(gamma=get_optimal_parameters_fs("svmk")["gamma"], n_components=300)), 
                         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svmk")["C"],))])
            
            clf.fit(X_resampled, y_resampled)
            decs  = clf.decision_function(Xt)
            p,r,t = metrics.precision_recall_curve(yt,decs)
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)
            scores[i] = (p,r,robust_auc)
        except:
            print("Failure at proportion",i,"method",method)
        
    with open(results_folder_imbalance+"oversampling/train="+train+"_test="+test+"_"+kernel+"_method"+method+"_curves.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)       
    
def oversampling_generate_data_all_tiles(method="naive",kernel="linear"):
    scores1= oversampling_generate_data(train="b278",test="b234",method=method,kernel=kernel)
    scores2= oversampling_generate_data(train="b278",test="b261",method=method,kernel=kernel)
    scores3= oversampling_generate_data(train="b234",test="b261",method=method,kernel=kernel)
    scores4= oversampling_generate_data(train="b234",test="b360",method=method,kernel=kernel)
    scores5= oversampling_generate_data(train="b261",test="b360",method=method,kernel=kernel)
    scores6= oversampling_generate_data(train="b261",test="b278",method=method,kernel=kernel)
    scores7= oversampling_generate_data(train="b360",test="b278",method=method,kernel=kernel)
    scores8= oversampling_generate_data(train="b360",test="b234",method=method,kernel=kernel) 
    
def oversampling_generate_data_all_tiles_all_methods(kernel="linear"):
    oversampling_generate_data_all_tiles("naive",kernel)
    oversampling_generate_data_all_tiles("SMOTE",kernel)
    oversampling_generate_data_all_tiles("ADASYN",kernel)
    oversampling_generate_data_all_tiles("SMOTEENN",kernel)

def plot_oversampling_performance(train="b278",test="b234",method="naive",kernel="linear"):
    with open(results_folder_imbalance+"oversampling/train="+train+"_test="+test+"_"+kernel+"_method"+method+"_curves.pkl", 'rb') as output:
        scores = pickle.load(output)  
              
    X,y = retrieve_tile(train) 

    aucs = {}
    for k in scores.keys(): 
            (p,r,a) = scores[k]
            aucs[1.0/k] = a

    fig, ax = plt.subplots()

    domain = get_supported_rates(train) + [len(y)/sum(y)]
    ax.plot(list(aucs.keys()), list(aucs.values()),marker='.') 
   
    ticks = [0,250,500,750,1000,1250,1500,1750,2000,3000]

    # GET THE BASELINE
    horiz_line_data = np.array([get_baseline_fs_stage(train,test,kernel) for i in ticks])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(ticks, horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_fs_stage(train,test,"rf") for i in ticks])
    ax.plot(ticks, horiz_line_data, 'r--',label="Random Forest",color='g') 
    
    ax.set_ylim([0,.55])
    ax.set_xlim([0,2500])

    plt.xlabel('Number of no-RRL per RRL') 
    plt.ylabel('Robust AUC-PRC')
    #plt.xticks(ticks,ticks)
    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_imbalance+"oversampling/"+method+"train="+train+"_test="+test+"_"+kernel+"_individual_curves.png",bbox_inches='tight')
    plt.close(fig)
    return aucs

def oversampling_plot_all_tiles(method="naive",kernel="linear"):
    scores1= plot_oversampling_performance(train="b278",test="b234",method=method,kernel=kernel)
    scores2= plot_oversampling_performance(train="b278",test="b261",method=method,kernel=kernel)
    scores3= plot_oversampling_performance(train="b234",test="b261",method=method,kernel=kernel)
    scores4= plot_oversampling_performance(train="b234",test="b360",method=method,kernel=kernel)
    scores5= plot_oversampling_performance(train="b261",test="b360",method=method,kernel=kernel)
    scores6= plot_oversampling_performance(train="b261",test="b278",method=method,kernel=kernel)
    scores7= plot_oversampling_performance(train="b360",test="b278",method=method,kernel=kernel)
    scores8= plot_oversampling_performance(train="b360",test="b234",method=method,kernel=kernel) 
  
def plot_balancing_unified_performance(train="b278",test="b234",kernel="linear"):
    
    #aucs_undersampling = plot_undersampling_performance(train,test,kernel)
    aucs_naive_over = plot_oversampling_performance(train,test,"naive",kernel)
    aucs_smote_over = plot_oversampling_performance(train,test,"SMOTE",kernel)
    aucs_adasyn_over = plot_oversampling_performance(train,test,"ADASYN",kernel)
    aucs_smoteenn_over = plot_oversampling_performance(train,test,"SMOTEENN",kernel)


    fig, ax = plt.subplots()
    # GET THE BASELINE
    ticks = [0,250,500,750,1000,1250,1500,1750,2000,3000]
    horiz_line_data = np.array([get_baseline_fs_stage(train,test,kernel) for i in ticks])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(ticks, horiz_line_data, 'r--',label=label) 
    horiz_line_data = np.array([get_baseline_fs_stage(train,test,"rf") for i in ticks])
    ax.plot(ticks, horiz_line_data, 'r--',label="Random Forest",color='g') 
    
    
    #ax.plot(list(aucs_undersampling.keys()), list(aucs_undersampling.values()),label="Undersampling",color="blue") 
    ax.plot(list(aucs_naive_over.keys()), list(aucs_naive_over.values()),label="Naive Oversampling") 
    ax.plot(list(aucs_smote_over.keys()), list(aucs_smote_over.values()),label="SMOTE Oversampling") 
    ax.plot(list(aucs_adasyn_over.keys()), list(aucs_adasyn_over.values()),label="ADASYN Oversampling",color="grey") 
    ax.plot(list(aucs_smoteenn_over.keys()), list(aucs_smoteenn_over.values()),label="SMOTEENN Hybrid",color="blueviolet") 


    ax.set_ylim([0,.55])
    ax.set_xlim([-10,2500])

    plt.xlabel('Número de no-RRLs por cada RRL') 
    plt.ylabel('Robust AUC-PRC')
    #plt.xticks(ticks,ticks)
    plt.title('Train '+train+' - Test '+test)
    if (train=="b234" and test=="b261"):
        leg = ax.legend(loc="lower right");
    plt.savefig(results_folder_imbalance+"oversampling/UNIFIED_train="+train+"_test="+test+"_"+kernel+"_curves.png",bbox_inches='tight')
    plt.close(fig)
    
def plot_all_balancing_unified_performance(kernel="linear"):
    scores1= plot_balancing_unified_performance(train="b278",test="b234",kernel=kernel)
    scores2= plot_balancing_unified_performance(train="b278",test="b261",kernel=kernel)
    scores3= plot_balancing_unified_performance(train="b234",test="b261",kernel=kernel)
    scores4= plot_balancing_unified_performance(train="b234",test="b360",kernel=kernel)
    scores5= plot_balancing_unified_performance(train="b261",test="b360",kernel=kernel)
    scores6= plot_balancing_unified_performance(train="b261",test="b278",kernel=kernel)
    scores7= plot_balancing_unified_performance(train="b360",test="b278",kernel=kernel)
    scores8= plot_balancing_unified_performance(train="b360",test="b234",kernel=kernel)

def calculate_oversampling_gain(kernel="linear",method="naive"):
    
    scores = {}
    scores_diff = {}

    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            try:    
                aucs = plot_oversampling_performance(train,test,method,kernel)
            except:
                print("faail")
                continue # We didn't calculate undersampling performance for that pair.
                
            for key in aucs.keys():
                if not(key in get_supported_rates(train)):
                    continue
                auc_pair = aucs[key]
                auc_diff = aucs[key] - get_baseline_fs_stage(train,test,kernel)  
                
                if (key in scores.keys()):
                    scores[key] = scores[key]+[auc_pair]
                    scores_diff[key] = scores_diff[key] + [auc_diff]
                else:
                    scores[key] = [auc_pair]
                    scores_diff[key] = [auc_diff]

    scores_avg = {}
    scores_min = {}
    scores_sdev = {}

    for key in scores.keys():
        if len(scores[key])==8:
            scores_avg[key] = np.mean(scores_diff[key])
            scores_sdev[key] = np.std(scores_diff[key])
            scores_min[key] = np.min(scores_diff[key])

    avgs = np.asarray(list(scores_avg.values()))
    sdevs = np.asarray(list(scores_sdev.values()))
    
    fig, ax = plt.subplots()

    ax.set_ylim([-0.3,0.05])
    ax.set_xlim([-10,1500])

    domain = [int(x) for x in scores_min.keys()]

    horiz_line_data = np.array([0 for i in domain])
    ax.plot(domain, horiz_line_data, 'r--',color='r') 
    ax.plot(domain, list(scores_avg.values()),label="Ganancia promedio",marker=".")
    ax.fill_between(domain, avgs-sdevs, avgs+sdevs,color="lightgrey")
    ax.plot(domain, list(scores_min.values()),label="Ganancia mínima",marker=".")
    #plt.xticks(xticks,xticks)

    plt.xlabel('Número de no-RRL por cada RRL')
    plt.ylabel('Ganancia en R-AUCPRC respecto al baseline')
    if (method=="naive" and kernel=="linear"):
        leg = ax.legend(loc="lower right")
    
    if (method=="naive"):
        method="aleatorio"
    plt.title("Oversampling "+method)
    
    plt.savefig(results_folder_imbalance+"oversampling/"+method+"_"+kernel+"BEST.png",bbox_inches='tight')
    
    return (scores_avg,scores_min)

def calculate_oversampling_gain_all():
    for kernel in ["linear","rbf"]:
        for method in ["naive","SMOTE","ADASYN","SMOTEENN"]:
            calculate_oversampling_gain(kernel,method)

def get_optimal_hyp_oversampling(kernel="linear"):
    (scores_avg,scores_min)= calculate_oversampling_gain(kernel,"naive")
    [ print(k,"{:.3f}".format(scores_avg[k]),"{:.3f}".format(scores_min[k])) for k in scores_min.keys() if scores_min[k]>=-0.1 ]
    
##### CLASS WEIGHT #######
def class_weight(test="b278",train="b234",kernel="linear"):
    
    X,y=get_feature_selected_tile(train,kernel,train,"full")   
    Xt,yt=get_feature_selected_tile(test,kernel,train,"full")  

    #fig, ax = plt.subplots(figsize=(20,10))
    scores = {}

    ############ USING CW
    for i in [1.5,2,5,10,25,50,75,100]: 


        if (kernel=="linear"):
            clf = Pipeline( 
                [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svml")["n_bins"], encode='ordinal', strategy='quantile')),
                 ("scaler",StandardScaler()), 
                 #("feature_selector", SelectKBest(get_optimal_parameters_fs("svml")["fc"], k=get_optimal_parameters_fs("svml")["k"])),
                 ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svml")["C"],class_weight={0:1,1:i}))])
            
        elif (kernel=="rbf"):
            clf = Pipeline( 
                    [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_fs("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
                     ("scaler",StandardScaler()), 
                   #  ("feature_selector", SelectKBest(get_optimal_parameters_fs("svmk")["fc"], k=get_optimal_parameters_fs("svmk")["k"])),
                     ("feature_map", Nystroem(gamma=get_optimal_parameters_fs("svmk")["gamma"], n_components=300)), 
                     ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_fs("svmk")["C"],class_weight={0:1,1:i}))])
                         
        clf.fit(X,y)
        decs  = clf.decision_function(Xt)
        p,r,t = metrics.precision_recall_curve(yt,decs)
        
        precision_fold, recall_fold, thresh = p[::-1], r[::-1], t[::-1]
        recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
        precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
        robust_auc = auc(recall_interpolated, precision_interpolated)
        scores[i]=(p,r,robust_auc)
        
    with open(results_folder_imbalance+"cw/train="+train+"_test="+test+"_"+kernel+"_curves.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL) 


def calculate_all_class_weight_data(kernel="linear"):
    scores1= class_weight(train="b278",test="b234",kernel=kernel)
    scores2= class_weight(train="b278",test="b261",kernel=kernel)
    scores3= class_weight(train="b234",test="b261",kernel=kernel)
    scores4= class_weight(train="b234",test="b360",kernel=kernel)
    scores5= class_weight(train="b261",test="b360",kernel=kernel)
    scores6= class_weight(train="b261",test="b278",kernel=kernel)
    scores7= class_weight(train="b360",test="b278",kernel=kernel)
    scores8= class_weight(train="b360",test="b234",kernel=kernel)
    

def plot_class_weight(train="b278",test="b234",kernel="linear"):
    with open(results_folder_imbalance+"cw/train="+train+"_test="+test+"_"+kernel+"_curves.pkl", 'rb') as output:
        scores = pickle.load(output)  
              
    X,y = retrieve_tile(train) 

    aucs = {}
    for k in scores.keys(): 
            (p,r,a) = scores[k]
            aucs[k] = a

    fig, ax = plt.subplots()

    domain = list(aucs.keys())
    ax.plot(domain, list(aucs.values()),marker='.') 
   
    # GET THE BASELINE
    horiz_line_data = np.array([get_baseline_fs_stage(train,test,kernel) for i in domain])
    if (kernel=="linear"):
        label = "SVM Lineal Baseline"
    elif (kernel=="rbf"):
        label = "SVM RBF Baseline"
    ax.plot(domain, horiz_line_data, 'r--',label=label) 

    horiz_line_data = np.array([get_baseline_fs_stage(train,test,"rf") for i in domain])
    ax.plot(domain, horiz_line_data, 'r--',label="Random Forest",color='g') 
    
    ax.set_ylim([0,.55])

    plt.xlabel('Weight multiplier for RRL class') 
    plt.ylabel('Robust AUC-PRC')
    #plt.xticks(ticks,ticks)
    plt.title('Train '+train+' - Test '+test)
    plt.savefig(results_folder_imbalance+"cw/train="+train+"_test="+test+"_"+kernel+"_individual_curves.png",bbox_inches='tight')
    plt.close(fig)
    return aucs

def calculate_all_class_weight_plots(kernel="linear"):
    scores1= plot_class_weight(train="b278",test="b234",kernel=kernel)
    scores2= plot_class_weight(train="b278",test="b261",kernel=kernel)
    scores3= plot_class_weight(train="b234",test="b261",kernel=kernel)
    scores4= plot_class_weight(train="b234",test="b360",kernel=kernel)
    scores5= plot_class_weight(train="b261",test="b360",kernel=kernel)
    scores6= plot_class_weight(train="b261",test="b278",kernel=kernel)
    scores7= plot_class_weight(train="b360",test="b278",kernel=kernel)
    scores8= plot_class_weight(train="b360",test="b234",kernel=kernel)
    
def class_weight_analyse_gain(kernel="linear"):

    scores = {}
    scores_diff = {}

    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue
            try:    
                aucs = plot_class_weight(train,test,kernel)
            except:
                print("plot_class_weight ",train,test,"failed")
                continue # We didn't calculate undersampling performance for that pair.
                
            for key in aucs.keys():
                auc_pair = aucs[key]
                auc_diff = aucs[key] - get_baseline_fs_stage(train,test,kernel)  
                
                if (key in scores.keys()):
                    scores[key] = scores[key]+[auc_pair]
                    scores_diff[key] = scores_diff[key] + [auc_diff]
                else:
                    scores[key] = [auc_pair]
                    scores_diff[key] = [auc_diff]

    scores_avg = {}
    scores_min = {}
    scores_sdev = {}

    for key in scores.keys():
        if len(scores[key])==8:
            scores_avg[key] = np.mean(scores_diff[key])
            scores_sdev[key] = np.std(scores_diff[key])
            scores_min[key] = np.min(scores_diff[key])

    domain = scores_avg.keys()

    avgs = np.asarray(list(scores_avg.values()))
    sdevs = np.asarray(list(scores_sdev.values()))

    fig, ax = plt.subplots()

    ax.set_ylim([-0.4,0.05])
    #ax.set_xlim([0,1500])

    domain = [int(x) for x in domain]

    horiz_line_data = np.array([0 for i in domain])
    ax.plot(domain, horiz_line_data, 'r--',color='r') 
    ax.plot(domain, list(scores_avg.values()),label="Ganancia promedio",marker=".")
    ax.fill_between(domain, avgs-sdevs, avgs+sdevs,color="lightgrey")
    ax.plot(domain, list(scores_min.values()),label="Ganancia mínima",marker=".")
    #plt.xticks(xticks,xticks)

    plt.xlabel('Weight multiplier for RRL class')
    plt.ylabel('Ganancia en R-AUCPRC respecto al baseline')
    
    leg = ax.legend(loc="lower right")

    plt.savefig(results_folder_imbalance+"cw/"+kernel+"BEST.png",bbox_inches='tight')
   
    return (scores_avg,scores_min,scores,scores_diff)

def get_optimal_hyp_cw(kernel="linear"):
    (scores_avg,scores_min,scores,scores_diff)= class_weight_analyse_gain(kernel)
    [ print(k,"{:.3f}".format(scores_avg[k]),"{:.3f}".format(scores_min[k])) for k in scores_min.keys() ]
      

############## FINAL COMPARISON


def get_optimal_parameters_imb(kernel="linear"):
    optimal = {}
    if (kernel=="linear" or kernel=="svml"):
        optimal["C"]=10
        optimal["n_bins"]=150
        optimal["k"]=48
        optimal["fc"]=mutual_info_classif
        optimal["undersampling_rate"] = "full"
        optimal["oversampling_rate"] = 1.0/500 #i.e. default
        
    elif (kernel=="rbf" or kernel=="svmk"):
        optimal["C"]=get_optimal_parameters_fs("svmk")["C"]
        optimal["gamma"]=get_optimal_parameters_fs("svmk")["gamma"]
        optimal["n_bins"]=get_optimal_parameters_fs("svmk")["n_bins"]       
        optimal["k"]=45
        optimal["fc"]=mutual_info_classif
        optimal["undersampling_rate"] = "full" #i.e. default
    return optimal

def generate_test_performance_data_imb(train_tile="b278",test_tiles=["b234","b261","b360"]):

    # RF
    #X,y=retrieve_tile(train_tile)
    #clf = RandomForestClassifier(n_estimators=400, criterion="entropy", min_samples_leaf=2, max_features="sqrt",n_jobs=7)
    #clf.fit(X,y)

    #SVM
    X,y=get_feature_selected_tile(train_tile,"linear",train_tile,get_optimal_parameters_imb("svml")["undersampling_rate"])
    
    ros = RandomOverSampler(sampling_strategy=get_optimal_parameters_imb("svml")["oversampling_rate"])
    X_resampled, y_resampled = ros.fit_resample(X, y)
                
    clf2 = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_imb("svml")["n_bins"], encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         ("feature_selector", SelectKBest(get_optimal_parameters_imb("svml")["fc"], k=get_optimal_parameters_imb("svml")["k"])),
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_imb("svml")["C"]))])
            
    clf2.fit(X_resampled,y_resampled)
    
    """
    #SVM-K
    X,y=get_feature_selected_tile(train_tile,"rbf",train_tile,get_optimal_parameters_imb("svmk")["undersampling_rate"])
    clf3 = Pipeline( 
        [("discretizer",KBinsDiscretizer(n_bins=get_optimal_parameters_imb("svmk")["n_bins"], encode='ordinal', strategy='quantile')),
         ("scaler",StandardScaler()), 
         #("feature_selector", SelectKBest(get_optimal_parameters_imb("svmk")["fc"], k=get_optimal_parameters_imb("svmk")["k"])),
         ("feature_map", Nystroem(gamma=get_optimal_parameters_imb("svmk")["gamma"], n_components=300)), 
         ("svm", LinearSVC(dual=False,max_iter=100000,C=get_optimal_parameters_imb("svmk")["C"]))])

                         
    clf3.fit(X,y)    
        """

    for test in test_tiles:
        curves = {}
        
        #RF
        #test_predictions = clf.predict_proba(Xtest)[:,1]
        #precision, recall, thresh = metrics.precision_recall_curve(ytest, test_predictions)
        #curves["rf"] = (precision,recall)
        
        # SVM-L
        Xt,yt=get_feature_selected_tile(test,"linear",train_tile,"full")
        test_predictions = clf2.decision_function(Xt)
        precision, recall, thresh = metrics.precision_recall_curve(yt, test_predictions)
        curves["svml"] = (precision,recall)
        
        """
        # SVM-K
        Xt,yt=get_feature_selected_tile(test,"rbf",train_tile,"full")
        test_predictions = clf3.decision_function(Xt)
        precision, recall, thresh = metrics.precision_recall_curve(yt, test_predictions)
        curves["svmk"] = (precision,recall)
        """
        with open(results_folder_imbalance+"best-train="+train_tile+ "test="+test+".pkl", 'wb') as output:
            pickle.dump(curves,output, pickle.HIGHEST_PROTOCOL)    

def generate_test_performance_data_imb_all():
    generate_test_performance_data_imb(train_tile="b278",test_tiles=["b234","b261","b360"])
    generate_test_performance_data_imb(train_tile="b234",test_tiles=["b278","b261","b360"])
    generate_test_performance_data_imb(train_tile="b261",test_tiles=["b234","b278","b360"])
    generate_test_performance_data_imb(train_tile="b360",test_tiles=["b234","b261","b278"])

def generate_test_performance_data_imb_subplots():
    
    scores = {}
    
    for train in ["b278","b234","b261","b360"]:
        for test in ["b278","b234","b261","b360"]:
            if (train==test):
                continue

            fig, ax = plt.subplots()

            # GET RANDOM FOREST DATA FROM PREPROCESSING STAGE
            with open(results_folder_preproces+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
                rf_curves = pickle.load(input_file)
                
            p,r = rf_curves["rf"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("rf",train,test)] = robust_auc
            ax.plot(r,p, label="Random Forest")


            # GET SVM-L DATA FROM IMBLEARNING STAGE
            with open(results_folder_imbalance+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)
                
            p,r = curves["svml"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svml",train,test)] = robust_auc
            ax.plot(r,p, label="Linear SVM")
            
            # GET SVM-K DATA FROM FEATURE SELECTION STAGE
            with open(results_folder_dimensionality_reduction+"best-train="+train+ "test="+test+".pkl", 'rb') as input_file:
                curves = pickle.load(input_file)
            p,r = curves["svmk"]
            precision_fold, recall_fold = p[::-1], r[::-1]
            recall_interpolated    = np.linspace(min_recall_global, 1, n_samples_prc)
            precision_interpolated = np.interp(recall_interpolated, recall_fold, precision_fold)
            robust_auc = auc(recall_interpolated, precision_interpolated)     
            scores[("svmK",train,test)] = robust_auc

            ax.plot(r,p, label="RBF SVM")

            plt.title('Train ' + train + "- Test" + test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            leg = ax.legend();
    
            plt.savefig(results_folder_imbalance+"best-train="+train+ "test="+test+".png",bbox_inches='tight')

    with open(results_folder_imbalance+"baseline_aucs.pkl", 'wb') as output:
        pickle.dump(scores,output, pickle.HIGHEST_PROTOCOL)   
        
# generate_table_comparison(results_folder_dimensionality_reduction+ "baseline_aucs.pkl", results_folder_preproces+"baseline_aucs.pkl")
# generate_table_comparison(results_folder_imbalance+ "baseline_aucs.pkl", results_folder_dimensionality_reduction+"baseline_aucs.pkl")

def get_baseline_imb_stage(train,test,method):
    
    if (method=="rbf"):
        method = "svmk"

    if (method=="linear" or method=="lineal"):
        method = "svml"
            
    if (method=="rf"):
        return get_baseline_preprocessing_stage(train,test,method)
    elif (method=="svmk"):
        return get_baseline_fs_stage(train,test,method)
    else:
        with open(results_folder_imbalance+"baseline_aucs.pkl", 'rb') as output:
            scores = pickle.load(output)      
        return scores[(method,train,test)]
