""" VISUALIZATION """


def plot_fetures_distribution(tile="b278"):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    X,y = retrieve_tile(tile,"full") 

    h = 4
    k = 5

    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(30.0,15))
    for i in range(h):
        for j in range(k):
            index = j*k+i
            col = X.columns[index]
            a = (X[col]).hist(bins=100,ax=ax[i,j])  
            a.set_title(col)
            a.tick_params(axis='both', which='major', labelsize= 0)

    fig.suptitle('Features in tile '+tile+' part 1')
    plt.savefig(save_folder+"allfeatures_"+tile+"_1.png")
    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(30.0,15))

    for i in range(h):
        for j in range(k):
            index = j*k+i+20
            col = X.columns[index]
            a = (X[col]).hist(bins=100,ax=ax[i,j])  
            a.set_title(col)
            a.tick_params(axis='both', which='major', labelsize= 0)

    fig.suptitle('Features in tile '+tile+' part 2')
    plt.savefig(save_folder+"allfeatures_"+tile+"_2.png")
    fig, ax = plt.subplots(h, k, sharey=False,sharex=False,figsize=(30,15))

    for i in range(h):
        for j in range(k):
            index = j*k+i+40    
            if (index < len(X.columns)):
                col = X.columns[index]
                a = (X[col]).hist(bins=100,ax=ax[i,j])  
                a.set_title(col)
                a.tick_params(axis='both', which='major', labelsize= 0)

    fig.suptitle('Features in tile '+tile+' part 3')
    plt.savefig(save_folder+"allfeatures_"+tile+"_3.png")



        #(X[index])[y==0]..hist(bins=2000)(label="No-RRL")
        #(X[index])[y==1]..hist(bins=2000)(kind='kde',label="RRL")
        #leg = ax.legend()

def compare_same_feature_different_tiles(f_index=0,tiles=["b234","b360","b278","b261"]):

    save_folder = "/home/jere/Desktop/section7-fs/automated/"

    fig, ax = plt.subplots(2, 2, sharey=True,sharex=True,figsize=(15.0, 10.0))


    X,y = retrieve_tile(tiles[0],"full") 
    col = X.columns[f_index]

    a = (X[col]).hist(bins=100,ax=ax[0,0],density=True)  
    a.set_title(tiles[0])


    X,y = retrieve_tile(tiles[1],"full") 
    a = (X[col]).hist(bins=100,ax=ax[0,1],density=True)  
    a.set_title(tiles[1])

    X,y = retrieve_tile(tiles[2],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,0],density=True)  
    a.set_title(tiles[2])

    X,y = retrieve_tile(tiles[3],"full") 
    a = (X[col]).hist(bins=100,ax=ax[1,1],density=True)  
    a.set_title(tiles[3])

    fig.suptitle('Feature '+col)
    plt.savefig(save_folder+"feature_comparison_"+str(f_index)+".png")


def compare_all_features():
    for i in range(62):
        compare_same_feature_different_tiles(f_index=i)
