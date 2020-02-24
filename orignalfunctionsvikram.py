def  df_summary(df,target=None,num_but_cat_list=[],missing_only='no',impossible_only='no',outliers_only='no'):
    """ Parameters-
        df : DataFrame
        target : Target varaible of the data set. If given will not calculate for it
        num_but_cat_list : These are numerical but we will consider them as categorical
        only_missing : 'yes' or 'no'. Default is 'no'
        only_impossible : 'yes' or 'no'. Default is 'no'
        only_outliers : 'yes' or 'no'.  Default is 'no'
        Returns:
        Summary of table(Impossible values may show wrong in case of column inherently
        containing and requiring special characters.When missing values are present sometimes(very less chance)
        will also show impossible values as present.)
        Summary conatins 
        1.Name and Total count
        2.Dtypes 
        3.Missing values number and percentage
        4.Number of unique values
        5.Range
        6.Mean,Median,Mode value and which imputaion method should be used if needed
        7.Impossible values presence and percentage
        8.Ouliers presence and percentage
        9.First,second and third value
        10.Entropy"""
    import pandas as pd
    import re
    from scipy import stats
    if target!=None and target in df.columns:
        dfn=df.drop(labels=target,axis=1)
    elif target==None:
        dfn=df
    else:
        raise ValueError('Target column inputed not in DataFrame')
    print(f"Dataset Shape: {dfn.shape}")
    summary = pd.DataFrame(dfn.dtypes,columns=['dtypes'])
    summary['Name'] = list(dfn.columns)
    summary['Total_Count']=dfn.isna().count()
    summary['Missing'] = dfn.isnull().sum().values
    summary['Missing_Percentage']=(dfn.isnull().sum().values/dfn.isna().count())*100
    summary['Uniques'] = dfn.nunique().values
    l9=[]
    for i in dfn.columns:
        l10=[]
        if dfn[i].dtype!='O' and i not in num_but_cat_list:
            l10.append(min(dfn[i].dropna()))
            l10.append(max(dfn[i].dropna()))
        elif dfn[i].dtype=='O' or i in num_but_cat_list:
            l10.append('Cannot find range for object')
        else:
            raise ValueError('Backend code wrong at this point. Please rewrite the backend code')
        l9.append(l10)
    summary['Range']=l9     
    l1=[]
    l2=[]
    for i in dfn.columns:
        if dfn[i].dtype!='O' and i not in num_but_cat_list:
            a=dfn[i].mean()
            b=dfn[i].median()
            l3=[]
            l3.append(a)
            l3.append(b)
            l2.append(l3)
            l1.append('Mean_or_Median')
        elif dfn[i].dtype=='O' or i in num_but_cat_list :
            a=dfn[i].mode()[0]
            l2.append(a)
            l1.append('Mode')
    summary['Mean_Median_Mode']=l2
    summary['Imputation_Method']=l1
    l4=[]
    l6=[]
    for i in dfn.columns:
        j=0
        l5=[]
        for k in dfn[i].dropna().astype(int,errors='ignore'):
            z=re.sub(r'\W+', '', str(k))
            if str(z)==str(k):
                l5.append(True)
            else:
                l5.append(False)
        for p in l5:
            if p==True:
                j=j+1
        if j==len(l5):
            l4.append('Not Present')
            l6.append(0)
        else:    
            l4.append('Present')
            l6.append(((len(l5)-j)/len(l5))*100)    
    summary['Impossible_Values']=l4
    summary['Impossible_Values_Percentage']=l6
    l7=[]
    l8=[]
    for i in dfn.columns:
        if dfn[i].dtype!='O' and i not in num_but_cat_list:
            x1=dfn[i].quantile(0.25)
            x2=dfn[i].quantile(0.75)
            x3=x2-x1
            a= (len(dfn[(dfn[i]>=(x1-1.5*(x3)))&(dfn[i]<=(x2+1.5*(x3)))])/len(dfn[i]))
            if a==1:
                l7.append('Outliers Are Not There')
            elif a<1:
                l7.append('Outliers are There')
            else:
                raise ValueError('Backend code wrong at this point. Please rewrite the backend code')
            l8.append((1-a)*100)
        elif dfn[i].dtype=='O' or i in num_but_cat_list:
            l7.append('Cant Find Outlier For object')
            l8.append(0)
        else:
            raise ValueError('Backend code wrong at this point. Please rewrite the backend code')
    summary['Outliers_Present_or_Absent']=l7
    summary['Outliers_Percentage']=l8
    summary['First Value'] = dfn.loc[0].values
    summary['Second Value'] = dfn.loc[1].values
    summary['Third Value'] = dfn.loc[2].values
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(dfn[name].value_counts(normalize=True), base=2),2) 
    summary.style.set_properties(subset=['Name'],**{'font-weight': 'bold'})
    a=summary
    if missing_only=='yes':
        b=a[a['Missing']!=0]
    elif missing_only=='no':
        b=a
    else:
        raise ValueError('Can only input yes or no')
    if impossible_only=='yes':
        c=b[b['Impossible_Values']=='Present']
    elif impossible_only=='no':
        c=b
    else:
        raise ValueError('Can only input yes or no')
    if outliers_only=='yes':
        d=c[c['Outliers_Percentage']!=0]
    elif outliers_only=='no':
        d=c
    else:
        raise ValueError('Can only input yes or no')
    return d
                                        
    


def svisualizer(x, ncluster):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import numpy as np
    from matplotlib import cm
    from sklearn.metrics import silhouette_samples 

    km = KMeans(n_clusters=ncluster, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(x)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0

    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.show()	


def cluster_plot(data, nclusters):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    X = data.copy()

    km = KMeans(n_clusters=nclusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)


    # Visualize it:
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=km.labels_.astype(float))

    # plot the centroids
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

def corr_matrix(df,annot=True):
    """Paramters-
       df : Dataframe
       annot: Wether to annot correlation values in the Heatmap. Default is True
       Returns-
       Heatmap of correlation matrix of the dataframe"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    if annot==True:
        annot=True
    elif annot==False:
        annot=False       
    else:
        raise ValueError('Use only True or False')
    df=df.corr()
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    
    sns.heatmap(df, mask=mask, cmap=cmap, vmax=.3, center=0, annot=annot, square=True, linewidths=.5, cbar_kws={"shrink": .5})


def model_eval_classifier(algo,Xtrain,ytrain,Xtest,ytest,voting=None):
    """Used for evaluating model performance for classification
       Can also be used when algorithim is a Voting Classifier
       Parameters-
       algo : Model Algorithim used. eg random forest,logistic regression etc.
       Xtrain : Train data without target variable.
       ytrain : Train data with only target variable.
       Xtest : Test data without target variable.
       ytest : Test data with only target variable.
       Returns-
       ROC_AUC score(Not for above 2 subcategories in y)
       ROC_AUC plot(Not for above 2 subcategories in y)
       Confusion matrix with heatmap
       Accuracy score
       Misclassified score
       Classification report"""
    from sklearn.metrics import confusion_matrix , accuracy_score, classification_report,roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # for 2 subcategories in target will give roc_auc_score and plot, otherwise for 3 and above will not give these 2 but will give rest 
    
    algo.fit(Xtrain,ytrain)
 
    y_train_pred=algo.predict(Xtrain)
    y_test_pred=algo.predict(Xtest)
    if voting=='soft' or voting==None:
        y_train_prob=algo.predict_proba(Xtrain)[:,1]
        y_test_prob=algo.predict_proba(Xtest)[:,1]
    elif voting=='hard':
        pass
    else:
        raise ValueError('Enter either hard or soft or default None nothing else ')

    print('Confusion Matrix-Train\n',confusion_matrix(ytrain,y_train_pred))
    sns.heatmap(confusion_matrix(ytrain,y_train_pred),annot=True,cmap="coolwarm_r",linewidths=0.5)
    plt.show()
    print('Accuracy Score-Train\n',accuracy_score(ytrain,y_train_pred))
    print('Misclassified Score-Train\n',1-accuracy_score(ytrain,y_train_pred))
    print('Classification Report-Train\n',classification_report(ytrain,y_train_pred))
    if (len(np.unique(ytrain))==2 and voting==None) or (len(np.unique(ytrain))==2 and voting=='soft'):
        print('AUC Score-Train\n',roc_auc_score(ytrain,y_train_prob))
    else:
        pass
    print('\n'*2)
    print('Confusion Matrix-Test\n',confusion_matrix(ytest,y_test_pred))
    sns.heatmap(confusion_matrix(ytest,y_test_pred),annot=True,cmap="coolwarm_r",linewidths=0.5)
    plt.show()
    print('Accuracy Score-Test\n',accuracy_score(ytest,y_test_pred))
    print('Misclassified Score-Test\n',1-accuracy_score(ytest,y_test_pred))
    print('Classification Report-Test\n',classification_report(ytest,y_test_pred))
    if (len(np.unique(ytrain))==2 and voting==None) or (len(np.unique(ytrain))==2 and voting == 'soft'):
        print('AUC Score-Test\n',roc_auc_score(ytest,y_test_prob))
        print('\n'*3)
        print('Plot')
        fpr,tpr,thresholds= roc_curve(ytest,y_test_prob)
        if thresholds[0]>1:
            thresholds[0]=thresholds[0]+(1-thresholds[0])    
        fig,ax1 = plt.subplots()
        ax1.plot(fpr,tpr)
        ax1.plot(fpr,fpr)
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TRP')
        ax2=ax1.twinx()
        ax2.plot(fpr,thresholds,'-g')
        ax2.set_ylabel('TRESHOLDS')
        plt.show()
    else:
        pass


def features_to_drop(df,corr_value=0.95):
    import numpy as np
    
    """
    Used to drop Highly correlated columns.
    Parameters-
    df : dataFrame
    corr_value : Correlation Value .Default is 0.95
    Returns-
    Columns which were dropped
    New dataframe
    """
    corrMatrix = df.corr()
    # Select upper triangle of correlation matrix
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    x= [column for column in upper.columns if any(upper[column] > corr_value)]
    toDrop = x
    print("Correlated features which are dropped: {}".format(toDrop))
    return df.drop(toDrop, inplace=True, axis=1)

def sensitivity_specificity_from_threshold(algo,Xtest,ytest,threshold=0.5):
    """Parameters-
       algo : Model Algorithim used. eg random forest,logistic regression etc.
       Xtest : Test data without target variable.
       ytest : Test data with only target variable.
       threshold : threshold value. Default = 0.5
       Returns-
       Specificity,Sensitivity"""
    from sklearn.metrics import confusion_matrix
    y_test_pred=[]
    z2= (algo.predict_proba(Xtest)[:,1]<=threshold)
    y_test_prob=algo.predict_proba(Xtest)[:,1]
    for i in z2:
        if i == True:
            y_test_pred.append(0)
        else:
            y_test_pred.append(1)
    specificity= confusion_matrix(ytest,y_test_pred)[0][0]/(confusion_matrix(ytest,y_test_pred)[0][0]+confusion_matrix(ytest,y_test_pred)[0][1])
    sensitivity= confusion_matrix(ytest,y_test_pred)[1][1]/(confusion_matrix(ytest,y_test_pred)[1][0]+confusion_matrix(ytest,y_test_pred)[1][1])
    return specificity,sensitivity


def algorithim_boxplot_comparison(X,y,algo_list=[],random_state=3,scoring_name1=None,scoring_name2=None,n_splits=3):
    """To compare metric of different algorithims
       Paramters-
       algo_list : a list conataining algorithim models like random forest, decision trees etc.
       X : dataframe without Target variable
       y : dataframe with only Target variable
       random_state : The seed of randomness. Default is 3
       n_splits : Number of splits used. Default is 3
       ( Default changes from organization to organization)
       Returns-
       mean accuracy and the standard deviation accuracy.
       Box Plot of Acuuracy"""
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    import numpy as np
    results = []
    names = []
    if len(np.unique(y))==2 or scoring_name1 :
        scoring = 'roc_auc'
    elif len(np.unique(y))>2 or scoring_name2:
        scoring = 'f1_weighted'
    else:
        raise ValueError('Incorrect Scoring name')
    for algo_name, algo_model in algo_list:
        kfold = model_selection.KFold(n_splits=n_splits, random_state=random_state)
        cv_results = model_selection.cross_val_score(algo_model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(algo_name)
        msg = "%s: %f (%f)" % (algo_name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def forward_selection(X, y, significance_level=0.05,algo='logistic'):
    """To perform Foward selection on the data
       Parameters-
       X : Data without target variable
       y : Data with target variable
       significance_level : Significance_level being used. Default is 0.05
       algo : Algorithim to use either linear or logistic. Default is logistic
       Returns-
       Best Features"""
    import statsmodels.api as sm
    import pandas as pd

    if algo=='logistic':
        algo=sm.Logit
    elif algo =='linear':
        algo=sm.OLS
    else:
        raise ValueError('Enter either linear or logistic nothing else')
        
    initial_features = X.columns.tolist()
    best_features = []
    while (len(list(initial_features))>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = algo(y, sm.add_constant(X[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_elimination(X, y,significance_level = 0.05,algo='logistic'):
    """To perform Foward selection on the data
       Parameters-
       X : Data without target variable
       y : Data with target variable
       significance_level : Significance_level being used. Default is 0.05
       algo : Algorithim to use either linear or logistic. Default is logistic
       Returns-
       Best Features"""
    import statsmodels.api as sm
    import pandas as pd
    if algo=='logistic':
        algo=sm.Logit
    elif algo =='linear':
        algo=sm.OLS
    else:
        raise ValueError('Enter either linear or logistic nothing else')
    features = X.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(X[features])
        p_values = algo(y, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features


def stepwise_selection(X, y, significance_level=0.05,algo='logistic',SL_in=0.05,SL_out = 0.05):
    """To perform Foward selection on the data
       Parameters-
       X : Data without target variable
       y : Data with target variable
       significance_level : Significance_level being used. Default is 0.05
       algo : Algorithim to use either linear or logistic. Default is logistic
       Returns-
       Best Features"""
    import statsmodels.api as sm
    import pandas as pd

    if algo=='logistic':
        algo=sm.Logit
    elif algo =='linear':
        algo=sm.OLS
    else:
        raise ValueError('Enter either linear or logistic nothing else')
    initial_features = X.columns.tolist()
    best_features = []
    while (len(list(initial_features))>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = algo(y, sm.add_constant(X[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(X[best_features])
                p_values = algo(y, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features

def column_may_be_categorical(df,target=None,threshold=10):
    """ Parameters-
        df : DataFrame
        target : Target varaible of the data set. If given will not calculate for it
        threshold : The max number of unique values to check for
        Returns-
        The columns mentioned may be considred categorical for analysis. Upon further introspection can be found out"""

    if target!=None and target in df.columns:
        dfn=df.drop(labels=target,axis=1)
    elif target==None:
        dfn=df
    l1=[]
    for i in dfn.columns:
        if dfn[i].dtype!='O':
            if (len(dfn[i].unique())<threshold)==True:
                l1.append(i)
    else:
        pass
    print('The columns which could be considred categorical while performing analysis is',l1,'.Further analysis on these columns needed to confirm')