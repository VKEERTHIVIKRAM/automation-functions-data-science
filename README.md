# automated-and-orignal-functions-data-science
A few functions which can be automated for data science and ease our work
## About it
Various data science steps which are repetative or Orignal. This can help save time and Help infer our data in a better way.
## Functions
#### 1. def  df_summary(df,target=None,num_but_cat_list=[],missing_only='no',impossible_only='no',outliers_only='no'):
Takes The data frame and returns a customized summary of it.
Summary conatins 
-Name and Total count
-Dtypes 
-Missing values number and percentage
-Number of unique values
-Range
-Mean,Median,Mode value and which imputaion method should be used if needed
-Impossible values presence and percentage
-Ouliers presence and percentage
-First,second and third value
-Entropy
#### 2. def corr_matrix(df,annot=True):
Returns a neater version of the heatmap of a correlation matrix of a dataframe
#### 3. def model_eval_classifier(algo,Xtrain,ytrain,Xtest,ytest,voting=None):
Returns ROC_AUC score ,ROC_AUC plot, Confusion matrix ,Accuracy score, Misclassified score, Classification report for binary and 
Confusion matrix ,Accuracy score, Misclassified score, Classification report for multiclass
#### 4. def features_to_drop(df,corr_value=0.95):
Returns a Dataframe with one of the feature having correlation value with another feature above given value is dropped
#### 5. def sensitivity_specificity_from_threshold(algo,Xtest,ytest,threshold=0.5):
Returns sensitivity and specificity from a given treshold
#### 6. def algorithim_boxplot_comparison(X,y,algo_list=[],random_state=3,scoring_name1=None,scoring_name2=None,n_splits=3):
Returns an boxplot comparing different algorithims
#### 7. def forward_selection(X, y, significance_level=0.05,algo='logistic'):
Performs Forward selection on the data 
#### 8. def backward_elimination(X, y,significance_level = 0.05,algo='logistic'):
Performs Backward selection on the data 
#### 9. def stepwise_selection(X, y, significance_level=0.05,algo='logistic',SL_in=0.05,SL_out = 0.05):
Performs Step-wise selection on the data 
#### 10. def column_may_be_categorical(df,target=None,threshold=10):
Returns column which could be considred as categorical for analysis but is numerical in nature
#### 11. def optimized_n_cluster_value(df,range_min=2,range_max=10,random_state=3):
Plot of No. of Clusters vs SSD and Plot of No. of Clusters vs Silhouette Score. From these plots We can infer n_cluster value
