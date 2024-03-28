from exploratory.chi_squared import chi_squared
#from exploratory.mutual_information import mutual_information
#from exploratory.pearson_correlation import pearson_correlation
#from exploratory.variance_threshold import variance_threshold
import exploratory.hypothesis_testing
#from exploratory.tree_search import breadth_first_search, depth_first_search
from domain.feature_model.dependency_graph import DependencyGraph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from itertools import compress, product
import numpy as np
import pandas as pd

def binary_feature_permutations(length):
    return list(product([0,1], repeat=length))

def conditional_filtering(df, *features, **conditions):
    filtered_df = df[features] # compare bayesify or correct syntax
    if conditions["isometric"]:
        permutations = binary_feature_permutations(conditions["isometric"])
        for p in permutations:
            pass
            

    return filtered_df

# https://xai4se.github.io/defect-prediction/data-preprocessing.html

def select_kbest_chi2(X, y, k=10):
    return SelectKBest(chi2, k=k).fit_transform(X, y)

def recursive_feature_elimination(X, y, k=10):
    rfe_fit = RFE(RandomForestClassifier(n_estimators=100, random_state=0), n_features_to_select=k).fit(X, y)
    return list(compress(X.columns, rfe_fit.support_))

def auto_spearman(X, y, k=10):
    X_AS_train = X.copy()
    AS_metrics = X.columns
    print('(Part 1) Automatically select non-correlated metrics based on a Spearman rank correlation test')
    while True:
        corrmat = X.corr(method='spearman')
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)
        
        # identify correlated metrics with the correlation threshold of 0.7
        highly_correlated_metrics = ((corrmat > .7) | (corrmat < -.7)) & (corrmat != 1)
        n_correlated_metrics = np.sum(np.sum(highly_correlated_metrics))
        if n_correlated_metrics > 0:
            # find the strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index = False)
            find_top_corr.reset_index(inplace = True)
            find_top_corr = find_top_corr[find_top_corr['value'] != 1]
            top_corr_index = find_top_corr['value'].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            # get the 2 correlated metrics with the strongest correlation
            correlated_metric_1 = top_corr_i[0]
            correlated_metric_2 = top_corr_i[1]
            print('Step', count,'comparing between', correlated_metric_1, 'and', correlated_metric_2)
            
            # compute their correlation with other metrics outside of the pair
            correlation_with_other_metrics_1 = np.mean(abs_corrmat[correlated_metric_1][[i for i in top_corr_features if i not in [correlated_metric_1, correlated_metric_2]]])
            correlation_with_other_metrics_2 = np.mean(abs_corrmat[correlated_metric_2][[i for i in top_corr_features if i not in [correlated_metric_1, correlated_metric_2]]])
            print('>', correlated_metric_1, 'has the average correlation of', np.round(correlation_with_other_metrics_1, 3), 'with other metrics')
            print('>', correlated_metric_2, 'has the average correlation of', np.round(correlation_with_other_metrics_2,3) , 'with other metrics')
            # select the metric that shares the least correlation outside of the pair and exclude the other
            if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1
            print('>', 'Exclude',exclude_metric)
            count = count+1
            AS_metrics = list(set(AS_metrics) - set([exclude_metric]))
            X_AS_train = X_AS_train[AS_metrics]
        else:
            break

    print('According to Part 1 of AutoSpearman,', AS_metrics,'are selected.')
    from adapters.visualization import generate_heatmap
    generate_heatmap(X_AS_train)