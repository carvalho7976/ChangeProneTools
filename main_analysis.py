#%%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import time
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, 
                             average_precision_score, classification_report, accuracy_score)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


from sklearn.preprocessing import MinMaxScaler
import itertools


datasets = ['commons-bcel','commons-io','jfreechart','jhotdraw','junit4','pdfbox','wro4j']
main_columns = ["project","commit","class",
                "cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
                "AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential",																					
                "BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF",																														
                "FANIN","FANOUT","LazyClass","DataClass","ComplexClass","SpaghettiCode","SpeculativeGenerality","GodClass","RefusedBequest","ClassDataShouldBePrivate","BrainClass","TotalClass","LongParameterList","LongMethod","FeatureEnvy","DispersedCoupling","MessageChain","IntensiveCoupling","ShotgunSurgery","BrainMethod","TotalMethod","TotalClassMethod","DiversityTotal","DiversityMethod","DiversityClass",																				
                "TOTAL_CHANGES","release","will_change"]
for dataset in datasets:
        
   

    all_releases_df = pd.read_csv('datasets/' + dataset + '-all-releases.csv', usecols=main_columns)
    print("Total of instances:", all_releases_df.shape[0])


    print("Replace nan")
    all_releases_df = all_releases_df.dropna(axis=1, how='all')


    print("Filtering required columns into X features...")
    X = all_releases_df.drop(columns=["project","commit","class","TOTAL_CHANGES","release","will_change"])

    print("Setting y column containing label of change-proneness...")
    y = pd.DataFrame(all_releases_df.loc[:,'will_change'])

    print("Declaring a dictionary to save results...")
    results_dict = dict()

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print("General information:")
    print("X Train set:", X_train.shape[0], "X Test set:", X_test.shape[0])
    print("y Train set:", y_train.shape[0], "y Test set:", y_test.shape[0])

    #print(all_releases_df.describe())

    print("Distribution")
    #ax = y_train.groupby(['will_change'])['will_change'].count().plot.bar(title="Class Distribution", figsize=(5,5))
    #ax.figure.savefig('results/distribution-' + dataset + '.png')
    #plt.close()

    ax = sns.countplot(x="will_change", data=y_train)
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
    ax.figure.savefig('results/distribution-' + dataset + '.png')
    
    plt.close()

    print(y_train.groupby(['will_change'])['will_change'].count())

    print("Correlation")
    correlacao = X_train.corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
    correlacao.to_excel("results/correlacao-" + dataset + ".xlsx")
    

