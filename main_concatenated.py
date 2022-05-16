
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

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
import operator

import tensorflow as tf
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile,chi2,SelectFpr
from sklearn.model_selection import StratifiedKFold as kf

def generateStandardTimeSeriesStructure(all_releases_df, ws, featureList):
 
    print("Generating a new dataframe without containing the last release...")
    df = all_releases_df[all_releases_df['release'] != all_releases_df['release'].max()]
    print("... DONE!")

    df.drop(columns=["project","commit","TOTAL_CHANGES","release","will_change"])
    
    print("checking class larger than window size...")

    window_size=ws

    class_names_list = df['class'].unique().tolist()
    classes_to_drop_list = list()
    for class_name in class_names_list:
        if len(df[df['class'] == class_name].iloc[::-1]) <= window_size:
            for drop_class in df.index[df['class']==class_name].tolist():
                classes_to_drop_list.append(drop_class)


    df = df.drop(classes_to_drop_list, axis=0)
    df = df.iloc[::-1]

    print("DONE")
    
    print("Setting the features...")
    class_names_list = df['class'].unique().tolist()
    features_list = featureList
    print("DONE")
    
    timeseries_list = list()
    timeseries_labels = list()
    for class_name in class_names_list:
        class_sequence = df[df['class'] == class_name].reset_index()
        for row in range(len(class_sequence)-1):
            window = list()
            # print('row: ', row)
            if row + window_size < len(class_sequence) + 1:
                for i in range(window_size):
                    #print(row+i)
                    window.extend(class_sequence.loc[row + i, features_list].values.astype(np.float64))
                timeseries_labels.append(class_sequence.loc[row + i, 'will_change'])
                timeseries_list.append(window)
                
    timeseries_X = np.array(timeseries_list)
    timeseries_X = timeseries_X[:, ~np.isnan(timeseries_X).any(axis=0)]
    timeseries_labels = np.array(timeseries_labels).astype(np.bool)
    #np.savetxt("results/test.csv",timeseries_X, delimiter=",")
    
    return timeseries_X, timeseries_labels

def get_scores(y_test, y_pred, dataset,algorithm,rs,model,ws):
    scores = []
    scores.append(dataset)
    scores.append(algorithm)
    scores.append(ws)
    scores.append(model)
    scores.append(rs)

    scores.append(f1_score(y_test, y_pred, average='micro'))
    print("F1-Score(micro): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average='macro'))
    print("F1-Score(macro): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average='weighted'))
    print("F1-Score(weighted): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average=None))
    print("F1-Score(None): " + str(scores[-1]))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    #ACC
    scores.append(accuracy_score(y_test, y_pred, normalize=True))
    print("Accuracy: " + str(scores[-1]))
    
    #Sensitivity
    sensitivity = tp / (tp+fn)
    scores.append(sensitivity)
    print("Sensitivity: " + str(scores[-1]))

    #Specificity
    specificity = tn / (tn+fp)
    scores.append (specificity)
    print("Specificity: " + str(scores[-1]))
  
    #Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1],2)) + "]")
    plot_confusion_matrix(cnf_matrix,dataset)

    #ROC_AUC
    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))

    
    scores.append([tn, fp, fn, tp])
    head = ['Dataset','Algoritm','window','model','resample','F1-Score(micro)','F1-Score(macro)','F1-Score(weighted)','F1-Score(None)','Accuracy','Sensitivity','Specificity','ROC AUC score','Confusion matrix']

    if not os.path.exists('results/results-concat-no-feature-selection-model6-7.csv'):
        f = open("results/results-concat-no-feature-selection-model6-7.csv", "a")
        writer = csv.writer(f)
        writer.writerow(head)
        f.close()
    
    f = open("results/results-concat-no-feature-selection-model6-7.csv", "a")
    writer = csv.writer(f)
    writer.writerow(scores)
    f.close()

   
    
    return scores

def plot_confusion_matrix(cm,dataset,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/cf-' + dataset + '.png')
    plt.close()

def plot_confusion_matrixes(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plt.subplots(1,2,figsize=(20,4))
    #plt.subplot(1,2,1)
    #plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.subplot(1,2,2)
    plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

    plt.tight_layout()
    plt.show()

#UNDERSAMPLING
def LogisticRegr_(Xtrain, Ytrain, Xtest, Ytest, dataset,rs,model,ws):
    print("\nLOGISTIC REGRESSION")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_LR, xvl_LR = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_LR, yvl_LR = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        #model
        lr = LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced',n_jobs=-1)
        lr.fit(xtr_LR, ytr_LR.values.ravel())
        score = roc_auc_score(yvl_LR, lr.predict(xvl_LR))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))

    print("\nTEST SET:")
    get_scores(Ytest, lr.predict(Xtest), dataset, "LogistRegression",rs,model,ws)

def RandomForest_(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_RF, xvl_RF = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_RF, yvl_RF = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        #model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, n_jobs=-1)
        rf.fit(xtr_RF, ytr_RF.values.ravel())
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest), dataset,"RandomForest",rs,model,ws)

def NN_(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
   
    print("NEURAL NETWORK")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_NN, xvl_NN = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_NN, yvl_NN = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        #model
        nn = MLPClassifier(random_state=42)
        #nn.fit(xtr_NN, ytr_NN.values.ravel())
        grid = GridSearchCV(nn, {}, n_jobs=-1,
                    verbose=0)
        grid.fit(xtr_NN, ytr_NN.values.ravel())
        score = roc_auc_score(yvl_NN, grid.predict(xvl_NN))
        #score = roc_auc_score(yvl_NN, nn.predict(xvl_NN))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))   

    print("\nTEST SET:")
    get_scores(Ytest, grid.predict(Xtest), dataset,"MLP",rs,model,ws)

def DecisionTree_(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
    print("\nDECISION TREE")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_DT, xvl_DT = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_DT, yvl_DT = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        #model
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        #dt.fit(xtr_DT, ytr_DT.values.ravel())
        

        grid = GridSearchCV(dt, {}, n_jobs=-1,
                    verbose=0)
        grid.fit(xtr_DT, ytr_DT.values.ravel())
        score = roc_auc_score(yvl_DT, grid.predict(xvl_DT))
        #score = roc_auc_score(yvl_DT, dt.predict(xvl_DT))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1


    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, grid.predict(Xtest),dataset,"DT",rs,model,ws)

#OVERSAMPLING
def LogisticRegr_NoIloc(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
    print("\nLOGISTIC REGRESSION")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_LR, xvl_LR = Xtrain[train_index], Xtrain[test_index]
        ytr_LR, yvl_LR = Ytrain[train_index], Ytrain[test_index]

        #model
        lr = LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced',n_jobs=-1)
        lr.fit(xtr_LR, ytr_LR)
        score = roc_auc_score(yvl_LR, lr.predict(xvl_LR))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))

    print("\nTEST SET:")
    get_scores(Ytest, lr.predict(Xtest),dataset,"LogistRegression",rs,model,ws)

def RandomForest_NoIloc(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_RF, xvl_RF = Xtrain[train_index], Xtrain[test_index]
        ytr_RF, yvl_RF = Ytrain[train_index], Ytrain[test_index]
        

        #model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100,n_jobs=-1)
        rf.fit(xtr_RF, ytr_RF)
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest),dataset,"RandomForest",rs,model,ws)

def NN_NoIloc(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):

    print("NEURAL NETWORK")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_NN, xvl_NN = Xtrain[train_index], Xtrain[test_index]
        ytr_NN, yvl_NN = Ytrain[train_index], Ytrain[test_index]

        #model
        nn = MLPClassifier(random_state=42)
        nn.fit(xtr_NN, ytr_NN)
        score = roc_auc_score(yvl_NN, nn.predict(xvl_NN))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))   

    print("\nTEST SET:")
    get_scores(Ytest, nn.predict(Xtest),dataset,"MLP",rs,model,ws)

def DecisionTree_NoIloc(Xtrain, Ytrain, Xtest, Ytest,dataset,rs,model,ws):
    print("\nDECISION TREE")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_DT, xvl_DT = Xtrain[train_index], Xtrain[test_index]
        ytr_DT, yvl_DT = Ytrain[train_index], Ytrain[test_index]
        #model
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        dt.fit(xtr_DT, ytr_DT)
        score = roc_auc_score(yvl_DT, dt.predict(xvl_DT))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, dt.predict(Xtest),dataset,"DT",rs,model,ws)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    model1 = ["cbo", "wmc", "dit", "rfc", "lcom", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "visibleFieldsQty", "finalFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty",
    "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
    
    "BOC", "TACH", "FCH", "LCH", "CHO", "FRCH", "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF",
    
    "FANIN", "FANOUT",  "TotalClassMethod", "DiversityTotal"]
    
    #"LazyClass", "DataClass", "ComplexClass", "SpaghettiCode", "SpeculativeGenerality", "GodClass", "RefusedBequest", "ClassDataShouldBePrivate", "BrainClass", "TotalClass", "LongParameterList", "LongMethod", "FeatureEnvy", "DispersedCoupling", "MessageChain", "IntensiveCoupling", "ShotgunSurgery", "BrainMethod", "TotalMethod",  "DiversityMethod", "DiversityClass"
    model2 = ["cbo", "wmc", "dit", "rfc", "lcom", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "visibleFieldsQty", "finalFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty",
    "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
    "BOC", "TACH", "FCH", "LCH", "CHO", "FRCH", "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF", "FANIN", "FANOUT"]
    
    model3 = ["BOC", "TACH", "FCH", "LCH", "CHO", "FRCH", "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF", "LazyClass", "DataClass", "ComplexClass", "SpaghettiCode", "SpeculativeGenerality", "GodClass", "RefusedBequest", "ClassDataShouldBePrivate",
    "TotalClassMethod", "DiversityTotal"]

    model4 = ["cbo", "wmc", "dit", "rfc", "lcom", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "visibleFieldsQty", "finalFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty",
    "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
    "FANIN", "FANOUT", "TotalClassMethod", "DiversityTotal"]

    model5= ["LCH", "WFR", "FRCH", "FANOUT", "FANIN", "CHD", "ACDF", "ATAF", "WCD", "CSB"]

    model6= ["cbo", "wmc", "dit", "rfc", "lcom", "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty", "defaultMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty", "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty", "protectedFieldsQty", "defaultFieldsQty", "visibleFieldsQty", "finalFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty", "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty",
    "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode", "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
    "FANIN", "FANOUT"]

    model7 = ["BOC", "TACH", "FCH", "LCH", "CHO", "FRCH", "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF"]

    #datasets = ['commons-bcel','commons-io','junit4','pdfbox','wro4j']
    datasets = ['pdfbox','wro4j']
    
    main_columns = ["project","commit","class",
                "cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
                "AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential",																					
                "BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF",																														
                "FANIN","FANOUT","LazyClass","DataClass","ComplexClass","SpaghettiCode","SpeculativeGenerality","GodClass","RefusedBequest","ClassDataShouldBePrivate","BrainClass","TotalClass","LongParameterList","LongMethod","FeatureEnvy","DispersedCoupling","MessageChain","IntensiveCoupling","ShotgunSurgery","BrainMethod","TotalMethod","TotalClassMethod","DiversityTotal","DiversityMethod","DiversityClass",																				
                "TOTAL_CHANGES","release","will_change"]
    #resamples= ['NONE','RUS','ENN','TL','ROS','SMOTE','ADA']
    #resamples= ['RUS','ENN','TL','ROS','SMOTE','ADA']
    resamples= ['NONE','ROS','SMOTE','ADA']
    windowsize = [2,3,4]
    #models= [{'key':'model1', 'value': model1},{'key':'model2', 'value': model2},{'key':'model3', 'value': model3},{'key':'model4', 'value': model4} ]
    #models= [{'key':'model5', 'value': model5}]
    models= [{'key':'model6', 'value': model6},{'key':'model7', 'value': model7}]
    for dataset in datasets:
        for ws in windowsize:
            for rs in resamples:
                for model in models:
                    all_releases_df = pd.read_csv(
                        'datasets/' + dataset + '-all-releases.csv', usecols=main_columns)
                    all_releases_df = all_releases_df.fillna(0)

                    #x_raw = all_releases_df[model.get('value')]
                    #y_raw = all_releases_df['will_change']
                    #Feature selection
                    #X_new = SelectPercentile(chi2, percentile=50).fit(x_raw, y_raw)
                    #X_new = SelectFpr(chi2, alpha=0.05).fit(x_raw, y_raw)

                    #mask = X_new.get_support()  # list of booleans
                    #new_features = []  # The list of your K best features

                    #for bool, feature in zip(mask, model.get('value')):
                    #    if bool:
                    #        new_features.append(feature)
                    #print(new_features)

                    X, y = generateStandardTimeSeriesStructure(all_releases_df, ws, model.get('value'))

                    print("Declaring a dictionary to save results...")
                    results_dict = dict()
                    print("... DONE!")

                    print("Splitting dataset into train and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42)
                    print("General information:")
                    print("X Train set:",
                        X_train.shape[0], "X Test set:", X_test.shape[0])
                    print("y Train set:",
                        y_train.shape[0], "y Test set:", y_test.shape[0])
                    print("... DONE!")

                    print("Scaling features...")
                    scaler = MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))
                    X_test = pd.DataFrame(scaler.fit_transform(X_test))
                    print("... DONE!")

                    print("Setting stratified k-fold...")
                    k = 10
                    kf = StratifiedKFold(n_splits=k, shuffle=False)
                    print("k =", k)
                    print("... DONE!\n")
                    y_test = pd.DataFrame(y_test)
                    y_train = pd.DataFrame(y_train)
                   
                    #WITHOUT OVER OR UNDERSUMPLING
                    if rs == 'NONE':
                        RandomForest_(X_train, y_train, X_test, y_test, dataset,rs,model.get('key'),ws)
                        DecisionTree_(X_train, y_train, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_train, y_train, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_train, y_train, X_test, y_test,dataset,rs,model.get('key'),ws)
                    # UNERSAMPLING RUS','ENN','TL'
                    if rs == 'RUS':
                        X_RUS, y_RUS = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        y_RUS = pd.DataFrame(y_RUS)
                        
                        RandomForest_(X_RUS, y_RUS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        DecisionTree_(X_RUS, y_RUS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_RUS, y_RUS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_RUS, y_RUS, X_test, y_test,dataset,rs,model.get('key'),ws)
                    if rs == 'ENN':
                        X_ENN, y_ENN = EditedNearestNeighbours(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        RandomForest_(X_ENN, y_ENN, X_test, y_test,dataset,rs,model.get('key'),ws)
                        DecisionTree_(X_ENN, y_ENN, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_ENN, y_ENN, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_ENN, y_ENN, X_test, y_test,dataset,rs,model.get('key'),ws)
                    if rs == 'TL':
                        X_TL, y_TL = TomekLinks(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        RandomForest_(X_TL, y_TL, X_test, y_test,dataset,rs,model.get('key'),ws)
                        DecisionTree_(X_TL, y_TL, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_TL, y_TL, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_TL, y_TL, X_test, y_test,dataset,rs,model.get('key'),ws)
                    #OVERSAMPLING 'ROS','SMOTE','ADA'
                    if rs == 'ROS':
                        ros = RandomOverSampler(random_state=42)
                        X_ROS, y_ROS = ros.fit_resample(X_train, y_train)
                        DecisionTree_(X_ROS, y_ROS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        RandomForest_(X_ROS, y_ROS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_ROS, y_ROS, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_ROS, y_ROS, X_test, y_test,dataset,rs,model.get('key'),ws)
                    if rs == 'SMOTE':
                        sm = SMOTE(random_state=42)
                        X_SMO, y_SMO = sm.fit_resample(X_train, y_train)
                        RandomForest_( X_SMO, y_SMO, X_test, y_test,dataset,rs,model.get('key'),ws)
                        DecisionTree_( X_SMO, y_SMO, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_( X_SMO, y_SMO, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_( X_SMO, y_SMO, X_test, y_test,dataset,rs,model.get('key'),ws)
                    if rs == 'ADA':
                        ada = ADASYN(random_state=42)
                        X_ADA, y_ADA = ada.fit_resample(X_train, y_train)
                        RandomForest_(X_ADA, y_ADA, X_test, y_test,dataset,rs,model.get('key'),ws)
                        DecisionTree_(X_ADA, y_ADA, X_test, y_test,dataset,rs,model.get('key'),ws)
                        LogisticRegr_(X_ADA, y_ADA, X_test, y_test,dataset,rs,model.get('key'),ws)
                        NN_(X_ADA, y_ADA, X_test, y_test,dataset,rs,model.get('key'),ws)
