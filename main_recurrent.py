import pandas as pd
import numpy as np
import statistics as st

from sklearn.model_selection import train_test_split

from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


import tensorflow as tf
import keras
from keras import backend as K
from sklearn.metrics import roc_auc_score
import time
import os

from sklearn.model_selection import StratifiedKFold

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def auc(y_true, y_pred):
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

def evaluation_metrics(y_test, y_pred, weight_t):
    print("ROC_AUC: " + str(roc_auc_score(y_test, y_pred, sample_weight=weight_t)))
    print("F1-Score: " + str(f1_score(y_test, y_pred, sample_weight=weight_t)))
    print("Precision: " + str(precision_score(y_test, y_pred, sample_weight=weight_t)))
    print("Recall: " + str(recall_score(y_test, y_pred, sample_weight=weight_t)))
    print("Accuracy: " + str(accuracy_score(y_test, y_pred, sample_weight=weight_t)))


def make_GRU(input_shape, output_dim, dropout=0.3):
    print("model dim: ", input_shape, output_dim)
    model = Sequential()
    model.add(GRU(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(64))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    model.summary()
    return model

def getTimeSeriesWindow(window_size, df):
    
    print("generating tensor...")
    class_names_list = df['class'].unique().tolist()
    classes_to_drop_list = list()
    for class_name in class_names_list:
        if len(df[df['class'] == class_name].iloc[::-1]) <= window_size:
            for drop_class in df.index[df['class']==class_name].tolist():
                classes_to_drop_list.append(drop_class)
    

    df = df.drop(classes_to_drop_list, axis=0)
    df = df.iloc[::-1]
    df = df.fillna(0)
    class_names_list = df['class'].unique().tolist()
    features_list = ["cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
                "AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential",																					
                "BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF",																														
                "FANIN","FANOUT","LazyClass","DataClass","ComplexClass","SpaghettiCode","SpeculativeGenerality","GodClass","RefusedBequest","ClassDataShouldBePrivate","BrainClass","TotalClass","LongParameterList","LongMethod","FeatureEnvy","DispersedCoupling","MessageChain","IntensiveCoupling","ShotgunSurgery","BrainMethod","TotalMethod","TotalClassMethod","DiversityTotal","DiversityMethod","DiversityClass"]
  
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
                    window.append(class_sequence.loc[row + i, features_list].values.astype(np.float64))
                timeseries_labels.append(class_sequence.loc[row + i, 'will_change'])
                timeseries_list.append(window)
    
    
    

    timeseries_tensor = np.array(timeseries_list)
    #np.savetxt("results/test2.csv",timeseries_tensor.reshape((3,-1)), fmt="%s", delimiter=",")

    #timeseries_tensor =  timeseries_tensor[:, ~np.isnan(timeseries_tensor).any(axis=2)]
    #timeseries_tensor[~np.isnan(timeseries_tensor).any(axis=2)].reshape(timeseries_tensor.shape[0], -1, timeseries_tensor.shape[2])
    
    timeseries_tensor = timeseries_tensor.transpose((0,2,1))
    timeseries_labels = np.array(timeseries_labels).astype(np.bool)

    timeseries_labels = timeseries_labels.reshape(-1, 1)

    will_change_labels = timeseries_labels.astype(np.int)

    not_will_change = np.invert(timeseries_labels).astype(np.int)

    timeseries_labels = np.concatenate((will_change_labels, not_will_change), axis=1)

    print("...DONE!")
    
    print("tensor dimensions:")
    print("tensor input X:", timeseries_tensor.shape)

    print("tensor input y:", timeseries_labels.shape)
  
    print("proportion of y labels:", np.unique(not_will_change, return_counts=True))
        
    print("Splitting dataset into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(timeseries_tensor, timeseries_labels, test_size=0.30, random_state=42)
    X_train = X_train.transpose((0,2,1))
    X_test = X_test.transpose((0,2,1))
    print("Tensor X train:", X_train.shape)
    print("Tensor y train:", y_train.shape)
    print("Tensor X test:", X_test.shape)
    print("Tensor y test:", y_test.shape)
    
    return X_train, X_test, y_train, y_test  


def applyTensor(window_size, df):

    X_train, X_test, y_train, y_test = getTimeSeriesWindow(window_size, df)
 
    print("computing weights...")
    
    fractions = 1-y_train.sum(axis=0)/len(y_train)
    weights = fractions[y_train.argmax(axis=1)]
    print("... DONE!")

    print("setting stratified k-fold...")
    k=5
    print("number of k:",k)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
   
    print("... DONE!")
    
    print("Executing algorithm...")
    
  
    
    
    lastModels = {}
    history_general = {}
    val_history_general = {}
    
    set_number_of_epochs=20
    set_batch_size=512
    print("number of epochs:", set_number_of_epochs)
    print("number of batch:", set_batch_size)
    
    start = time.time()
    for index, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train[:,0])):
        print ("Training on fold " + str(index + 1) + "/"+str(k)+"...") 
        xtrain, xval = X_train[train_indices], X_train[val_indices]
        ytrain, yval = y_train[train_indices], y_train[val_indices] 
        weights_train = weights[train_indices]
        weights_val = weights[val_indices]
        model = None
        print(xtrain.shape[1])
        print(xtrain.shape[2])
        print(xtrain.shape)
        print(ytrain.shape)
        model = make_GRU((xtrain.shape[1], xtrain.shape[2]), 2)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', auc, f1])

        history = model.fit(xtrain, ytrain, validation_data=(xval, yval, weights_val), epochs=set_number_of_epochs, batch_size=set_batch_size, sample_weight=weights_train)


        output = model.predict_classes(xval)

        print(confusion_matrix(yval.argmax(axis=1), output))

        print(classification_report(yval.argmax(axis=1), output))

    end = time.time()
    time_in_seconds =  end - start
    print("time (in seconds)", time_in_seconds)
    
    lastModel = model
    history_general = history.history
    
    print("... DONE!")
    
    return [lastModel , history_general, X_test, y_test, time_in_seconds]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    datasets = ['commons-bcel','commons-io','jhotdraw','junit4','pdfbox','wro4j']
    main_columns = ["project","commit","class",
                "cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
                "AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential",																					
                "BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF",																														
                "FANIN","FANOUT","LazyClass","DataClass","ComplexClass","SpaghettiCode","SpeculativeGenerality","GodClass","RefusedBequest","ClassDataShouldBePrivate","BrainClass","TotalClass","LongParameterList","LongMethod","FeatureEnvy","DispersedCoupling","MessageChain","IntensiveCoupling","ShotgunSurgery","BrainMethod","TotalMethod","TotalClassMethod","DiversityTotal","DiversityMethod","DiversityClass",																				
                "TOTAL_CHANGES","release","will_change"]
    for dataset in datasets:
        print("TEST DATASET - " + dataset)

        all_releases_df = pd.read_csv('datasets/' + dataset + '-all-releases.csv', usecols=main_columns)
        all_releases_df = all_releases_df.dropna(axis=1, how='all')
        print("Generating a new dataframe without containing the last release...")
        df = all_releases_df[all_releases_df['release'] != all_releases_df['release'].max()] 
        df.drop(columns=["project","commit","TOTAL_CHANGES","release","will_change"])
        listResults = {} 

        for i in [2, 3, 4, 5, 6]:
            print("window_size: "+str(i))
            listResults[i] = applyTensor(i, df)
            print("... DONE!")   
        for i in [2, 3, 4, 5, 6]:
            print("RESULTS FROM WINDOW SIZE",i)
            lastModel = listResults[i][0]
            history_general = listResults[i][1]
            X_test = listResults[i][2]
            y_test = listResults[i][3]
            time_in_seconds = listResults[i][4]

            fractions_t = 1-y_test.sum(axis=0)/len(y_test)
            weights_t = fractions_t[y_test.argmax(axis=1)]

            output = lastModel.predict_classes(X_test)
            print(confusion_matrix(y_test.argmax(axis=1), output))
            print(classification_report(y_test.argmax(axis=1), output))
            evaluation_metrics(y_test.argmax(axis=1), output, weights_t)
            print("Time in seconds:", time_in_seconds)
            
            print("\n")