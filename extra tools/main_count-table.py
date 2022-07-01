import pandas as pd
import collections

from sqlalchemy import true
databases = ['commons-bcel','commons-io','junit4','pdfbox','wro4j']
algorithms = ['DT','LogistRegression','MLP','RandomForest']
sets = ['model1','model2','model3','model4','model6','model7']
results = {'model1-our-f1': 0, 'model2-our-f1': 0, 'model3-our-f1': 0, 'model4-our-f1': 0, 'model6-our-f1': 0, 'model7-our-f1': 0, 'model4-trad-f1': 0, 'model6-trad-f1': 0, 'model1-trad-f1': 0, 'model2-trad-f1': 0, 'model3-trad-f1': 0, 'model4-eq-f1': 0, 'model6-eq-f1': 0, 'model7-trad-f1': 0, 'model1-our-acc': 0, 'model1-our-sen': 0, 'model1-our-auc': 0, 'model2-our-acc': 0, 'model2-trad-sen': 0, 'model2-our-auc': 0, 'model3-our-acc': 0, 'model3-trad-sen': 0, 'model3-trad-auc': 0, 'model4-our-acc': 0, 'model4-trad-sen': 0, 'model4-our-auc': 0, 'model6-our-acc': 0, 'model6-our-sen': 0, 'model6-our-auc': 0, 'model7-our-acc': 0, 'model7-trad-sen': 0, 'model7-trad-auc': 0, 'model2-eq-acc': 0, 'model2-our-sen': 0, 'model3-our-sen': 0, 'model3-our-auc': 0, 'model4-trad-acc': 0, 'model4-our-sen': 0, 'model6-trad-acc': 0, 'model7-our-sen': 0, 'model7-our-auc': 0, 'model1-eq-sen': 0, 'model2-eq-auc': 0, 'model4-trad-auc': 0, 'model6-trad-sen': 0, 'model6-trad-auc': 0, 'model1-trad-acc': 0, 'model1-trad-sen': 0, 'model1-trad-auc': 0, 'model2-trad-acc': 0, 'model2-trad-auc': 0, 'model3-trad-acc': 0, 'model4-eq-acc': 0, 'model6-eq-acc': 0, 'model6-eq-sen': 0, 'model6-eq-auc': 0, 'model7-trad-acc': 0, 'model2-eq-sen': 0, 'model1-eq-auc': 0, 'model4-eq-auc': 0}
df = pd.read_csv('charts/resultadospormodelo.csv', decimal=",")
cols = ['F1-Score(weighted)','Accuracy','Sensitivity','ROC AUC score']
df = df.round(decimals = 2)

df = df.sort_values('window', ascending=False)
#count SETS
for db in databases:
    for al in algorithms:
        for set in sets:
             filtered = df[(df['Dataset'] == db) & (df['Algoritm']== al) &  (df['model']== set) ]
             
             #count f1
             f1Our = filtered.iloc[0, filtered.columns.get_loc("F1-Score(weighted)")]
             f1Trad = filtered.iloc[1, filtered.columns.get_loc("F1-Score(weighted)")]
             if(f1Our > f1Trad):
                 results[set+'-our-f1'] += 1
             elif(f1Our < f1Trad):
                  results[set+'-trad-f1'] += 1
             else:
                  results[set+'-eq-f1'] += 1
            #count acc
             accOur = filtered.iloc[0, filtered.columns.get_loc('Accuracy')]
             accTrad = filtered.iloc[1, filtered.columns.get_loc('Accuracy')]
             if(accOur > accTrad):
                 results[set+'-our-acc'] += 1
             elif(accOur < accTrad):
                  results[set+'-trad-acc'] += 1
             else:
                  results[set+'-eq-acc'] += 1
            #count Sensitivity
             senOur = filtered.iloc[0, filtered.columns.get_loc('Sensitivity')]
             senTrad = filtered.iloc[1, filtered.columns.get_loc('Sensitivity')]
             if(senOur > senTrad):
                 results[set+'-our-sen'] += 1
             elif(senOur < senTrad):
                  results[set+'-trad-sen'] += 1
             else:
                  results[set+'-eq-sen'] += 1
            #count ROC AUC score
             aucOur = filtered.iloc[0, filtered.columns.get_loc('ROC AUC score')]
             aucTrad = filtered.iloc[1, filtered.columns.get_loc('ROC AUC score')]
             if(aucOur > aucTrad):
                 results[set+'-our-auc'] += 1
             elif(aucOur < aucTrad):
                  results[set+'-trad-auc'] += 1
             else:
                  results[set+'-eq-auc'] += 1    

results = collections.OrderedDict(sorted(results.items()))              

for k,v in results.items():
    print(k + ":" + str(v))

#COMPARE SMELLS
sets = ['model1','model2','model3','model4','model6','model7']
smells = {'eq-f1': 0, 'eq-acc': 0, 'SemSmell-sen': 0, 'eq-auc': 0, 'ComSmell-acc': 0, 'eq-sen': 0, 'ComSmell-sen': 0, 'ComSmell-auc': 0,'SemSmell-auc': 0,  'SemSmell-acc': 0, 'ComSmell-f1': 0, 'SemSmell-f1': 0}
df = df.sort_values('model', ascending=true)
for db in databases:
    for al in algorithms:
            filtered = df[(df['Dataset'] == db) & (df['Algoritm']== al)  &  (df['window']== 3) & ((df['model']== 'model4') | (df['model']== 'model6') ) ]
              #count f1
            print(filtered.head(2))
            f1ComSmell = filtered.iloc[0, filtered.columns.get_loc("F1-Score(weighted)")]
            f1SemSmell = filtered.iloc[1, filtered.columns.get_loc("F1-Score(weighted)")]
            if(f1ComSmell > f1SemSmell):
                smells['ComSmell-f1'] += 1
            elif(f1ComSmell < f1SemSmell):
                smells['SemSmell-f1'] += 1
            else:
                smells['eq-f1'] += 1
        #count acc
            accComSmell = filtered.iloc[0, filtered.columns.get_loc('Accuracy')]
            accSemSmell = filtered.iloc[1, filtered.columns.get_loc('Accuracy')]
            if(accComSmell > accSemSmell):
                smells['ComSmell-acc'] += 1
            elif(accComSmell < accSemSmell):
                smells['SemSmell-acc'] += 1
            else:
                smells['eq-acc'] += 1
        #count Sensitivity
            senComSmell = filtered.iloc[0, filtered.columns.get_loc('Sensitivity')]
            senSemSmell = filtered.iloc[1, filtered.columns.get_loc('Sensitivity')]
            if(senComSmell > senSemSmell):
                smells['ComSmell-sen'] += 1
            elif(senComSmell < senSemSmell):
                smells['SemSmell-sen'] += 1
            else:
                smells['eq-sen'] += 1
            #count ROC AUC score
            aucComSmell = filtered.iloc[0, filtered.columns.get_loc('ROC AUC score')]
            aucSemSmell = filtered.iloc[1, filtered.columns.get_loc('ROC AUC score')]
            if(aucComSmell > aucSemSmell):
                smells['ComSmell-auc'] += 1
            elif(aucComSmell < aucSemSmell):
                smells['SemSmell-auc'] += 1
            else:
                smells['eq-auc'] += 1    

print("######################### smells ###############")
smells = collections.OrderedDict(sorted(smells.items()))           
for k,v in smells.items():
    print(k + ":" + str(v))

#COMPARE RESAMPLES
sets = ['model1','model2','model3','model4','model6','model7']
resamples = {'ADA-DT-3': 0, 'ADA-DT-0': 0, 'ROS-DT-3': 0, 'ROS-DT-0': 0, 'SMOTE-DT-3': 0, 'SMOTE-DT-0': 0, 'NONE-DT-3': 0, 'NONE-DT-0': 0, 'ADA-LogistRegression-3': 0, 'ADA-LogistRegression-0': 0, 'ROS-LogistRegression-3': 0, 'ROS-LogistRegression-0': 0, 'SMOTE-LogistRegression-3': 0, 'SMOTE-LogistRegression-0': 0, 'NONE-LogistRegression-3': 0, 'NONE-LogistRegression-0': 0, 'ADA-MLP-3': 0, 'ADA-MLP-0': 0, 'ROS-MLP-3': 0, 'ROS-MLP-0': 0, 'SMOTE-MLP-3': 0, 'SMOTE-MLP-0': 0, 'NONE-MLP-3': 0, 'NONE-MLP-0': 0, 'ADA-RandomForest-3': 0, 'ADA-RandomForest-0': 0, 'ROS-RandomForest-3': 0, 'ROS-RandomForest-0': 0, 'SMOTE-RandomForest-3': 0, 'SMOTE-RandomForest-0': 0, 'NONE-RandomForest-3': 0, 'NONE-RandomForest-0': 0}
resamplesNames = ['ADA','ROS','SMOTE','NONE']
df = df.sort_values('model', ascending=true)
windows = [3,0]
for db in databases:
    for al in algorithms:
        for set in sets:
            for w in windows:
                filtered = df[(df['Dataset'] == db) & (df['Algoritm']== al)  &  (df['window']== w) & (df['model']== set)  ]
                resamples[filtered.iloc[0, filtered.columns.get_loc('resample')] + '-'+al+'-'+str(w)] += 1
           
print(resamples)
print("######################### RESAMPLES ###############")
resamples = collections.OrderedDict(sorted(resamples.items()))           
for k,v in resamples.items():
    print(k + ":" + str(v))
            