import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
metrics = ['F1-Score(weighted)', 'Accuracy', 'Sensitivity', 'Specificity', 'ROC AUC score']
for metric in metrics:
    df = pd.read_csv('../datasets/resultadospormodelo.csv', decimal=",")
    #print(df.columns)
    print('######### ' + metric)

    # approach
    dfApp = df.query("model == 'model1'")
    dfApp['window'] = dfApp['window'].replace({0: 'trad', 3: 'our'})
    # dfAvg = dfApp.groupby(['Dataset', 'window'])[metric].mean()
    # print(dfAvg)
    #print(dfApp.head)
    ax = pd.pivot_table(dfApp, index='Dataset', columns='window')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(title='Approach', loc='lower right')
    plt.savefig('results/window-set1'+metric+'.pdf')
    # plt.show()

    # -- set 2
    dfApp = df.query("model == 'model2'")
    dfApp['window'] = dfApp['window'].replace({0: 'trad', 3: 'our'})
    # dfAvg = dfApp.groupby(['Dataset', 'window'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfApp, index='Dataset', columns='window')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(title='Approach', loc='lower right')
    plt.savefig('results/window-set2'+metric+'.pdf')

     # -- set 6
    dfApp = df.query("model == 'model6'")
    dfApp['window'] = dfApp['window'].replace({0: 'trad', 3: 'our'})
    # dfAvg = dfApp.groupby(['Dataset', 'window'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfApp, index='Dataset', columns='window')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(title='Approach', loc='lower right')
    plt.savefig('results/window-set6'+metric+'.pdf')

     # -- set 7
    dfApp = df.query("model == 'model7'")
    dfApp['window'] = dfApp['window'].replace({0: 'trad', 3: 'our'})
    # dfAvg = dfApp.groupby(['Dataset', 'window'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfApp, index='Dataset', columns='window')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(title='Approach', loc='lower right')
    plt.savefig('results/window-set7'+metric+'.pdf')


    dfApp['window'] = df['window'].replace({0: 'trad', 3: 'our'})
    # dfAvg = dfApp.groupby(['Dataset', 'window'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfApp, index='Dataset', columns='window')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(title='Approach', loc='lower right')
    plt.savefig('results/window-sets1_2_6_7'+metric+'.pdf')

    # feature sets
    dfSet = df.query("window == 3")
    dfSet['model'] = dfSet['model'].replace({'model1': 'AF', 'model2': 'StEv','model3': 'EvSm','model4': 'StSm','model6': 'St','model7': 'Ev'}
)
    dfAvg = dfSet.groupby(['model'])[metric].mean()
    print(dfAvg)
    dfMaxAll = dfSet.groupby(['Dataset','model'])[metric].mean()
    #print(dfMaxAll)
    ax = pd.pivot_table(dfSet, index='Dataset', columns='model')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/sets'+metric+'.pdf')

   
    
    #algorithms
    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model1'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set1'+metric+'.pdf')

    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model2'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set2'+metric+'.pdf')

    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model3'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set3'+metric+'.pdf')

    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model4'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set4'+metric+'.pdf')

    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model6'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set6'+metric+'.pdf')

    dfAlg = df.query("window == 3")
    dfAlg = dfAlg.query("model == 'model7'")
    # dfSet['window'] = dfSet['window'].replace({0: 'trad', 3: 'our'})
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-set7'+metric+'.pdf')



    dfAlg = df.query("window == 3")
    dfAvg = dfAlg.groupby(['Dataset', 'Algoritm'])[metric].mean()
    # print(dfAvg)

    ax = pd.pivot_table(dfAlg, index='Dataset', columns='Algoritm')[metric].plot.bar(alpha=0.75, rot=0)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    plt.legend(loc='lower right')
    plt.savefig('results/algs-sets1_2_3_4_5_6'+metric+'.pdf')


    #plt.show()

    
 