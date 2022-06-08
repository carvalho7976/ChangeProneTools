import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

feature_names = ["TACH","WFR","FRCH","FANOUT","FANIN","CHD","ACDF","ATAF","WCD"]

   

main_columns = ["project","commit","class",
                "cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
                "AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential",																					
                "BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF",																														
                "FANIN","FANOUT","LazyClass","DataClass","ComplexClass","SpaghettiCode","SpeculativeGenerality","GodClass","RefusedBequest","ClassDataShouldBePrivate","BrainClass","TotalClass","LongParameterList","LongMethod","FeatureEnvy","DispersedCoupling","MessageChain","IntensiveCoupling","ShotgunSurgery","BrainMethod","TotalMethod","TotalClassMethod","DiversityTotal","DiversityMethod","DiversityClass",																				
                "TOTAL_CHANGES","release","will_change"]


databases = ['commons-bcel','commons-io','junit4','pdfbox','wro4j']

for db in databases:

    all_releases_df = pd.read_csv(
                        'datasets/'+db+'-all-releases.csv', usecols=main_columns)                 
    all_releases_df = all_releases_df.fillna(0)

    x = all_releases_df.loc[:, feature_names].values
    y = all_releases_df.loc[:,['will_change']].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, all_releases_df[['will_change']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('PCA ' + db, fontsize = 20)

    targets = [0,1]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['will_change'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('results/pca-'+db+'.pdf')


