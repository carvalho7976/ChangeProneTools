from asyncore import write
from importlib.resources import path
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git
import pandas as pd
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Join Metrics')
    #args = ap.parse_args()

    #folder with repo: projectA and projectB
    pathCK = "/Volumes/backup-geni/projects-smells/results/ck/junit4/repo/junit4"
    csvPathCK = "/Volumes/backup-geni/projects-smells/results/ck/junit4/junit4-all/"
    
    csvPathUndestand = "/Volumes/backup-geni/projects-smells/results/understand/junit4/" 
    csvPathProcessMetrics = "/Volumes/backup-geni/projects-smells/results/processMetrics/junit4-results-processMetrics.csv"
    csvPathChangeDistiller = "/Volumes/backup-geni/projects-smells/results/ChangeDistiller/junit4-results.csv"
    csvOrganic =  "/Volumes/backup-geni/projects-smells/results/organic/junit4.csv"
    csvResults = "/Volumes/backup-geni/projects-smells/results/junit4-all-releases.csv"

    ckRepo = pydriller.Git(pathCK)
    #understandRepo = pydriller.Git(csvPathUndestand)
    repo = git.Repo(pathCK)
    tags = repo.tags
    release = 1
    #REMOVED FROM CK - "file"
    ckClassMetricsAll = ["class","type","cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty"]

    understandMetrics = ["Kind","Name","File","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","CountClassBase","CountClassCoupled","CountClassDerived","CountDeclClass","CountDeclClassMethod","CountDeclClassVariable","CountDeclFile","CountDeclFunction","CountDeclInstanceMethod","CountDeclInstanceVariable","CountDeclMethod","CountDeclMethodAll","CountDeclMethodDefault","CountDeclMethodPrivate","CountDeclMethodProtected","CountDeclMethodPublic","CountInput","CountLine","CountLineBlank","CountLineCode","CountLineCodeDecl","CountLineCodeExe","CountLineComment","CountOutput","CountPath","CountSemicolon","CountStmt","CountStmtDecl","CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential"]

    #rename "class" to "className", remove "release"
    processMetrics = ["project","commit","commitprevious","className","BOC","TACH","FCH","LCH","CHO","FRCH","CHD","WCD","WFR","ATAF","LCA","LCD","CSB","CSBS","ACDF"]

    chageDistillerMetrics = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT","CLASS_PREVIOUSCOMMIT",
		    		"STATEMENT_DELETE", "STATEMENT_INSERT", "STATEMENT_ORDERING_CHANGE","STATEMENT_PARENT_CHANGE","STATEMENT_UPDATE","TOTAL_STATEMENTLEVELCHANGES",
		    		"PARENT_CLASS_CHANGE", "PARENT_CLASS_DELETE", "PARENT_CLASS_INSERT","CLASS_RENAMING","TOTAL_CLASSDECLARATIONCHANGES",
		    		"RETURN_TYPE_CHANGE","RETURN_TYPE_DELETE","RETURN_TYPE_INSERT","METHOD_RENAMING","PARAMETER_DELETE","PARAMETER_INSERT","PARAMETER_ORDERING_CHANGE","PARAMETER_RENAMING","PARAMETER_TYPE_CHANGE","TOTAL_METHODDECLARATIONSCHANGES",
		    		"ATTRIBUTE_RENAMING","ATTRIBUTE_TYPE_CHANGE","TOTAL_ATTRIBUTEDECLARATIONCHANGES",
		    		"ADDING_ATTRIBUTE_MODIFIABILITY","REMOVING_ATTRIBUTE_MODIFIABILITY","REMOVING_CLASS_DERIVABILITY","REMOVING_METHOD_OVERRIDABILITY","ADDING_CLASS_DERIVABILITY","ADDING_CLASS_DERIVABILITY","ADDING_METHOD_OVERRIDABILITY", "TOTAL_DECLARATIONPARTCHANGES","TOTAL_CHANGES"]
    organicMetrics = ["projectName","commitNumber","fullyQualifiedName",
                    "PublicFieldCount","IsAbstract","ClassLinesOfCode","WeighOfClass",
                    "FANIN","TightClassCohesion","FANOUT","OverrideRatio","LCOM3",
                    "WeightedMethodCount","LCOM2","NumberOfAccessorMethods",
                    'LazyClass', 'DataClass', 'ComplexClass', 'SpaghettiCode', 
                    'SpeculativeGenerality', 'GodClass', 'RefusedBequest', 
                    'ClassDataShouldBePrivate', 'BrainClass', 'TotalClass',
                    'LongParameterList', 'LongMethod', 'FeatureEnvy',
                        'DispersedCoupling', 'MessageChain', 'IntensiveCoupling', 
                        'ShotgunSurgery', 'BrainMethod', 'TotalMethod', 'TotalClassMethod',
                         "DiversityTotal","DiversityMethod","DiversityClass"]
    #f = open(csvPath, "w")
    #writer = csv.writer(f)  
    missing = []
    for tag in tags:
        hashCurrent = ckRepo.get_commit_from_tag(tag.name).hash
      
        try:
            
            
            releaseUnderstand = pd.read_csv(csvPathUndestand + hashCurrent+'.csv', usecols=understandMetrics, sep=',',engine='python', index_col=False)

            print("Understand ")
            print(releaseUnderstand.shape[0])

            releaseCK = pd.read_csv(csvPathCK + hashCurrent+'-class.csv', usecols=ckClassMetricsAll, sep=',', index_col=False)

            print("CK ")
            print(releaseCK.shape[0])

           
            releaseOrganicMetrics = pd.read_csv(csvOrganic, usecols=organicMetrics, sep=',',engine='python', index_col=False)
            releaseOrganicMetrics = releaseOrganicMetrics[(releaseOrganicMetrics['commitNumber'] == hashCurrent)]
            
            print("Organic ")
            print(releaseOrganicMetrics.shape[0])

            releaseChangeDistillerMetrics = pd.read_csv(csvPathChangeDistiller, usecols=chageDistillerMetrics, sep=',',engine='python', index_col=False)
            releaseChangeDistillerMetrics = releaseChangeDistillerMetrics[(releaseChangeDistillerMetrics['CURRENT_COMMIT'] == hashCurrent)]
           

            print("Change distiller ")
            print(releaseChangeDistillerMetrics.shape[0])

            releaseProcessMetrics = pd.read_csv(csvPathProcessMetrics, usecols=processMetrics, sep=',', engine='python', index_col=False)
            releaseProcessMetrics = releaseProcessMetrics[(releaseProcessMetrics['commit'] == hashCurrent)]
            
            print("Process ")
            print(releaseProcessMetrics.shape[0])

           
            #para cada release procurar as classes correspondentes e agregar em um só dataframe se "name" = "class"
            ck_understand = pd.merge(left=releaseCK, right=releaseUnderstand, left_on='class', right_on='Name')
            ck_understand_process = pd.merge(left=ck_understand, right=releaseProcessMetrics, left_on='class', right_on='className')
            ck_understand_process_organic = pd.merge(left=ck_understand_process, right=releaseOrganicMetrics, left_on='class', right_on='fullyQualifiedName')
            
            merged_full = pd.merge(left=ck_understand_process_organic, right=releaseChangeDistillerMetrics, left_on='class', right_on='CLASS_PREVIOUSCOMMIT')
            
            #merged_full.loc[:,'class_frequency'] = 1
            merged_full.loc[:,'will_change'] = 0
            #merged_full.loc[:,'number_of_changes'] = 0
            merged_full.loc[:,'release'] = release
            medianChanges = merged_full['TOTAL_CHANGES'].median()
            merged_full['will_change'] = np.where(merged_full['TOTAL_CHANGES'] > medianChanges, 1,0)
            if(release == 1):
                merged_full.to_csv(csvResults, index=False)
            else:
                 merged_full.to_csv(csvResults,mode="a", header=False, index=False)

            release += 1
        except Exception as e: 
            print(e)
           # print(hashCurrent)
            missing.append(hashCurrent)
      
    print(missing)
        