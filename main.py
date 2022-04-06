from asyncore import write
from importlib.resources import path
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Join Metrics')
    #args = ap.parse_args()

    #folder with repo: projectA and projectB
    pathCK = "/Volumes/backup-geni/projects-smells/results/ck/commons-bcel/repo/commons-bcel"
    csvPathCK = "/Volumes/backup-geni/projects-smells/results/ck/commons-bcel/commons-bcel-all/"

    csvPathUndestand = "/Volumes/backup-geni/projects-smells/results/understand/commons-bcel/"
   
    ckRepo = pydriller.Git(pathCK)
    understandRepo = pydriller.Git(csvPathUndestand)
    repo = git.Repo(pathCK)
    tags = repo.tags
    csvResults = "/Volumes/backup-geni/projects-smells/results/commons-bcel-all-releases.csv"
    #/Volumes/backup-geni/projects-smells/results/ck/commons-bcel/repo/commons-bcel
    release = 1
    #REMOVED FROM CK - "file"
    ckClassMetricsAll = ["class","type","cbo","wmc","dit","rfc","lcom","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","visibleFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty"]

    understandMetrics = ["Kind","Name","File","AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine","AvgLineBlank","AvgLineCode","AvgLineComment","CountClassBase","CountClassCoupled","CountClassDerived","CountDeclClass","CountDeclClassMethod","CountDeclClassVariable","CountDeclFile","CountDeclFunction","CountDeclInstanceMethod","CountDeclInstanceVariable","CountDeclMethod","CountDeclMethodAll","CountDeclMethodDefault","CountDeclMethodPrivate","CountDeclMethodProtected","CountDeclMethodPublic","CountInput","CountLine","CountLineBlank","CountLineCode","CountLineCodeDecl","CountLineCodeExe","CountLineComment","CountOutput","CountPath","CountSemicolon","CountStmt","CountStmtDecl","CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential","MaxCyclomatic","MaxCyclomaticModified","MaxCyclomaticStrict","MaxEssential","MaxInheritanceTree","MaxNesting","PercentLackOfCohesion","RatioCommentToCode","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential"]

    #f = open(csvPath, "w")
    #writer = csv.writer(f)  
    missing = []
    for tag in tags:
        hashCurrent = ckRepo.get_commit_from_tag(tag.name).hash
      
        try:
            releaseCK = pd.read_csv(csvPathCK + hashCurrent+'-class.csv', usecols=ckClassMetricsAll, sep=',', index_col=False)
           
            releaseUnderstand = pd.read_csv(csvPathUndestand + hashCurrent+'.csv', usecols=understandMetrics, sep=',', index_col=False)
            #para cada release procurar as classes correspondentes e agregar em um só dataframe se "name" = "class"
            merged_inner = pd.merge(left=releaseCK, right=releaseUnderstand, left_on='class', right_on='Name')
            merged_inner.loc[:,'class_frequency'] = 1
            merged_inner.loc[:,'will_change'] = 0
            merged_inner.loc[:,'number_of_changes'] = 0
            merged_inner.loc[:,'release'] = release
            if(release == 1):
                merged_inner.to_csv(csvResults, index=False)
            else:
                 merged_inner.to_csv(csvResults,mode="a", header=False, index=False)

            release += 1
        except:
            print(hashCurrent)
      
    print(missing)
        