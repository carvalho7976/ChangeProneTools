import json
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



pathOrganic = "/Volumes/backup-geni/projects-smells/results/organic/openfire/repo/Openfire/"
organicRepo = pydriller.Git(pathOrganic)
repo = git.Repo(pathOrganic)
tags = repo.tags
row = ["projectName","commitNumber","fullyQualifiedName",
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
csvPath = "/Volumes/backup-geni/projects-smells/results/organic/" + "openfire.csv"
f = open(csvPath, "w")
writer = csv.writer(f)
smells_template = {
        #CLASSE
        "LazyClass": 0,
        "DataClass": 0,
        "ComplexClass": 0,
        "SpaghettiCode": 0,
        "SpeculativeGenerality": 0,
        "GodClass": 0,
        "RefusedBequest": 0,
        "ClassDataShouldBePrivate": 0,
        "BrainClass": 0,
        "TotalClass": 0,
        #METODO
        "LongParameterList": 0,
        "LongMethod": 0,
        "FeatureEnvy": 0,
        "DispersedCoupling": 0,
        "MessageChain": 0,
        "IntensiveCoupling": 0,
        "ShotgunSurgery": 0,
        "BrainMethod": 0,
        "TotalMethod": 0,
        #class + method
        "TotalClassMethod": 0,
        "DiversityTotal": 0,
        "DiversityMethod": 0,
        "DiversityClass" : 0
    }  
writer.writerow(row)
for tag in tags:
    hashCurrent = organicRepo.get_commit_from_tag(tag.name).hash
    try:
        # Carrega o arquivo do organic
        file = open("/Volumes/backup-geni/projects-smells/results/organic/openfire/openfire-all/" + hashCurrent + "-openfire.json")
        smells_classes = json.load(file)
        for _class in smells_classes:
         smells = smells_template.copy()
         for _smell in  _class['smells']:
             smells[_smell['name']] = smells[_smell['name']] + 1
             smells["TotalClass"] = smells["TotalClass"] + 1
             if(smells[_smell['name']] == 1):
                smells["DiversityClass"] = smells["DiversityClass"] + 1

         for _method in _class['methods']:
             for _smell in _method['smells']:
                smells[_smell['name']] = smells[_smell['name']] + 1
                smells["TotalMethod"] = smells["TotalMethod"] + 1
                if smells[_smell['name']] == 1:
                    smells["DiversityMethod"] = smells["DiversityMethod"] + 1
               #   writer.writerow(_smell['name'])
         projectName = "commons-bcel"
         className = _class['fullyQualifiedName']
         commitVersion = hashCurrent
         smells["TotalClassMethod"] = smells["TotalMethod"] + smells["TotalClass"]
         smells["DiversityTotal"] = smells["DiversityMethod"] +  smells["DiversityClass"]
         _row = [projectName,commitVersion,className]
         metricsValuesArray = _class["metricsValues"]

         for m in metricsValuesArray:
            _row.append(metricsValuesArray[m])
         for s in smells:
             _row.append(smells[s])
            
         writer.writerow(_row)
                   

    except Exception as e: 
         print(hashCurrent)
         #print(e)
f.close()

    