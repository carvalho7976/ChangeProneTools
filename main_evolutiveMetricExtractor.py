from asyncore import write
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract process metrics')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--pathB', required=True)
    ap.add_argument('--commits', required=True, help='csv with list of commits to compare commitA and commitB')
    ap.add_argument('--projectName', required=True)
    ap.add_argument('--absolutePath', required=True)
    ap.add_argument('--mode', required=True,help='mode - tag for commits with tag, csv - for csv of commits')
    args = ap.parse_args()

    #folder with repo: projectA and projectB

    pathA = pydriller.Git(args.pathA)
    pathB = pydriller.Git(args.pathB)
    repo = git.Repo(args.pathA)
    tags = repo.tags
   
    release = 1
    commit_A = ''
    commit_B = ''
    bocArray = {}
    fchArray = {}
    frchArray = {}
    wcdArray = {}
    wfrArray = {}
    lcaArray = {}
    lcdArray = {}
    csbArray = {}
    csbsArray = {}
    acdfArray = {}
    if(args.mode == 'tag'):
        csvPath = args.absolutePath +"results/" +args.projectName + "-results-processMetrics.csv"
        f = open(csvPath, "w")
        writer = csv.writer(f)   
        for tag in tags:
           
            hashCurrent = pathB.get_commit_from_tag(tag.name).hash
            pathA.checkout(hashCurrent)
            if(commit_B == ''):
                hashPrevious = None
            filesA = pathA.files()
            filesA = [x for x in filesA if x.endswith('.java')]
            
           
                
            if(release ==1):
                commit_A = tag
                row = ['project', 'commit', 'commitprevious', 'class','release','BOC','TACH','FCH', 'LCH','CHO','FRCH','CHD','WCD' ,'WFR','ATAF','LCA','LCD','CSB','CSBS','ACDF']
                for file in filesA:
                    if(file not in bocArray):
                        bocArray[file] = release
                        fchArray[file] = 0
                writer.writerow(row)
            else:
                project = args.projectName
                commit = hashCurrent
                commitprevious = hashPrevious
                boc = release
                tach = 0
                fch = 0
                lch = release
                cho = 0
                frch = 0
                chd = 0
                wcd = 0
                wfr = 0
                ataf = 0
                lca = 0
                lcd = 0
                csb = 0
                csbs = 0
                acdf = 0
                hashPrevious = pathA.get_commit_from_tag(commit_A.name).hash
                pathB.checkout(hashPrevious)
                filesB = pathB.files()
                filesB = [x for x in filesB if x.endswith('.java')]
                for file in filesA:
                    if(file not in bocArray):
                        bocArray[file] = release
                        boc = release
                    else:
                        boc = bocArray.get(file)
                    if(file not in fchArray):
                        fchArray[file] = 0
                    if(file not in frchArray):
                        frchArray[file] = 0
                    if(file not in wcdArray):
                        wcdArray[file] = 0
                    if(file not in wfrArray):
                        wfrArray[file] = 0
                    if(file not in lcaArray):
                        lcaArray[file] = 0
                    if(file not in lcdArray):
                        lcdArray[file] = 0
                    if(file not in csbArray):
                        csbArray[file] = 0
                    if(file not in csbsArray):
                        csbsArray[file] = 0
                    if(file not in  acdfArray):
                        acdfArray[file] = 0
                    #get all commits from release n-1 to n, the goal is to find the total amount of changes on a file
                    commits_touching_path = Repository(args.pathA, from_commit=hashPrevious, to_commit=hashCurrent).traverse_commits()
                    file_temp = file.replace(args.absolutePath+"projectA/"+args.projectName+"/", '')
                    added_lines = 0
                    removed_lines =0
                    loc = 0
                    for cc in commits_touching_path:
                        modifiedFiles = [x for x in cc.modified_files if x.filename.endswith('.java')]
                        for m in modifiedFiles:
                            if(m.change_type.name == 'ADD' and (m.new_path == file_temp or m.old_path == file_temp)):
                                #size of 
                                csbsArray[file] = m.nloc
                            if(m.change_type.name == 'MODIFY' and (m.new_path == file_temp or m.old_path == file_temp)):
                                loc = m.nloc
                                added_lines += m.added_lines
                                removed_lines += m.deleted_lines
                                #first time change
                                if( fchArray[file] == 0):
                                    fchArray[file] = release
                                    fch = release
                                #last time change, the lastest released analayzed
                                lhc = release
                                #if changes have occurred
                                cho = 1
                                #frequency of change
                                frchArray[file] += 1
                                frch = frchArray[file]

                    #total amount change, added lines + deleted lines (changed lines are already counted twice )
                    tach = added_lines + removed_lines
                    if(tach > 0):
                        chd = tach/loc
                        #cumulative weight of change
                        wcdArray[file] += tach * pow(2,boc-release)
                        #sum of change density, to normalize later
                        acdfArray[file] += chd
                        #agregate change size, normalized by frequency change
                        if(frch > 0):
                            ataf = tach/frch
                            #agregate change density normalized by frch
                            acdf = acdfArray[file]/frch
                        #last amount of change
                        lcaArray[file] = tach
                        #last change density
                        lcdArray[file] = chd
                        csbArray[file] += tach
                        
                    wcd =  wcdArray[file]
                    wch = wcd * pow(2,boc-release)
                    #cumultive weight frequecy
                    wfrArray[file] += (release - 1) *  cho
                    wfr = wfrArray[file]
                    lca = lcaArray[file]
                    lcd = lcdArray[file]
                    csb = csbArray[file]
                    if(csb > 0 ):
                        csbs = csbsArray[file]/csb
                    
                    row = [args.projectName, hashCurrent, hashPrevious, file,release,boc,tach,fch, lch,cho,frch,chd,wch ,wfr,ataf,lca,lcd,csb,csbs,acdf]
                    writer.writerow(row)

            commit_A = tag
            release +=1
        f.close()