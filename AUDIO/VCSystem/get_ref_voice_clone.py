import glob
import shutil
import os 

Test_ref = glob.glob("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_test/*.wav")

Test_ref = [x.replace("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_test/","") for x in Test_ref]

Train_ref = glob.glob("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_train/*.wav")

Train_ref = [x.replace("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_train/","") for x in Train_ref]

DictTest = {}
DictTrain = {}

with open("/home/trinhan/AILAB/VCSystem/DATA/VoiceCloneMCDtrain.txt", "r", encoding="utf-8") as ftrain:
    lines = ftrain.read().splitlines()
    for idx,line in enumerate(lines):
        try:
            a,b,c = line.split("\t")
        except:
            pass
        DictTrain[str(idx-1)+".wav"] = (a,b,c)

        #print(str(idx-1)+ " | " + b +  " | "+c )

with open("/home/trinhan/AILAB/VCSystem/DATA/VoiceCloneMCDtest.txt", "r", encoding="utf-8") as ftest:
    lines = ftest.read().splitlines()
    for idx,line in enumerate(lines):
        try:
            a,b,c = line.split("\t")
        except:
            pass
        DictTest[str(idx-1)+".wav"] = (a,b,c)

        #print(str(idx-1)+ " | " + b +  " | "+c )

cout_test = 0
cout_train = 0
for test in Test_ref:
    cout_test = cout_test +1
    print(test,DictTest[test])

    os.rename("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_test/"+test,"/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_test/"+DictTest[test][1]+"-"+DictTest[test][2]+".wav")

    #shutil.copy2("/home/trinhan/AILAB/VoiceClone/DATA/VIVOS/vivos/test/waves/"+DictTest[test][1].split("_")[0]+"/"+DictTest[test][1]+".wav", "/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/Ref_test/")
    #shutil.copy2("/home/trinhan/AILAB/VoiceClone/DATA/VIVOS/vivos/test/waves/"+DictTest[test][2].split("_")[0]+"/"+DictTest[test][2]+".wav", "/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/Ground_test/")

print("======")

for train in Train_ref:
    cout_train = cout_train +1
    print(train,DictTrain[train])

    os.rename("/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_train/"+train,"/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/VoiceClone_train/"+DictTrain[train][1]+"-"+DictTrain[train][2]+".wav")

    #shutil.copy2("/home/trinhan/AILAB/VoiceClone/DATA/VIVOS/vivos/train/waves/"+DictTrain[train][1].split("_")[0]+"/"+DictTrain[train][1]+".wav", "/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/Ref_train/")
    #shutil.copy2("/home/trinhan/AILAB/VoiceClone/DATA/VIVOS/vivos/train/waves/"+DictTrain[train][2].split("_")[0]+"/"+DictTrain[train][2]+".wav", "/home/trinhan/AILAB/VCSystem/AUDIO/VCSystem/Ground_train/")


print(cout_test)
print(cout_train)