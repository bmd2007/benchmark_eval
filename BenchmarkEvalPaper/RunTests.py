import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
currentDir = currentdir + '/'

import PPIPUtils
from Methods.Tian2019SVM.Tian2019SVM import Tian2019SVM
from Methods.Guo2008.GuoSVM import GuoSVM
from Methods.Li2020DeepEnsemble.LiDeepNetwork import LiDeepNetworkModule
from Methods.Sun2017AutoEncoderNetwork.SunStackAutoEncoder import SunStackAutoEncoderAC
from Methods.Sun2017AutoEncoderNetwork.SunStackAutoEncoder import SunStackAutoEncoderCT
from Methods.Chen2019RNNNetwork.ChenNetwork import ChenNetworkModule
from Methods.RichouxDeepNetwork.RichouxDeepNetwork import RichouxNetworkModuleLSTM
from Methods.RichouxDeepNetwork.RichouxDeepNetwork import RichouxNetworkModuleFULL
from Methods.Li2018DeepNetwork.Li2018DeepNetwork import Li2018DeepNetworkModule
from Methods.Czibula2021AutoPPI.Czibula2021AutoPPI import Czibula2021AutoPPIModule
from Methods.Czibula2021AutoPPI.Czibula2021AutoPPI import Czibula2021AutoPPIModuleSS
from Methods.Czibula2021AutoPPI.Czibula2021AutoPPI import Czibula2021AutoPPIModuleJJ
from Methods.Czibula2021AutoPPI.Czibula2021AutoPPI import Czibula2021AutoPPIModuleSJ
from Methods.Zhang2019DeepEnsemble.Zhang2019DeepEnsemble import ZhangDeepModule
from Methods.Yao2019DeepNetwork.Yao2019DeepNetwork import Yao2019NetworkModule
from Methods.Zhou2011SVM.ZhouSVM import ZhouSVM
from Methods.GonzalezLopez2019DeepNetwork.GonzalezLopez2019DeepNetwork import GonzalezLopez2019Module
from Methods.Zhao2012SVM.Zhao2012SVM import Zhao2012SVM
from Methods.Hashemifar2018DeepNetwork.Hashemifar2018DeepNetwork import Hashemifar2018DeepNetworkModule
from Methods.Goktepe2018SVM.Goktepe2018SVM import Goktepe2018SVM
from Methods.Pan2010.Pan2010 import Pan2010ModuleLDACTRANDFOREST, Pan2010ModuleLDACTROTFOREST, Pan2010ModuleLDACTSVM, Pan2010ModuleACRANDFOREST, Pan2010ModuleACROTFOREST, Pan2010ModuleACSVM,Pan2010ModulePSAACRANDFOREST,Pan2010ModulePSAACROTFOREST,Pan2010ModulePSAACSVM
from Methods.Du2017DeepNetwork.Du2017DeepNetwork import Du2017DeepNetworkModuleComb, Du2017DeepNetworkModuleSep
from Methods.Jia2015.Jia2015RF import Jia2015RFModule
from Methods.You2015RF.You2015RF import You2015RFModule
from Methods.Ding2016RF.Ding2016RF import Ding2016RFModule
from Methods.Wang2017RotF.Wang2017RotF import Wang2017RotFModule
from Methods.Chen2019LGBM.Chen2019LGBM import Chen2019LGBMModule
from Methods.Jia2019RF.Jia2019RF import Jia2019RFModule
from Methods.RandomNetwork.RandomNetwork import RandomNetworkModule
from Methods.RandomRF.RandomRF import RandomRFModule
from Methods.BiasModules.BasicBiasModuleGOSimSeqSim import BasicBiasModuleGOSimSeqSim
from Methods.BiasModules.BasicBiasModuleSeqSim import BasicBiasModuleSeqSim
from Methods.BiasModules.BasicBiasModule import BasicBiasModule
from Methods.MaetschkeVar2011.MaetschkeVar2011 import MaetschkeVar2011Module
from Methods.Chen2005RF.Chen2005RF import Chen2005RFModule
from Methods.GouVar2006GOLR.GouVar2006GOLR import GouVar2006GOLRModule
from Methods.ZhangDomainVar2016.ZhangDomainVar2016 import ZhangDomainVar2016AllModule, ZhangDomainVar2016NonTestModule, ZhangDomainVar2016HeldOutModule
from Methods.Zhang2016GO.Zhang2016GO import Zhang2016GOModule
from Methods.SimpleEnsemble.SimpleEnsemble import SimpleEnsembleAllModule, SimpleEnsembleNonTestModule, SimpleEnsembleHeldOutModule
	
import time
import numpy as np
from ProjectDataLoader import *
from PreProcessDatasets import createFeatures
from RunTrainTest import *

#algorithms
guo2008Test = True
li2020Test = True
sun2017Test = True
tian2019Test = True
Chen2019RNN = True
richouxANN = True
li2018Deep = True
Czibula2021AutoPPI = True
ZhangDeep2019 = True
YaoDeep2019 = True
zhou2011SVM = True
GonzalezLopez2019 = True
Zhao2012SVMTest = True
Hashemifar2018Test = True
Goktepe2018SVMTest = True
pan2010TestForests = True
pan2010TestSVMs = True
du2017DeepNetworkTest = True
jia2015RandomForestTest = True 
you2015RandomForestTest = True
ding2016RandomForestTest = True
wang2017RotFTest = True
chen2019LGBMTest = True
jia2019RandomForestTest = True
randomNetworkTest = True
randomRFTest = True
biasTests = True
MaetschkeVarTest = True
Chen2005RF = True
GouVar2006GOLRTest = True
ZhangDomainVar2016Test = True
Zhang2016GOTest = True
SimpleEnsembleTest = True
#data Types
orgData = True
HumanRandom50 = True
HumanRandom20 = True
HumanHeldOut50 = True
HumanHeldOut20 = True

baseResultsFolderName = 'results/'


	
	
	
	
	
#runs based on global variables
#can be toggled before calling function
def RunAll():	
	if guo2008Test:
		#create results folders if they do not exist
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Guo2008Results/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {'Model':'THUNDERSVM'}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(GuoSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(GuoSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTest(GuoSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(GuoSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTest(GuoSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if li2020Test:
		#create results folders if they do not exist
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName=  baseResultsFolderName+'Li2020Results/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {'fullGPU':True,'deviceType':'cuda'} 
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
			runTest(LiDeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
			
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(LiDeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTest(LiDeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(LiDeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTest(LiDeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)

	if tian2019Test:
		#create results folders if they do not exist
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'tian2019Results/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {'Model':'THUNDERSVM'}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTest(Tian2019SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if sun2017Test:
		PPIPUtils.makeDir(baseResultsFolderName)
		PPIPUtils.makeDir(baseResultsFolderName+'Sun2017Results/')
		for pair in [(SunStackAutoEncoderAC,'SunResults2017AC'),(SunStackAutoEncoderCT,'SunResults2017CT')]:
			resultsFolderName = baseResultsFolderName+'Sun2017Results/'+pair[1]+'/'
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {'fullGPU':True,'deviceType':'cuda'}
			if orgData:
				trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
				runTest(pair[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom50:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
				runTest(pair[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
					
			if HumanRandom20:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
				runTestLst(pair[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
				runTest(pair[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
				runTestLst(pair[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs,startIdx=13)
				

	if Chen2019RNN:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Chen2019Results/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
			runTest(ChenNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
		hyp = {'fullGPU':True,'schedPatience':1}
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(ChenNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTest(ChenNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(ChenNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTest(ChenNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)



	if richouxANN:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Richoux2019Results/'
		PPIPUtils.makeDir(resultsFolderName)
		resultsFolderName1=  resultsFolderName+'LSTM/'
		resultsFolderName2=  resultsFolderName+'FULL/'
		PPIPUtils.makeDir(resultsFolderName1)
		PPIPUtils.makeDir(resultsFolderName2)
		
		
		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadRichouxHumanDataStrict(resultsFolderName1)
			runTest(RichouxNetworkModuleLSTM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadRichouxHumanDataStrict(resultsFolderName2)
			runTest(RichouxNetworkModuleFULL, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName1,augment=True)
			runTest(RichouxNetworkModuleLSTM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName2,augment=True)
			runTest(RichouxNetworkModuleFULL, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName1,augment=True)
			runTestLst(RichouxNetworkModuleLSTM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName2,augment=True)
			runTestLst(RichouxNetworkModuleFULL, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName1,augment=True)
			runTest(RichouxNetworkModuleLSTM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName2,augment=True)
			runTest(RichouxNetworkModuleFULL, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName1,augment=True)
			runTestLst(RichouxNetworkModuleLSTM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName2,augment=True)
			runTestLst(RichouxNetworkModuleFULL, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if li2018Deep:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Li2018DeepResults/'
		PPIPUtils.makeDir(resultsFolderName)
		
		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
			runTest(Li2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Li2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Li2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Li2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Li2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if Czibula2021AutoPPI:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Czibula2021AutoPPI/'
		PPIPUtils.makeDir(resultsFolderName)
		resultsFolderNames = [resultsFolderName+'Czibula2021AutoPPISS/',resultsFolderName+'Czibula2021AutoPPISJ/',resultsFolderName+'Czibula2021AutoPPIJJ/']
		
		modelTypes = [Czibula2021AutoPPIModuleSS,Czibula2021AutoPPIModuleSJ,Czibula2021AutoPPIModuleJJ]
		for i in range(0,3):
			PPIPUtils.makeDir(resultsFolderNames[i])
		hyp = {'fullGPU':True}
		if orgData:
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderNames[i])
				runTest(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadGuoMultiSpeciesChen(resultsFolderNames[i])
				runTest(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
		if HumanRandom50:
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderNames[i])
				runTest(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				
		if HumanRandom20:
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderNames[i])
				runTestLst(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderNames[i])
				runTest(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			for i in range(0,3):
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderNames[i])
				runTestLst(modelTypes[i], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if ZhangDeep2019:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'ZhangDeep2019/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
			runTest(ZhangDeepModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)

		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(ZhangDeepModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(ZhangDeepModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(ZhangDeepModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(ZhangDeepModule, None,trainSets,testSets,folderName,hyp,saveModels=convertToFolder(saves),predictionsFLst = pfs)


	if YaoDeep2019:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'YaoDeep2019/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
			runTest(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Yao2019NetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)




	if zhou2011SVM:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'zhou2011SVMResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'Model':'THUNDERSVM'}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(ZhouSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(ZhouSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(ZhouSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(ZhouSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(ZhouSVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if GonzalezLopez2019:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'GonzalezLopez2019/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(GonzalezLopez2019Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

		
	if Zhao2012SVMTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'zhao2012SVMResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'Model':'THUNDERSVM'}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
			runTest(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadLiuFruitFly(resultsFolderName)
			runTest(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Zhao2012SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
			

		
	if Hashemifar2018Test:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Hashemifar2018DeepResults/'
		PPIPUtils.makeDir(resultsFolderName)
	   
		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Hashemifar2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Hashemifar2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Hashemifar2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Hashemifar2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Hashemifar2018DeepNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
	if Goktepe2018SVMTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Goktepe2018SVMResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'Model':'ThunderSVM'}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=5)
			runTest(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName,kfolds=5)
			runTest(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Goktepe2018SVM, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if pan2010TestForests:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Pan2010/'
		PPIPUtils.makeDir(resultsFolderName)

		pan2010TestForestModules = [Pan2010ModuleLDACTRANDFOREST, Pan2010ModuleLDACTROTFOREST, Pan2010ModuleACRANDFOREST, Pan2010ModuleACROTFOREST, Pan2010ModulePSAACRANDFOREST,Pan2010ModulePSAACROTFOREST]
		pan2010ResultsFolderNames = [resultsFolderName+'LDARand/',resultsFolderName+'LDARot/',resultsFolderName+'ACRand/',resultsFolderName+'ACRot/',resultsFolderName+'PSAACRand/',resultsFolderName+'PSAACRot']
		
		for i in range(0,len(pan2010TestForestModules)):
			modName = pan2010TestForestModules[i]
			resultsFolderName = pan2010ResultsFolderNames[i]
			PPIPUtils.makeDir(resultsFolderName)
			hyp={}
			if orgData:
				trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
				saves=None
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName,kfolds=5)
				saves=None
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName,kfolds=5)
				saves=None
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			hyp={}
			if HumanRandom50:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
				saves=None
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
				saves=None
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
				saves=None
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
				saves=None
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
	if pan2010TestSVMs:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Pan2010/'
		PPIPUtils.makeDir(resultsFolderName)

		PPIPUtils.makeDir('Results/')
		PPIPUtils.makeDir('Results/Pan2010/')
		pan2010TestSVMModules = [Pan2010ModuleLDACTSVM,  Pan2010ModuleACSVM, Pan2010ModulePSAACSVM]
		pan2010ResultsFolderNames = [baseResultsFolderName+'LDASVM/',baseResultsFolderName+'ACSVM/',baseResultsFolderName+'PSAACSVM/']
		
		for i in range(0,len(pan2010TestSVMModules)):
			modName = pan2010TestSVMModules[i]
			resultsFolderName = pan2010ResultsFolderNames[i]
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {'Model':'THUNDERSVM'}
			if orgData:
				trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName,kfolds=5)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName,kfolds=5)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
			if HumanRandom50:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			

	if du2017DeepNetworkTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Du2017/'
		PPIPUtils.makeDir(resultsFolderName)

		modLst = [Du2017DeepNetworkModuleSep, Du2017DeepNetworkModuleComb]
		resultFolders = [resultsFolderName+'Sep/',resultsFolderName+'Comb/']
		
		for i in range(0,len(modLst)):
			modName = modLst[i]
			resultsFolderName = resultFolders[i]
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {'fullGPU':True}
			if orgData:
				trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			
			if HumanRandom50:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
				runTest(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
				runTestLst(modName, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs,startIdx=2)




	if jia2015RandomForestTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Jia2015RFResults/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadJiaYeast(resultsFolderName)
			runTest(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=10)
			runTest(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Jia2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if you2015RandomForestTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'You2015RFResults/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=10)
			runTest(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(You2015RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)



	if ding2016RandomForestTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Ding2016RFResults/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=5)
			runTest(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Ding2016RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if wang2017RotFTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Wang2017RotFResults/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=5)
			runTest(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Wang2017RotFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if chen2019LGBMTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Chen2019LGBMTest/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=5)
			runTest(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Chen2019LGBMModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if jia2019RandomForestTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Jia2019RFResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadJiaYeast(resultsFolderName,trainDataPerClass='Max',full=False)
			runTest(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName,kfolds=10)
			runTest(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Jia2019RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if randomNetworkTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'RandomNetworkResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'fullGPU':True}
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadRichouxHumanDataStrict(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoMultiSpeciesChen(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadLiuFruitFly(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadJiaYeast(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)



		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(RandomNetworkModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if randomRFTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'RandomRFResults/'
		PPIPUtils.makeDir(resultsFolderName)

		hyp = {'fullGPU':True}
		
		if orgData:
			trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadRichouxHumanDataStrict(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadGuoMultiSpeciesChen(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadLiuFruitFly(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			trainSets, testSets, saves, pfs, folderName = loadJiaYeast(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(RandomRFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

	if biasTests:
		for mod in [(BasicBiasModule,'BasicBiasModule'),(BasicBiasModuleSeqSim,'BasicBiasModuleSeqSim'),(BasicBiasModuleGOSimSeqSim,'BasicBiasModuleGOSimSeqSim')]:
			PPIPUtils.makeDir(baseResultsFolderName)
			resultsFolderName = baseResultsFolderName+mod[1]+'/'
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {}
			if orgData:
				trainSets, testSets, saves, pfs, folderName = loadMartinHPylori(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataTian(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanHumanLarge(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanHumanSmall(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadPanMartinHuman(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadGuoYeastDataChen(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadRichouxHumanDataStrict(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadGuoMultiSpeciesChen(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadLiuFruitFly(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadDuYeast(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
				trainSets, testSets, saves, pfs, folderName = loadJiaYeast(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)

			if HumanRandom50:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20:
				trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
				runTestLst(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
				runTest(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
				testSets = [testSets[0]]
				pfs = [pfs[0]]
				runTestLst(mod[0], None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if MaetschkeVarTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'MaetschkeVarResults'+'/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(MaetschkeVar2011Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(MaetschkeVar2011Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(MaetschkeVar2011Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(MaetschkeVar2011Module, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)


	if Chen2005RF:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Chen2005RFResults'+'/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
			runTest(Chen2005RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
			runTestLst(Chen2005RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
			runTest(Chen2005RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
			runTestLst(Chen2005RFModule, None,trainSets,testSets,folderName,hyp,saveModels=saves,predictionsFLst = pfs)




	if GouVar2006GOLRTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Guo2007SimResults'+'/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {}
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderNames = loadHumanRandom50(resultsFolderName,dirLst=True)
			runTestPairwiseFoldersLst(GouVar2006GOLRModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderNames = loadHumanRandom20(resultsFolderName,dirLst=True)
			loads = [None]*len(saves)  #since Semantic Similarities do not change based on test set, can skip doing training 2nd time
			loads[len(saves)//2:] = saves[:len(saves)//2]
			runTestPairwiseFoldersLst(GouVar2006GOLRModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs,loads=loads)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut50(resultsFolderName,dirLst=True)
			runTestPairwiseFoldersLst(GouVar2006GOLRModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut20(resultsFolderName,dirLst=True)
			loads = [None]*len(saves) #since Semantic Similarities do not change based on test set, can skip doing training 2nd time
			loads[len(saves)//2:] = saves[:len(saves)//2]
			runTestPairwiseFoldersLst(GouVar2006GOLRModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs,loads=loads)


	if Zhang2016GOTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		resultsFolderName = baseResultsFolderName+'Zhang2016GO'+'/'
		PPIPUtils.makeDir(resultsFolderName)
		hyp = {'Model':'THUNDERSVM'}
		if HumanRandom50:
			trainSets, testSets, saves, pfs, folderNames = loadHumanRandom50(resultsFolderName,dirLst=True)
			runTestPairwiseFoldersLst(Zhang2016GOModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanRandom20:
			trainSets, testSets, saves, pfs, folderNames = loadHumanRandom20(resultsFolderName,dirLst=True)
			loads = [None]*len(saves)  #since Semantic Similarities do not change based on test set, can skip doing training 2nd time
			loads[len(saves)//2:] = saves[:len(saves)//2]
			runTestPairwiseFoldersLst(Zhang2016GOModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs,loads=loads,startIdx=8)
		if HumanHeldOut50:
			trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut50(resultsFolderName,dirLst=True)
			runTestPairwiseFoldersLst(Zhang2016GOModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
		if HumanHeldOut20:
			trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut20(resultsFolderName,dirLst=True)
			loads = [None]*len(saves) #since Semantic Similarities do not change based on test set, can skip doing training 2nd time
			loads[len(saves)//2:] = saves[:len(saves)//2]
			runTestPairwiseFoldersLst(Zhang2016GOModule, None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs,loads=loads)

	if ZhangDomainVar2016Test:
		PPIPUtils.makeDir(baseResultsFolderName)
		midFolder = baseResultsFolderName + 'ZhangDomainVar2016Results/'
		PPIPUtils.makeDir(midFolder)
		idx = 0
		for pair in [(ZhangDomainVar2016AllModule,midFolder+'All/'), (ZhangDomainVar2016NonTestModule,midFolder+'NonTest/'), (ZhangDomainVar2016HeldOutModule,midFolder+'HeldOut/')]:
			resultsFolderName = pair[1]
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {}
			if HumanRandom50 and idx !=2: #idx=2 is held out data, which only works on the held out protein datasets
				trainSets, testSets, saves, pfs, folderNames = loadHumanRandom50(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20 and idx !=2:
				trainSets, testSets, saves, pfs, folderNames = loadHumanRandom20(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut50(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut20(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)

			idx += 1
			

	if SimpleEnsembleTest:
		PPIPUtils.makeDir(baseResultsFolderName)
		midFolder = baseResultsFolderName + 'SimpleEnsembleResults/'
		PPIPUtils.makeDir(midFolder)
		idx = 0
		for pair in [(SimpleEnsembleAllModule,midFolder+'All/'), (SimpleEnsembleNonTestModule,midFolder+'NonTest/'), (SimpleEnsembleHeldOutModule,midFolder+'HeldOut/')]:
			resultsFolderName = pair[1]
			PPIPUtils.makeDir(resultsFolderName)
			hyp = {}
			if HumanRandom50 and idx !=2: #idx=2 is held out data, which only works on the held out protein datasets
				trainSets, testSets, saves, pfs, folderNames = loadHumanRandom50(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanRandom20 and idx !=2:
				trainSets, testSets, saves, pfs, folderNames = loadHumanRandom20(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut50:
				trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut50(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)
			if HumanHeldOut20:
				trainSets, testSets, saves, pfs, folderNames = loadHumanHeldOut20(resultsFolderName,dirLst=True)
				runTestPairwiseFoldersLst(pair[0], None,None,None,folderNames,hyp,saveModels=saves,predictionsFLst = pfs)

			idx += 1



def genSequenceFeatures():
	createFeatures(currentDir+'PPI_Datasets/Guo_Data_Yeast_Tian/',set(['EGBW11','AC30','MMI','LD10_CTD','PSAAC15','Moran','Geary','AC11','PSAAC9','PSSMAAC','PSSMDPC','SkipGramAA25H20','LD10_CTD','NumericEncoding20Skip3','MCD4CTD','PSSMLST','PSSMAAC','PSSMDPC','JIA_DWT','MLD4CTD','NMBROTO_6_30','AAC20','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','conjointTriad','CHAOS','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Guo_Data_Yeast_Chen/',set(['SkipGramAA7','OneHotEncoding7','SkipGramAA25H20','NumericEncoding20Skip3','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Guo_MultiSpecies_Chen/',set(['AC14_30','conjointTriad','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Du_Yeast/',set(['MCD5CTD','LD10_CTD','AC30','AAC20','AAC400','DUMULTIGROUPCTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','Grantham_Quasi_30','Schneider_Quasi_30','APSAAC30_2','PSEAAC_3','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Jia_Data_Yeast/',set(['JIA_DWT','CHAOS','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Martin_H_pylori/',set(['EGBW11','AC11','PSAAC9','NumericEncoding20Skip3','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','Grantham_Quasi_30','Schneider_Quasi_30','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','PSSMDPC','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','MMI','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','LD10_CTD','conjointTriad','CHAOS','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Liu_Fruit_Fly/',set(['Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','Grantham_Quasi_30','Schneider_Quasi_30','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Li_AD/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Large/',set(['AC30','NumericEncoding22','AC14_30','conjointTriad','PSAAC20','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Pan_Human_Data/Pan_Small/',set(['SkipGramAA25H20','NumericEncoding20Skip3','PSSMLST','PSSMDPC','SkipWeightedConjointTriad','PSAAC20','conjointTriad','AC30','AAC20','AAC400','DUMULTIGROUPCTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','Grantham_Quasi_30','Schneider_Quasi_30','APSAAC30_2','NMBROTO_6_30','MMI','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Pan_Human_Data/Martin_Human/',set(['PSSMDPC','SkipWeightedConjointTriad','PSAAC20','conjointTriad','AC30','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Richoux_Human_Data/',set(['OneHotEncoding24','Random500','AllvsAllSim']))
	
	createFeatures(currentDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))


if __name__ == '__main__':
	genSequenceFeatures()
	RunAll()