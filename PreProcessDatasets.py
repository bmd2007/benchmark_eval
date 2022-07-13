import sys
from preprocess.ConjointTriad import ConjointTriad
from preprocess.PreprocessUtils import readFasta
from preprocess.PreprocessUtils import AllvsAllSim
from preprocess.CTD_Composition import CTD_Composition
from preprocess.CTD_Transition import CTD_Transition
from preprocess.CTD_Distribution import CTD_Distribution
from preprocess.MMI import MMI
from preprocess.NMBrotoAC import NMBrotoAC
from preprocess.GearyAC import GearyAC
from preprocess.MoranAC import MoranAC
from preprocess.PSSMAAC import PSSMAAC
from preprocess.PSSMDPC import PSSMDPC
from preprocess.PSSMDCT import PSSMDCT
from preprocess.PSEAAC import PSEAAC
from preprocess.LDCTD import LDCTD
from preprocess.MCDCTD import MCDCTD
from preprocess.MLDCTD import MLDCTD
from preprocess.EGBW import EGBW
from preprocess.AutoCovariance import AutoCovariance
#from preprocess.BergerEncoding import BergerEncoding
from preprocess.SkipGram import SkipGram
from preprocess.OneHotEncoding import OneHotEncoding
from preprocess.NumericEncoding import NumericEncoding
from preprocess.AAC import AAC
from preprocess.PairwiseDist import PairwiseDist
from preprocess.QuasiSequenceOrder import QuasiSequenceOrder
from preprocess.PSSMLST import PSSMLST
from preprocess.SkipWeightedConjointTriad import SkipWeightedConjointTriad
from preprocess.DWTAC import DWTAC
from preprocess.Chaos import Chaos
from preprocess.Random import Random
import numpy as np
import time


def createFeatures(folderName,featureSets,processPSSM=True,deviceType='cpu'):
	t =time.time()
	fastas = readFasta(folderName+'allSeqs.fasta')
	print('fasta loaded',time.time()-t)
	
	if 'AC30' in featureSets:
		#Guo AC calculation
		#calc AC
		ac = AutoCovariance(fastas,lag=30,deviceType=deviceType)

		f = open(folderName+'AC30.tsv','w')
		for item in ac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
	
	
		print('AC30',time.time()-t)
	if 'AC11' in featureSets:
		#Guo AC calculation
		#calc AC
		ac = AutoCovariance(fastas,lag=11,deviceType=deviceType)

		f = open(folderName+'AC11.tsv','w')
		for item in ac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
	
	
		print('AC11',time.time()-t)
	if 'conjointTriad' in featureSets or 'CT' in featureSets:
		#Conjoint Triad
		ct = ConjointTriad(fastas,deviceType=deviceType)
		f = open(folderName+'ConjointTriad.tsv','w')
		for item in ct:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
	
	
		print('CT',time.time()-t)
	if 'LD10_CTD' in featureSets:
		#Composition/Transition/Distribution Using Conjoint Triad Features on LD encoding
		(comp, tran, dist) = LDCTD(fastas)
		
		lst1 = [comp,tran,dist]
		lst2 = ['LD10_CTD_ConjointTriad_C.tsv','LD10_CTD_ConjointTriad_T.tsv','LD10_CTD_ConjointTriad_D.tsv']
		for i in range(0,3):
			f = open(folderName+lst2[i],'w')
			for item in lst1[i]:
				f.write('\t'.join(str(s) for s in item)+'\n')
			f.close()

		print('LD10_CTD',time.time()-t)
	if 'MCD5CTD' in featureSets:
		#Composition/Transition/Distribution Using Conjoint Triad Features on MCD encoding
		(comp, tran, dist) = MCDCTD(fastas,5)
		
		lst1 = [comp,tran,dist]
		lst2 = ['MCD5_CTD_ConjointTriad_C.tsv','MCD5_CTD_ConjointTriad_T.tsv','MCD5_CTD_ConjointTriad_D.tsv']
		for i in range(0,3):
			f = open(folderName+lst2[i],'w')
			for item in lst1[i]:
				f.write('\t'.join(str(s) for s in item)+'\n')
			f.close()
		print('MCD5CTD',time.time()-t)

	if 'MLD4CTD' in featureSets:
		#Composition/Transform/Distribution Using Conjoint Triad Features on MLD encoding
		(comp, tran, dist) = MLDCTD(fastas,4)
		
		lst1 = [comp,tran,dist]
		lst2 = ['MLD4_CTD_ConjointTriad_C.tsv','MLD4_CTD_ConjointTriad_T.tsv','MLD4_CTD_ConjointTriad_D.tsv']
		for i in range(0,3):
			f = open(folderName+lst2[i],'w')
			for item in lst1[i]:
				f.write('\t'.join(str(s) for s in item)+'\n')
			f.close()
		print('MLD4CTD',time.time()-t)

	if 'MCD4CTD' in featureSets:
		#Composition/Transform/Distribution Using Conjoint Triad Features on MCD encoding
		(comp, tran, dist) = MCDCTD(fastas,4)
		
		lst1 = [comp,tran,dist]
		lst2 = ['MCD4_CTD_ConjointTriad_C.tsv','MCD4_CTD_ConjointTriad_T.tsv','MCD4_CTD_ConjointTriad_D.tsv']
		for i in range(0,3):
			f = open(folderName+lst2[i],'w')
			for item in lst1[i]:
				f.write('\t'.join(str(s) for s in item)+'\n')
			f.close()
		print('MCD4CTD',time.time()-t)


	if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
		#Li's PAAC used the first 3 variables from his moran AAC list, which appears to match what other authors have used
		paac = PSEAAC(fastas,lag=15)
		f = open(folderName+'PSAAC15.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()

		print('PSAAC15',time.time()-t)
		
	if 'PSAAC9' in featureSets or 'PSEAAC9' in featureSets:
		#Li's PAAC used the first 3 variables from his moran AAC list, which appears to match what other authors have used
		paac = PSEAAC(fastas,lag=9)
		f = open(folderName+'PSAAC9.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSAAC9',time.time()-t)
		
	if 'PSAAC20' in featureSets or 'PSEAAC20' in featureSets:
		#Li's PAAC used the first 3 variables from his moran AAC list, which appears to match what other authors have used
		paac = PSEAAC(fastas,lag=20)
		f = open(folderName+'PSAAC20.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()

		print('PSAAC20',time.time()-t)

	if 'MMI' in featureSets:
		vals = MMI(fastas,deviceType=deviceType)
		f = open(folderName+'MMI.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('MMI',time.time()-t)
	
	if 'Moran' in featureSets:
		vals = MoranAC(fastas,deviceType=deviceType)
		f = open(folderName+'MoranAC30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Moran',time.time()-t)

	if 'Geary' in featureSets:
		vals = GearyAC(fastas,deviceType=deviceType)
		f = open(folderName+'GearyAC30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Geary',time.time()-t)

	if 'PSSMAAC' in featureSets:
		vals = PSSMAAC(fastas,folderName+'PSSM/',processPSSM=processPSSM,deviceType=deviceType)
		processPSSM=False #don't try to compute PSSMs for any further variables utlizing PSSMs
		f = open(folderName+'PSSMAAC.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSSMAAC',time.time()-t)

	if 'PSSMDPC' in featureSets:
		vals = PSSMDPC(fastas,folderName+'PSSM/',processPSSM=processPSSM,deviceType=deviceType)
		processPSSM=False #don't try to compute PSSMs for any further variables utlizing PSSMs
		f = open(folderName+'PSSMDPC.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSSMDPC',time.time()-t)
	
	if 'EGBW11' in featureSets:
		vals = EGBW(fastas,11,deviceType=deviceType)
		f = open(folderName+'EGBW11.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('EGBW11',time.time()-t)

	if 'OneHotEncoding7' in featureSets:
		OneHotEncoding(fastas,folderName+'OneHotEncoding7.encode')
		print('OneHotEncoding7',time.time()-t)

	if 'SkipGramAA7' in featureSets:
		SkipGram(fastas,folderName+'SkipGramAA7H5.encode',hiddenSize=5,deviceType='cuda',fullGPU=True)
		print('SkipGramAA7',time.time()-t)

	if 'SkipGramAA25H20' in featureSets:
		#Yao's 2019 paper mentions 25 unique amino acids in uniprot
		#IUPAC defines B, Z, in addition to the standard 20 amino acids, and X, for any acid
		#uniprot defines U and O as non-standard letters
		#https://www.uniprot.org/help/sequences https://www.bioinformatics.org/sms2/iupac.html
		groupings = 'A R N D C Q E G H I L K M F P S T W Y V B Z U O X'.split()
		SkipGram(fastas,folderName+'SkipGramAA25H20.encode',groupings=groupings,hiddenSize=20,windowSize=4,deviceType='cuda',fullGPU=True)
		print('SkipGramAA25H20',time.time()-t)
	
	if 'OneHotEncoding24' in featureSets:
		#note, Richoux's paper converts all unknown amino acids to X, which is the last group, so we are trying to do something similar
		#23 amino acids
		groupings = 'A R N D C Q E G H I L K M F P S T W Y V U B Z'.split()
		#X amino acid, which includes all other amino acids.  Using Uniprot, the only non-standard letters are U (matching Richoux's U, and O, which is unlisted in their paper).
		groupings.append('JOX')
		OneHotEncoding(fastas,folderName+'OneHotEncoding24.encode',groupings=groupings,deviceType='cpu')
		print('OneHotEncoding24',time.time()-t)

	if 'NumericEncoding22' in featureSets:
		#note, Li's paper (Deep Neural Network Based Predictions of Protein Interactions Using Primary Sequences)
		#they mention an encoding input dim of 23.  Leaving 1 value for zero padding, there are only 20 standard and 2 non-standard values used by Uniprot, that I know of, so I am encoding using that.
		groupings = 'A R N D C Q E G H I L K M F P S T W Y V U O'.split()
		NumericEncoding(fastas,folderName+'NumericEncoding22.encode',groupings=groupings,deviceType='cpu')
		print('NumericEncoding22',time.time()-t)

	#encode 20 AAs, with non-overlapping windows of length 3, removing letters from the beginning with length doesn't divide equally
	if 'NumericEncoding20Skip3' in featureSets:
		aac = NumericEncoding(fastas,folderName+'NumericEncoding20Skip3.encode',groupings=None,groupLen=3,gap=3,truncate='left',deviceType='cpu')
		print('NumericEncoding20Skip3',time.time()-t)

	
	if 'AC14_30' in featureSets:
		#use 7 additional aa properties, in addition to the standard 7, based on Chen's Work:Protein-protein interaction prediction using a hybrid feature representation and a stacked generalization scheme


		#original 7 from GUO
		aaIDs = ['GUO_H1','HOPT810101','KRIW790103','GRAR740102','CHAM820101','ROSG850103_GUO_SASA','GUO_NCI']
		#new values from Chen
		#PONN and KOLA are from
		#HYDROPHOBIC PACKING AND SPATIAL ARRANGEMENT OF AMINO ACID RESIDUES IN GLOBULAR PROTEINS P.K. PONNUSWAMY, M. PRABHAKARAN and P. MANAVALAN
		#Kolaskar AS, Tongaonkar PC. A Semiempirical method for prediction of antigenic determinants on protein antigens. FEBS Lett. 1990;276(1–2):172–4.
		#PARJ and JANJ were already in amino acid index
		#remaining 3 were copied from Chen's table
		aaIDs += ['PARJ860101','PONN800101_CHEN_HYDRO','CHEN_FLEX','CHEN_ACCESS','JANJ780103','CHEN_TURNS','KOLA900101_CHEN_ANTE']
		ac = AutoCovariance(fastas,lag=30,aaIDs=aaIDs,deviceType=deviceType)
		
		f = open(folderName+'AC14_30.tsv','w')
		for item in ac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		
		print('AC14_30',time.time()-t)



	#zhou lists these as his features, but its unclear which ones he uses for which algorithm
	#I could not find features matching his first (Hydrophobicity scale) and last (Side-chain mass) attributes, so I'm using the general ones I have
	zhouAAIds = ['GUO_H1','KRIW790103','WOEC730101','CHAM820101','DAYM780201','BIGC670101','ROSG850103_GUO_SASA','GUO_NCI','GUO_H1','HOPT810101','CHOU_SIDE_MASS']

	if 'Geary_Zhao_30' in featureSets:
		vals = GearyAC(fastas,aaIDs=zhouAAIds[0:8],deviceType=deviceType)
		f = open(folderName+'Geary_Zhao_30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Geary_Zhao_30',time.time()-t)
	
	if 'NMBroto_Zhao_30' in featureSets:
		vals = NMBrotoAC(fastas,aaIDs=zhouAAIds[0:8],deviceType=deviceType)
		f = open(folderName+'NMBroto_Zhao_30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('NMBroto_Zhao_30',time.time()-t)
	
	
	if 'Moran_Zhao_30' in featureSets:
		vals = MoranAC(fastas,aaIDs=zhouAAIds[0:8],deviceType=deviceType)
		f = open(folderName+'Moran_Zhao_30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Moran_Zhao_30',time.time()-t)
	
	#same as regular PSEAAC, shouldn't have added Zhao's name to it.
	if 'PSEAAC_Zhao_30' in featureSets:
		paac = PSEAAC(fastas,aaIDs=zhouAAIds[8:],lag=30)
		f = open(folderName+'PSEAAC_Zhao_30.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSEAAC_Zhao_30',time.time()-t)
	
	
	if 'Grantham_Sequence_Order_30' in featureSets:
		ac = PairwiseDist(fastas, pairwiseAAIDs=['Grantham'], calcType='SumSq', lag=30)
		f = open(folderName+'Grantham_Sequence_Order_30.tsv','w')
		for item in ac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Grantham_Sequence_Order_30',time.time()-t)
		
		
	if 'Schneider_Sequence_Order_30' in featureSets:
		ac = PairwiseDist(fastas, pairwiseAAIDs=['Schneider-Wrede'],calcType='SumSq', lag=30)
		f = open(folderName+'Schneider_Sequence_Order_30.tsv','w')
		for item in ac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Schneider_Sequence_Order_30',time.time()-t)
		
	
	if 'Grantham_Quasi_30' in featureSets:
		paac = QuasiSequenceOrder(fastas, pairwiseAAIDs=['Grantham'],lag=30)
		f = open(folderName+'Grantham_Quasi_30.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Grantham_Quasi_30',time.time()-t)


	if 'Schneider_Quasi_30' in featureSets:
		paac = QuasiSequenceOrder(fastas, pairwiseAAIDs=['Schneider-Wrede'],lag=30)
		f = open(folderName+'Schneider_Quasi_30.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Schneider_Quasi_30',time.time()-t)
	

	if 'PSSMLST' in featureSets:
		PSSMLST(fastas,folderName+'PSSM/',folderName,processPSSM=processPSSM)
		processPSSM=False #don't try to compute PSSMs for any further variables utlizing PSSMs
		print('PSSMLST',time.time()-t)


	if 'SkipWeightedConjointTriad' in featureSets:
		
		#Weighted Skip Conjoint Triad
		#paper doesn't mention what weights are used, currently setting skip triad to half the weight of the sequence triad
		ct = SkipWeightedConjointTriad(fastas,weights=[1,.5,.5],deviceType=deviceType)
		f = open(folderName+'SkipWeightedConjointTriad.tsv','w')
		for item in ct:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('SkipWeightedConjointTriad',time.time()-t)
	
	if 'AAC20' in featureSets:
		aac = AAC(fastas)
		f = open(folderName+'AAC20.tsv','w')
		for item in aac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('AAC20',time.time()-t)
	
	if 'AAC400' in featureSets:
		aac = AAC(fastas,groupLen=2)
		f = open(folderName+'AAC400.tsv','w')
		for item in aac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('AAC400',time.time()-t)
	
	if 'DUMULTIGROUPCTD' in featureSets:
		groupings = {}
		groupings['Hydrophobicity'] = ['RKEDQN','GASTPHY','CLVIMFW']
		groupings['Normalized_van_der_Waals_volume'] = ['GASTPD','NVEQIL','MHKFRYW']
		groupings['Polarity'] = ['LIFWCMVY','PATGS','HQRKNED']
		groupings['Polarizability'] = ['GASDT','CPNVEQIL','KMHFRYW']
		groupings['Charge'] = ['KR','ANCQGHILMFPSTWYV','DE']
		groupings['Secondary_structure'] = ['EALMQKRH','VIYCWFT','GNPSD']
		groupings['Solvent_accessibility'] = ['ALFCGIVW','PKQEND','MPSTHY']
		groupings['Surface_tension'] = ['GQDNAHR','KTSEC','ILMFPWYV']
		groupings['Protein-protein_interface_hotspot_propensity-Bogan'] = ['DHIKNPRWY','EQSTGAMF','CLV']
		groupings['Protein-protein_interface_propensity-Ma'] = ['CDFMPQRWY','AGHVLNST','EIK']
		groupings['Protein-DNA_interface_propensity-Schneider'] = ['GKNQRSTY','ADEFHILVW','CMP']
		groupings['Protein-DNA_interface_propensity-Ahmad'] = ['GHKNQRSTY','ADEFIPVW','CLM']
		groupings['Protein-RNA_interface_propensity-Kim'] = ['HKMRY','FGILNPQSVW','CDEAT']
		groupings['Protein-RNA_interface_propensity-Ellis'] = ['HGKMRSYW','AFINPQT','CDELV']
		groupings['Protein-RNA_interface_propensity-Phipps'] = ['HKMQRS','ADEFGLNPVY','CITW']
		groupings['Protein-ligand_binding_site_propensity_-Khazanov'] = ['CFHWY','GILNMSTR','AEDKPQV']
		groupings['Protein-ligand_valid_binding_site_propensity_-Khazanov'] = ['CFHWYM','DGILNSTV','AEKPQR']
		groupings['Propensity_for_protein-ligand_polar_&_aromatic_non-bonded_interactions-Imai'] = ['DEHRY','CFKMNQSTW','AGILPV']
		groupings['Molecular_Weight'] = ['AGS','CDEHIKLMNQPTV','FRWY']
		groupings['cLogP'] = ['RKDNEQH','PYSTGACV','WMFLI']
		groupings['No_of_hydrogen_bond_donor_in_side_chain'] = ['HKNQR','DESTWY','ACGFILMPV']
		groupings['No_of_hydrogen_bond_acceptor_in_side_chain'] = ['DEHNQR','KSTWY','ACGFILMPV']
		groupings['Solubility_in_water'] = ['ACGKRT','EFHILMNPQSVW','DY']
		groupings['Amino_acid_flexibility_index'] = ['EGKNQS','ADHIPRTV','CFLMWY']
		for item in [(CTD_Composition,'DuMultiCTD_C'),(CTD_Transition,'DuMultiCTD_T'),(CTD_Distribution,'DuMultiCTD_D')]:
			func = item[0]
			vals = []
			for feat in groupings:
				results = func(fastas,groupings=groupings[feat])
				for i in range(0,len(results[0])): #header row
					results[0][i] = feat+'_'+results[0][i] #add feature name to each column in header row
				results = np.asarray(results)
				if len(vals) == 0:
					vals.append(results)
				else:
					vals.append(results[:,1:])#remove protein names if not first group calculated
			vals = np.hstack(vals).tolist()
			f = open(folderName+item[1]+'.tsv','w')
			for line in vals:
				f.write('\t'.join(str(s) for s in line)+'\n')
			f.close()
		print('DUMULTIGROUPCTD',time.time()-t)

	if 'APSAAC30_2' in featureSets:
		#Note, Du's paper states W=0.5, but the standard is W=0.05.  In theory, W should be lower with more attributes (due to amphipathic=True) to balance betters with AA counts.
		#Currently leaving at 0.05, assuming this may be a typo.  Can change later as necessary.
		paac = PSEAAC(fastas,aaIDs=['GUO_H1','HOPT810101'],lag=30,amphipathic=True)
		f = open(folderName+'APSAAC30.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('APSAAC30_2',time.time()-t)
	
	if 'JIA_DWT' in featureSets:
		paac = DWTAC(fastas)
		f = open(folderName+'DWTAC.tsv','w')
		for item in paac:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('JIA_DWT',time.time()-t)
	
	if 'NMBROTO_6_30' in featureSets:
		vals = NMBrotoAC(fastas,aaIDs=['GUO_H1','KRIW790103','GRAR740102','CHAM820101','ROSG850103_GUO_SASA','GUO_NCI'],deviceType=deviceType)
		f = open(folderName+'NMBroto_6_30.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('NMBROTO_6_30',time.time()-t)
	
	if 'PSSMDCT' in featureSets:
		vals = PSSMDCT(fastas,folderName+'PSSM/',deviceType=deviceType,processPSSM=processPSSM)
		processPSSM=False #don't try to compute PSSMs for any further variables utlizing PSSMs
		f = open(folderName+'PSSMDCT.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSSMDCT',time.time()-t)
		
	if 'NMBROTO_9' in featureSets:
		vals = NMBrotoAC(fastas,lag=9,deviceType=deviceType)
		f = open(folderName+'NMBroto_9.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('NMBROTO_9',time.time()-t)

	if 'MORAN_9' in featureSets:
		vals = MoranAC(fastas,lag=9,deviceType=deviceType)
		f = open(folderName+'Moran_9.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('MORAN_9',time.time()-t)

	if 'GEARY_9' in featureSets:
		vals = GearyAC(fastas,lag=9,deviceType=deviceType)
		f = open(folderName+'GEARY_9.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('GEARY_9',time.time()-t)

	if 'PSEAAC_3' in featureSets:
		vals = PSEAAC(fastas,lag=3,deviceType=deviceType)
		f = open(folderName+'PSEAAC_3.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('PSEAAC_3',time.time()-t)
		
	if 'CHAOS' in featureSets:
		vals = Chaos(fastas)
		f = open(folderName+'Chaos.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('CHAOS',time.time()-t)
	
	if 'Random500' in featureSets:
		vals = Random(fastas)
		f = open(folderName+'Random500.tsv','w')
		for item in vals:
			f.write('\t'.join(str(s) for s in item)+'\n')
		f.close()
		print('Random500',time.time()-t)
		
	if 'AllvsAllSim' in featureSets:
		AllvsAllSim(folderName)
		print('AllvsAllSim',time.time()-t)
