import re
import xml.etree.ElementTree
import os
import os.path
import sys
import argparse
#add parent and grandparent and great-grandparent to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
currentDir = currentdir + '/'
import PPIPUtils

def getHumanFromBioGRID(num = '4.3.194',folderName=currentDir+'BioGRIDOrganisms/',outputFile=currentDir+'BioGRIDOrganisms/BIOGRID_PARSED_HUMAN_4.3.194.tsv'):
	if folderName[-1] != '/' and folderName[-1] != '\\':
		folderName=folderName+'/'
	PPIPUtils.makeDir(folderName)

	if not os.path.isdir(folderName+'BIOGRID-ORGANISM-'+num+'.tab2'):
		PPIPUtils.downloadZipFile('https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.3.194/BIOGRID-ORGANISM-'+num+'.tab2.zip', folderName+'BIOGRID-ORGANISM-'+num+'.tab2')
	if not os.path.isdir(folderName+'BIOGRID-ORGANISM-'+num+'.psi25'):
		PPIPUtils.downloadZipFile('https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.3.194/BIOGRID-ORGANISM-'+num+'.psi25.zip', folderName+'BIOGRID-ORGANISM-'+num+'.psi25')


	#https://www.ebi.ac.uk/ols/ontologies/mi/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FMI_0407
	lst = []
	count = 0
	species = 'Human'
	species2 = 'Homo_sapiens'
	#f2 = open('../BIOGRID_PARSED_'+species+'_'+num+'.tsv','w')
	f2 = open(outputFile,'w')
	f2.write('\tSpecies Interaction ID (xml)\tGene-Gene name(xml)\tNCBI Gene1 ID\tNCBI Gene2 ID\tGene1 Name (tsv)\tGene2 Name (tsv)\tNCBI Gene1 Tax ID\tNCBI Gene 2 Tax ID\tInteraction Type\tEvidence Code\n')
	speciesCount = []
	for fname in os.listdir(folderName+'Biogrid-Organism-'+num+'.psi25/'):
		speciesname = fname[17:-(11+len(num))]
		if speciesname != species2:
			continue
		count = 0
		count2 = 0
		f = open(folderName+'Biogrid-Organism-'+num+'.tab2/Biogrid-Organism-'+speciesname+'-'+num+'.tab2.txt')
		intLst = [None]
		header = None
		for line in f:
			if not header:
				header = line.strip().split('\t')
				continue
			intLst.append(line.strip().split('\t'))
		f.close()
		f = open(folderName+'/Biogrid-Organism-'+num+'.psi25/'+fname)
		for line in f:
			if len(lst) == 0:
				if line.strip()[0:16] == '<interaction id=':
					lst.append(line)
			else:
				lst.append(line)
				if line.strip()[0:14] == '</interaction>':
					try:
						lst = ''.join(lst)
						xmlSt = xml.etree.ElementTree.fromstring(lst)
						id = xmlSt.attrib['id']
						geneNames = xmlSt.find('names').find('shortLabel').text
						iType = xmlSt.find('interactionType').find('names').find('shortLabel').text
						iTypeMSI = xmlSt.find('interactionType').find('xref').find('primaryRef').attrib['id']
						evidenceCode = xmlSt.findall('./attributeList/*[@name=\'BioGRID Evidence Code\']')[0].text
						intData = intLst[int(id)]
						geneIds = (intData[1],intData[2])
						geneSyms = (intData[7],intData[8])
						taxID = (intData[15],intData[16])
						if str.lower(geneSyms[0]+'-'+geneSyms[1]) != str.lower(geneNames):
								print('Possible Error',geneIds,geneSyms,geneNames,id,speciesname,taxID)
						lst = ['1237','1139','0220','0200','0568','0701','1310','0570','2272','0408','1250','2263','0199','0201','0207','0556','0559','0910','0213','0567','0212','0202','0871','0881','1027','1143','1251','0902','0945','2280','2356','2273','0558','0987','0197','0204','0195','0217','0198','0566','0203','1355','0216','0971','0214','0211','1148','0569','1327','1140','0883','0194','0414','0210','1127','0192','0209','1146','0193','0557','0206','0882','0986','1230','0572','0571','0844','1126','0985','2252']
						#these are all subtypes of 0407, but I've never seen biogrid use one. . . 
						if iTypeMSI.split(':')[1] in lst:
							print(geneIds,iTypeMSI)
							f2.write(speciesname+'\t'+id+'\t'+geneNames+'\t'+geneIds[0]+'\t'+geneIds[1]+'\t'+geneSyms[0]+'\t'+geneSyms[1]+'\t'+str(taxID[0])+'\t'+str(taxID[1]) + '\t' + iTypeMSI+'\t' + evidenceCode + '\n')
							count2 += 1
						if iTypeMSI.split(':')[1] == '0407':
							f2.write(speciesname+'\t'+id+'\t'+geneNames+'\t'+geneIds[0]+'\t'+geneIds[1]+'\t'+geneSyms[0]+'\t'+geneSyms[1]+'\t'+str(taxID[0])+'\t'+str(taxID[1]) + '\t' + iTypeMSI+'\t' + evidenceCode + '\n')
							count2 += 1
						count += 1
						lst = []
					except Exception as e:
						print(e)
						print(lst)
						exit(42)
						lst = []
					
		f.close()
		print(speciesname+'\tTotal Interactions '+str(count)+'\tBioPhysical Interactions '+str(count2))
	f2.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parses BioPhysical interactions from BioGRID to create a final output file')
	parser.add_argument('--Num',help='BioGRID version number',default='4.3.194')
	parser.add_argument('--FolderName',help='Folder to save/load BioGRID data from',default=currentDir+'BioGRIDOrganisms/')
	parser.add_argument('--OutputFile',help='File to write Parsed BioGRID data to',default=currentDir+'BioGRIDOrganisms/'+'BIOGRID_PARSED_ALL_4.3.194.tsv')
	args = parser.parser_args()
	getAllBioGRID(args.Num,args.FolderName,args.OutputFile)