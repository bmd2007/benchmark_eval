# benchmark_eval
MIT Licensed Code <br>
 <br>
This repistory contains all the code necessary to recreate models and results from Paper: Benchmark Evaluation of Protein–Protein Interaction Prediction Algorithms, 
as well as datasets produced by other authors that were compared in the paper. <br>
 <br>
<h2>Software and Packages</h2>
To run the code, you will need the following python packages and software <br>
Packages: <br>
&emsp;&emsp;	goatools  (for gene ontology computations) <br>
&emsp;&emsp;	lightgbm<br>
&emsp;&emsp;	matplotlib<br>
&emsp;&emsp;	numpy<br>
&emsp;&emsp;	pywt<br>
&emsp;&emsp;	rotation_forest (https://github.com/digital-idiot/RotationForest/tree/master/rotation_forest) <br>
&emsp;&emsp;	scipy<br>
&emsp;&emsp;	skimage<br>
&emsp;&emsp;	sklearn<br>
&emsp;&emsp;	thundersvm (optional, allows for running SVMs using gpu or multiple CPUs, but requires compiling code see below)<br>
&emsp;&emsp;	torch<br>
&emsp;&emsp;	libSVM (optional, allows running SVM's using libSVMs code.  Note that sklearn uses libsvm as their core svm library, so you can use that instead)<br>
<br>	
Software: <br>
&emsp;&emsp;	Blast+ (Needed for computing PSSMs, should be accessible from command line.  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) <br>
&emsp;&emsp;	wget (Needed for downloading some large files, should be accesible from command line). <br>
&emsp;&emsp;	thundersvm (Optional, needs to be downloaded and compiled to use thundersvm python library.  https://github.com/Xtra-Computing/thundersvm) <br>
 <br>
 <br>
 <br>
 <br>
<h2>Project Layout</h2>
The code is split into the following folders: <br>
 <br>
The Methods Folder contain scripts creating the machine learning models used to predict protein interactions <br>
 <br>
The preprocess and pairwisePreprocess folders contain scripts related to creating sequence-based and annotation-based features used by the machine learning models <br>
 <br>
The PPI_Datasets folder contains data and scripts related to individual datasets, such as parsing the protein pairs from the original documents and downloading the sequence and annotations necessary for the different experiments in the paper.  Additionally, each datasets folder will contain all features describing proteins and protein pairs in given dataset.  Unlike most code in this project, code found in this folder is only usable for the individual dataset of interest. <br>
 <br>
The main folder has scripts related to running tests and loading the previously mentioned datasets.  Notably: <br>
&emsp;&emsp;	PreProcessDatasets takes a folder from the PPI_Datasets directory, and a set of desired sequence-based features, and computes per protein feature vectors, saving them in the format and with the filenames expected by the algorithms in methods by default. <br><br>
&emsp;&emsp;	RunTrainTest contains code for running train/test splits, and saving the models and results to folders <br><br>
&emsp;&emsp;	ProjectDataLoader contains functions that load the protein pairs from different datasets in the PPI_Datasets folder <br><br>
&emsp;&emsp;	PPIPUtils contains a variety of utility functions related to processing, downloading, and plotting data <br><br>
 <br>
BenchmarkEvalPaper contains the master script for the project.  Run the BenchmarkEvalPaper to download the necessary data for features, compute all features, train all models, and compute all results used in our paper.  By default, this script does not call the scripts we used to create to parse the data from other authors, or split the human data into groups, as we provided these files directly to ensure an exact comparison.  Additionally, running this file with its default settings will generate and save all of the more than 2,500 trained models used in our paper, which take up approximately 2 TB of disk space and can take 2 months to compute.  The number of models trained can be reduced by setting variables in the RunTest.py file to false. <br>
 <br>
 <br>
	 <br>
 <br>
<h2>Datasets from Other Authors</h2>
The following files are from other authors, used to compute datasets used in previous works (we do not own these files, credit goes to original authors, please cite them if comparing their datasets) <br>
From DeepPPI: boosting prediction of protein–protein interactions with deep neural networks. <br>
&emsp;&emsp;	PPI_Datasets/Du_Yeast/Supplementary S1.csv <br><br>
From Multifaceted protein-protein interaction prediction based on Siamese residual RCNN (https://github.com/muhaochen/seq_ppi.git) <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Chen/protein.actions.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Chen/protein.dictionary.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.01.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/	CeleganDrosophilaEcoli.actions.filtered.10.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.25.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.40.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.dictionary.filtered.tsv <br>
&emsp;&emsp;	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.dictionary.tsv <br><br>
From Predicting protein-protein interactions by fusing various Chou's pseudo components and using wavelet denoising approach (https://github.com/QUST-AIBBDRC/PPIs-WDSVM) <br>
&emsp;&emsp;	(note: these are text versions of the data stored in matlab files on Tian's github page) <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Tian/n_protein_a.txt <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Tian/n_protein_b.txt <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Tian/p_protein_a.txt <br>
&emsp;&emsp;	PPI_Datasets/Guo_Data_Yeast_Tian/p_protein_b.txt <br><br>
From iPPI-Esml: An ensemble classifier for identifying the interactions of proteins by incorporating their physicochemical properties and wavelet transforms into PseAAC <br>
&emsp;&emsp;	PPI_Datasets/Jia_Data_Yeast/1-s2.0-S0022519315001733-mmc1.pdf (coverted into NegativePairs.txt and PositivePairs.tsv) <br><br>
From Protein Interaction Network Reconstruction Through Ensemble Deep Learning With Attention Mechanism (https://github.com/liff33/EnAmDNN) <br>
&emsp;&emsp;	PPI_Datasets/Li_AD/ad_train.txt (AD/train.txt on github) <br>
&emsp;&emsp;	PPI_Datasets/Li_AD/ad_test.txt	(AD/test.txt on github) <br><br>
From Prediction of protein–protein interactions based on PseAA composition and hybrid feature selection <br>
&emsp;&emsp;	PPI_Datasets/Liu_Fruit_Fly/Liu_Interactions.tsv  (listed as 1-s2.0-S0006291X09001119-mmc1.txt online) <br><br>
From iPPI-PseAAC(CGR): Identify protein-protein interactions by incorporating chaos game representation into PseAAC <br>
&emsp;&emsp;	PPI_Datasets/Martin_H_Pylori/Jia_2019.docx (converted to Jia_2019.txt) (listed as 1-s2.0-S0022519318304971-mmc2.docx online) <br><br>
	 <br>
Additionally, we used, but do not include the following files due to spacing concerns in the repository (large size), but they would be needed to recompute some datasets from initially provided data <br>
From Large-scale prediction of human protein-protein interactions from amino acid sequence based on latent topic features (http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Data.htm)<br>
&emsp;&emsp;	(all files were converted into txt files in our file directory before using in our code) <br>
&emsp;&emsp;	PPI_Datasets/Pan_Human_Data/Supp-A.doc <br>
&emsp;&emsp;	PPI_Datasets/Pan_Human_Data/Supp-B.doc <br>
&emsp;&emsp;	PPI_Datasets/Pan_Human_Data/Supp-C.doc <br>
&emsp;&emsp;	PPI_Datasets/Pan_Human_Data/Supp-D.doc <br>
&emsp;&emsp;	PPI_Datasets/Pan_Human_Data/Supp-E.doc <br><br>
From Comparing two deep learning sequence-based models for protein-protein interaction prediction (https://gitlab.univ-nantes.fr/richoux-f/DeepPPI/tree/v1.tcbb) <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/double-medium_1166_train_mirror.txt <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/double-medium_1166_val_mirror.txt <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/test_double_mirror.txt <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/medium_1166_val_mirror.txt <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/medium_1166_train_mirror.txt <br>
&emsp;&emsp;	PPI_Datasets/Richoux_Human_Data/medium_1166_test_mirror.txt <br>
	 <br>
	 <br>
Additionally, the preprocessing folder contains some data from the AAIndex necessary to create certain sequence features, as well as Grantham, Schneider-Wrede, and blosum62 matrices, which also should be credited to their original authors. <br>
 <br>
 <br>
 
<h3>Notes</h3>
There was a bug in our code when computing local descriptor, which was fixed prior to the final algorithm in our paper that utilized the given feature, but after 3 other models used it.  Thus, results may vary slightly on those models. <br>
<br>
We have seeded as much of our code as possible by default, but, some neural network models, particularly those running LSTMs or GRUs, were not seeded through cuDNN, which would have slowed down the models to gain recreatability.  Thus, some network models may slightly vary when being recreated. <br>
