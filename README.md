# benchmark_eval
MIT Licensed Code

This repistory contains all the code necessary to recreate models and results from Paper: Benchmark Evaluation of Protein–Protein Interaction Prediction Algorithms, 
as well as datasets produced by other authors that were compared in the paper.

To run the code, you will need the following python packages and software
Packages:
	goatools  (for gene ontology computations)
	lightgbm
	matplotlib
	numpy
	pywt
	rotation_forest (https://github.com/digital-idiot/RotationForest/tree/master/rotation_forest)
	scipy
	skimage
	sklearn
	thundersvm (optional, allows for running SVMs using gpu or multiple CPUs, but requires compiling code see below)
	torch
	libSVM (optional, allows running SVM's using libSVMs code.  Note that sklearn uses libsvm as their core svm library, so you can use that instead)
	
Software:
	Blast+ (Needed for computing PSSMs, should be accessible from command line.  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
	wget (Needed for downloading some large files, should be accesible from command line).
	thundersvm (Optional, needs to be downloaded and compiled to use thundersvm python library.  https://github.com/Xtra-Computing/thundersvm)




The code is split into the following folders:

The Methods Folder contain scripts creating the machine learning models used to predict protein interactions

The preprocess and pairwisePreprocess folders contain scripts related to creating sequence-based and annotation-based features used by the machine learning models

The PPI_Datasets folder contains data and scripts related to individual datasets, such as parsing the protein pairs from the original documents and downloading the sequence and annotations necessary for the different experiments in the paper.  Additionally, each datasets folder will contain all features describing proteins and protein pairs in given dataset.  Unlike most code in this project, code found in this folder is only usable for the individual dataset of interest.

The main folder has scripts related to running tests and loading the previously mentioned datasets.  Notably:
	PreProcessDatasets takes a folder from the PPI_Datasets directory, and a set of desired sequence-based features, and computes per protein feature vectors, saving them in the format and with the filenames expected by the algorithms in methods by default.
	RunTrainTest contains code for running train/test splits, and saving the models and results to folders
	ProjectDataLoader contains functions that load the protein pairs from different datasets in the PPI_Datasets folder
	PPIPUtils contains a variety of utility functions related to processing, downloading, and plotting data.

BenchmarkEvalPaper contains the master script for the project.  Run the BenchmarkEvalPaper to download the necessary data for features, compute all features, train all models, and compute all results used in our paper.  By default, this script does not call the scripts we used to create to parse the data from other authors, or split the human data into groups, as we provided these files directly to ensure an exact comparison.  Additionally, running this file with its default settings will generate and save all of the more than 2,500 trained models used in our paper, which take up approximately 2 TB of disk space and can take 2 months to compute.  The number of models trained can be reduced by setting variables in the RunTest.py file to false.


	

The following files are from other authors, used to compute datasets used in previous works (we do not own these files, credit goes to original authors, please cite them if comparing their datasets)
From DeepPPI: boosting prediction of protein–protein interactions with deep neural networks.
	PPI_Datasets/Du_Yeast/Supplementary S1.csv
From Multifaceted protein-protein interaction prediction based on Siamese residual RCNN (https://github.com/muhaochen/seq_ppi.git)
	PPI_Datasets/Guo_Data_Yeast_Chen/protein.actions.tsv
	PPI_Datasets/Guo_Data_Yeast_Chen/protein.dictionary.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.01.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/	CeleganDrosophilaEcoli.actions.filtered.10.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.25.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.filtered.40.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.actions.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.dictionary.filtered.tsv
	PPI_Datasets/Guo_MultiSpecies_Chen/CeleganDrosophilaEcoli.dictionary.tsv
From Predicting protein-protein interactions by fusing various Chou's pseudo components and using wavelet denoising approach (https://github.com/QUST-AIBBDRC/PPIs-WDSVM)
	(note: this are text version of the data stored in matlab files on Tian's github page)
	PPI_Datasets/Guo_Data_Yeast_Tian/n_protein_a.txt
	PPI_Datasets/Guo_Data_Yeast_Tian/n_protein_b.txt
	PPI_Datasets/Guo_Data_Yeast_Tian/p_protein_a.txt
	PPI_Datasets/Guo_Data_Yeast_Tian/p_protein_b.txt
From iPPI-Esml: An ensemble classifier for identifying the interactions of proteins by incorporating their physicochemical properties and wavelet transforms into PseAAC
	PPI_Datasets/Jia_Data_Yeast/1-s2.0-S0022519315001733-mmc1.pdf (coverted into NegativePairs.txt and PositivePairs.tsv)
From Protein Interaction Network Reconstruction Through Ensemble Deep Learning With Attention Mechanism (https://github.com/liff33/EnAmDNN)
	PPI_Datasets/Li_AD/ad_train.txt (AD/train.txt on github)
	PPI_Datasets/Li_AD/ad_test.txt	(AD/test.txt on github)
From Prediction of protein–protein interactions based on PseAA composition and hybrid feature selection
	PPI_Datasets/Liu_Fruit_Fly/Liu_Interactions.tsv  (listed as 1-s2.0-S0006291X09001119-mmc1.txt online)
From iPPI-PseAAC(CGR): Identify protein-protein interactions by incorporating chaos game representation into PseAAC
	PPI_Datasets/Martin_H_Pylori/Jia_2019.docx (converted to Jia_2019.txt) (listed as 1-s2.0-S0022519318304971-mmc2.docx online)
	
Additionally, we used, but do not include the following files due to spacing concerns in the repository (large size), but they would be needed to recompute some datasets from initially provided data
From Large-scale prediction of human protein-protein interactions from amino acid sequence based on latent topic features
	(all files were converted into txt files in our file directory before using in our code)
	PPI_Datasets/Pan_Human_Data/Supp-A.doc
	PPI_Datasets/Pan_Human_Data/Supp-B.doc
	PPI_Datasets/Pan_Human_Data/Supp-C.doc
	PPI_Datasets/Pan_Human_Data/Supp-D.doc
	PPI_Datasets/Pan_Human_Data/Supp-E.doc
From Comparing two deep learning sequence-based models for protein-protein interaction prediction (https://gitlab.univ-nantes.fr/richoux-f/DeepPPI/tree/v1.tcbb)
	PPI_Datasets/Richoux_Human_Data/double-medium_1166_train_mirror.txt
	PPI_Datasets/Richoux_Human_Data/double-medium_1166_val_mirror.txt
	PPI_Datasets/Richoux_Human_Data/test_double_mirror.txt
	PPI_Datasets/Richoux_Human_Data/medium_1166_val_mirror.txt
	PPI_Datasets/Richoux_Human_Data/medium_1166_train_mirror.txt
	PPI_Datasets/Richoux_Human_Data/medium_1166_test_mirror.txt
	
	
Additionally, the preprocessing folder contains some data from the AAIndex necessary to create certain sequence features, as well as Grantham, Schneider-Wrede, and blosum62 matrices, which also should be credited to their original authors.


Notes:
There was a bug in our code when computing local descriptor, which was fixed prior to the final algorithm in our paper that utilized the given feature, but after 3 other models used it.  Thus, results may vary slightly on those models.
We have seeded as much of our code as possible by default, but, some neural network models, particularly those running LSTMs or GRUs, were not seeded through cuDNN, which would have slowed down the models to gain recreatability.  Thus, some network models may slightly vary when being recreated.
