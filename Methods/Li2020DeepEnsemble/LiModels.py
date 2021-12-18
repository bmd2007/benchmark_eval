#Based on paper Protein Interaction Network Reconstruction Through Ensemble Deep Learning With Attention Mechanism by Li, Zhu, Ling, and Liu

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable




class ModelLiDeep(nn.Module):
	def __init__(self,vector_size,conv_size,conv_repeats,kernel_size,num_heads, poolSize, numLinLayers,seed=1):
		
		super().__init__()
		torch.manual_seed(seed)
		#size of incoming feature vectors
		self.vector_size = vector_size
		#size of data when going into convolution, must be equal to vector size to match output of attention for multiplication
		self.conv_size = conv_size
		#size of linear layer, to square features
		self.linear_size = conv_size**2
		#number of convolution repeats
		self.conv_repeats = conv_repeats
		#number of heads for attention
		self.num_heads = num_heads
		#embedding dim for attention, must be equal to conv_size for multiplication to work
		self.embed_dim = conv_size
		#size of kernel for max pool and average pool
		self.poolSize = poolSize
		#number of linear layers to use at end
		self.numLinLayers = numLinLayers
		
		self.activate = nn.ReLU()
		self.attention = nn.MultiheadAttention(self.embed_dim,num_heads,batch_first=True)
		self.linear0 = nn.Linear(self.vector_size,self.linear_size)
		self.convModules = nn.ModuleList()
		self.batchNorms = nn.ModuleList()
		for i in range(0,self.conv_repeats):
			self.convModules.append(nn.Conv2d(1,1,kernel_size,1,kernel_size-2))
			self.batchNorms.append(nn.BatchNorm2d(1))
		
		self.convOut = nn.Conv2d(1,1,kernel_size,1,kernel_size-2)
		
		#pooling prior to first concat
		self.maxPool = nn.MaxPool2d(poolSize,1)
		self.avgPool = nn.AvgPool2d(poolSize,1)
		
		#A weight for multiplication before final concat
		self.AWeight = nn.Parameter(torch.ones(((self.conv_size-self.poolSize+1)**2)*2).unsqueeze(0))
			
		#cosine similarity before final cat
		self.cosSim = nn.CosineSimilarity()
		
		#linear layers to transition to softmax output
		#size of data going into linear layers
		totalSize = (((self.conv_size-self.poolSize+1)**2)*2)*3 + 1
		#output size
		outSize = 2
		self.linearLayers = nn.ModuleList()
		self.batchNorms2 = nn.ModuleList()
		for i in range(0,self.numLinLayers):
			#interpolate from total size to outputsize
			prevSize = int(totalSize - (totalSize-outSize)*(i/self.numLinLayers))
			newSize = int(totalSize - (totalSize-outSize)*((i+1)/self.numLinLayers))
			if i+1 == self.numLinLayers:
				newSize=outSize
			self.linearLayers.append(nn.Linear(prevSize,newSize))
			self.batchNorms2.append(nn.BatchNorm1d(newSize))
		
	def process_protein(self,prot):
		#proteins are run through a CNN module
		#CNN requires proteins to be of size X**D where D is the CNN dimension
		#since D is undefined, we will use a linear layer to ensure that we can match X**D size whenver needed
		#So we will use a linear layer to max proteins size X
		pLinear = self.linear0(prot)
		
		#run CNN conv_repeats times
		#each CNN is represented as  relu(batch(conv(input))))
		#with a final CNN for the output
		
		#resize data
		pConv = pLinear.reshape(-1,1,self.conv_size,self.conv_size)
		for i in range(0,self.conv_repeats):
			pConv = self.convModules[i](pConv)
			pConv = self.batchNorms[i](pConv)
			pConv = self.activate(pConv)
		
		#run output convolution and remove the extra dimension (dim1)
		pOut = self.convOut(pConv).squeeze(1)
		
		return pOut
		
	def forward(self, data):
		featshape = data.shape[1]
		#get encoded proteins data
		p1Lst = data[:,:featshape//2]
		p2Lst = data[:,featshape//2:]
		
		#print(p1Lst.shape,p2Lst.shape)
		
		p1Processed = self.process_protein(p1Lst)
		p2Processed = self.process_protein(p2Lst)
		
		#print(p1Processed.shape,p2Processed.shape)
		#p1Processed and p2Processed are conv_size x conv_size matrices = vector_size x vector_size
		
		
		#run attention
		s1 = self.attention(p1Processed, p2Processed, p1Processed,need_weights=False)[0]
		s2 = self.attention(p2Processed, p1Processed, p2Processed,need_weights=False)[0]
		
		#print(s1.shape,s2.shape)
		#s1 and s2 are conv_size x embed_size = vector_size x vector_size
		
		
		#multiply attention values by original protein values
		
		s1 = torch.mul(p1Processed,s1)
		s2 = torch.mul(p2Processed,s2)
		
		#print(s1.shape,s2.shape)
		
		#do max pooling and average pooling over the attention embed dimension, also reduce to 1d
		
		s1max = self.maxPool(s1).view(-1,(self.conv_size-self.poolSize+1)**2)
		s1avg = self.avgPool(s1).view(-1,(self.conv_size-self.poolSize+1)**2)
		s2max = self.maxPool(s2).view(-1,(self.conv_size-self.poolSize+1)**2)
		s2avg = self.avgPool(s2).view(-1,(self.conv_size-self.poolSize+1)**2)
		
		#concat
		s1final = torch.cat((s1avg,s1max),dim=1)
		s2final = torch.cat((s2avg,s2max),dim=1)
		
		#create a vector of 4 values
		#cosine sim (s1final,s2final)
		#s1final
		#s2final
		#s1final * A * s2final.  A is a weight, but not clearly defined in any way. . .  Using nn.Parameter
		product = s1final * self.AWeight * s2final
		cos = self.cosSim(s1final,s2final).view(-1,1)
		
		finalCat = torch.cat((s1final,s2final,product,cos),dim=1)
		
		for i in range(0,self.numLinLayers):
			finalCat = self.linearLayers[i](finalCat)
			finalCat = self.batchNorms2[i](finalCat)
			if i != self.numLinLayers-1:
				finalCat = self.activate(finalCat)
		
		return finalCat
		
class EnsembleNetwork(nn.Module):
	def __init__(self,seed=1):
		super().__init__()
		torch.manual_seed(seed)
		self.l1 = nn.Linear(16, 16)
		self.l2 = nn.Linear(16, 2)
		self.l3 = nn.Linear(2, 2)
		self.bn1 = nn.BatchNorm1d(16)
		self.bn2 = nn.BatchNorm1d(2)
		self.activate = nn.ReLU()
	def forward(self, x):
		x = self.l1(x)
		x = self.activate(x)
		x = self.bn1(x)
		x = self.activate(x)
		x = self.l2(x)
		x = self.bn2(x)
		x = self.l3(x)
		return x