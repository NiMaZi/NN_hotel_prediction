# artificial neural network
# parameter initializing

import numpy as np
import math
import csv
import sys
import random

def sigmoid(x): # Here a logistic function is used as the kernel/trigger.
	try:
		return 1/(1+math.exp((-1)*x))
	except:
		if x>=0.0:
			return 1/(1+math.exp((-1)*float('inf')))
		else:
			return 1/(1+math.exp(float('inf')))

def sigmoid_dif(x):
	return sigmoid(x)*(1-sigmoid(x))

def LMS(x,y):  # least mean square error
	return ((x[0][0]-y[0])*(x[0][0]-y[0])+(x[1][0]-y[1])*(x[1][0]-y[1])+(x[2][0]-y[2])*(x[2][0]-y[2]))/2.0

D=47
H=64  # Here H stands for the nodes amount in hidden layer, it is adjustable.
C=3

partition=[0,1,2,3,4,5,6,7,8,9]
vpartition=[0,1,2,3,4,5,6,7,8,9]


threshold=0.001  # The threshold for least mean square error, it is adjustable.
lrate=0.1  # The learning rate for the network, it is adjustable.
nega_count=0

final_epoch=0
while vpartition:
	if len(vpartition)==10:
		imat=np.random.rand(H,D)
		imat=imat/math.sqrt(float(D))
		omat=np.random.rand(C,H)
		omat=omat/math.sqrt(float(H))
	else:
		imat=np.loadtxt("imat.txt")
		omat=np.loadtxt("omat.txt")
	bpimat=imat
	bpomat=omat
	random.shuffle(vpartition)
	vpoint=vpartition[0]
	vpartition.remove(vpoint)
	verr=float('inf')
	h_verr=verr
	p_count=0
	partition=[0,1,2,3,4,5,6,7,8,9]
	partition.remove(vpoint)
	random.shuffle(partition)
	gc=0
	gerr=0.0
	for epoch in range(0,len(partition)):
		point=partition[epoch] 
		c=0
		print "the current training set is subset %d"%(point)
		with open("%d.csv"%(point),'rb') as f:
			reader=csv.reader(f)
			for r in reader:
				iarr=[]
				lab=[0,0,0]
				if r[0]=='srch_id':
					continue
				if not len(r)==54:
					continue
				for i in range(0,len(r)):
					if i==0 or i==1 or i==7 or i==14 or i==51 or i==52 or i==53:
						continue
					if r[i]=="NULL" or r[i]=="":
						iarr.append(0.0)
					else:
						try:
							iarr.append(float(r[i]))
						except:
							print r[i]
							sys.exit(0)
				if r[51]=='0.0' and r[53]=='0.0':
					if nega_count>=33:
						nega_count=0
						lab=[1.0,0.0,0.0]
					else:
						nega_count=nega_count+1
						continue
				elif r[51]=='1.0' and r[53]=='1.0':
					lab=[0.0,0.0,1.0]
				else:
					lab=[0.0,1.0,0.0]
				ivec=np.array(iarr)
				ivec.shape=(D,1)
				hvec=imat.dot(ivec)
				fhvec=hvec
				for i in range(0,len(hvec)):
					fhvec[i][0]=sigmoid(hvec[i][0])
				ovec=omat.dot(fhvec)
				fovec=ovec
				for i in range(0,len(ovec)):
					fovec[i][0]=sigmoid(ovec[i][0])
				err=LMS(fovec,lab)
				if err<threshold:
					np.savetxt("imat.txt",imat)
					np.savetxt("omat.txt",omat)
					gc=gc+c
					break
				c=c+1
				delta_K=[]
				for k in range(0,C):
					tmp_delta=(lab[k]-fovec[k][0])*sigmoid_dif(ovec[k][0])
					delta_K.append(tmp_delta)
				delta_J=[]
				for j in range(0,H):
					tmp_delta=0.0
					for k in range(0,C):
						tmp_delta=tmp_delta+omat[k][j]*delta_K[k]
					tmp_delta=tmp_delta*sigmoid_dif(hvec[j][0])
					delta_J.append(tmp_delta)
				bpimat=imat
				bpomat=omat
				for j in range(0,H):
					for i in range(0,D):
						imat[j][i]=imat[j][i]+lrate*delta_J[j]*ivec[i][0]
				for k in range(0,C):
					for j in range(0,H):
						omat[k][j]=omat[k][j]+lrate*delta_K[k]*hvec[j][0]
	gc=gc+c
	gerr=float(gerr/len(partition))
	print "finish initializing on subset %d, with totally %d epoches."%(vpoint,gc)
	final_epoch=final_epoch+gc
	np.savetxt("imat.txt",imat)
	np.savetxt("omat.txt",omat)

print "finish initializing with %d epoches, result saved."%(final_epoch)
np.savetxt("imat.txt",imat)
np.savetxt("omat.txt",omat)