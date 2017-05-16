import numpy as np
import math
import csv
import sys

def sigmoid(x):
	try:
		return 1/(1+math.exp((-1)*x))
	except:
		if x>=0.0:
			return 1/(1+math.exp((-1)*float('inf')))
		else:
			return 1/(1+math.exp(float('inf')))

def sigmoid_dif(x):
	return sigmoid(x)*(1-sigmoid(x))

def LMS(x,y):
	return ((x[0][0]-y[0])*(x[0][0]-y[0])+(x[1][0]-y[1])*(x[1][0]-y[1])+(x[2][0]-y[2])*(x[2][0]-y[2])+(x[3][0]-y[3])*(x[3][0]-y[3]))/2.0

D=48
H=72
C=4

threshold=0.005
lrate=0.005

imat=np.random.rand(H,D)
imat=imat/math.sqrt(float(D))

omat=np.random.rand(C,H)
omat=omat/math.sqrt(float(H))

iarr=[]
lab=[0,0,0,0]

c=0

with open("training_set_VU_DM_2014.csv",'rb') as f:
	reader=csv.reader(f)
	for r in reader:
		iarr=[]
		lab=[0,0,0,0]
		if r[0]=='srch_id':
			continue
		for i in range(0,len(r)):
			if i==0 or i==1 or i==14 or i==51 or i==52 or i==53:
				continue
			if r[i]=="NULL" or r[i]=="":
				iarr.append(0.0)
			else:
				try:
					iarr.append(float(r[i]))
				except:
					print r[i]
					sys.exit(0)
		lab[0]=float(r[51])
		lab[1]=1-lab[0]
		lab[2]=float(r[53])
		lab[3]=1-lab[2]
		c=c+1
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
		print "current error:"
		print err
		if err<threshold:
			print "accepted, epoch:%d"%(c)
			break
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
		for j in range(0,H):
			for i in range(0,D):
				imat[j][i]=imat[j][i]+lrate*delta_J[j]*ivec[i][0]
		for k in range(0,C):
			for j in range(0,H):
				omat[k][j]=omat[k][j]+lrate*delta_K[k]*hvec[j][0]

np.savetxt("imat.txt",imat)
np.savetxt("omat.txt",omat)