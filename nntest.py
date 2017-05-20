import numpy as np
import math
import csv
import sys

def sigmoid(x):
	# x=0.00001*x # ???
	return 1/(1+math.exp((-1)*x))

def Score(x):  # Calculating the score based on the formula from the assignment instruction.
	return x[0][0]+5.0*x[2][0]

D=47
H=80  # Hidden layer dimension, must be the same as the training part.
C=4

imat=np.loadtxt("imat.txt")
omat=np.loadtxt("omat.txt")

iarr=[]

res=[]
c=0

with open("test_normal.csv",'rb') as f:
	reader=csv.reader(f)
	for r in reader:
		iarr=[]
		if r[0]=='srch_id':
			continue
		if not len(r)==50:
			continue
		for i in range(0,len(r)):
			if i==0 or i==1 or i==7:
				continue
			if r[i]=="NULL" or r[i]=="":
				iarr.append(0.0)
			else:
				iarr.append(float(r[i]))
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
		score=Score(fovec)
		res.append((r[0],r[7],score))
		

res.sort(key=lambda tup:(tup[0],-tup[2]))
g=open("result.txt","w+")
g.write("SearchId,PropertyId\n")
for r in res:
	g.write("%s,%s\n"%(r[0],r[1]))
g.close()