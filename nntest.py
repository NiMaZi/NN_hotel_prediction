import numpy as np
import math
import csv
import sys

def sigmoid(x):
	x=0.00001*x # ???
	try:
		return 1/(1+math.exp((-1)*x))
	except:
		if x>=0.0:
			return 1/(1+math.exp((-1)*float('inf')))
		else:
			return 1/(1+math.exp(float('inf')))

def Score(x):
	return x[0][0]-x[1][0]+5.0*x[2][0]-5.0*x[3][0]

D=48
H=72
C=4

imat=np.loadtxt("imat.txt")
omat=np.loadtxt("omat.txt")

iarr=[]

res=[]
c=0

with open("test_set_VU_DM_2014.csv",'rb') as f:
	reader=csv.reader(f)
	for r in reader:
		iarr=[]
		if r[0]=='srch_id':
			continue
		for i in range(0,len(r)):
			if i==0 or i==1:
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
		c=c+1
		# if c>=5000:
		# 	break

res.sort(key=lambda tup:(tup[0],-tup[2]))
g=open("result.txt","w+")
g.write("SearchId ,PropertyId\n")
for r in res:
	g.write("%s, %s\n"%(r[0],r[1]))
g.close()