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

def LMS(x,y):  # Mean square error
	return ((x[0][0]-y[0])*(x[0][0]-y[0])+(x[1][0]-y[1])*(x[1][0]-y[1])+(x[2][0]-y[2])*(x[2][0]-y[2])+(x[3][0]-y[3])*(x[3][0]-y[3]))/2.0

D=47
H=80  # Here H stands for the nodes amount in hidden layer, it is adjustable.
C=4

partition=[0,1,2,3,4,5,6,7,8,9]
vpartition=[0,1,2,3,4,5,6,7,8,9]


threshold=0.001  # The threshold for mean square error, it is adjustable.
lrate=0.1  # The learning rate for the network, it is adjustable.

# imat=np.random.rand(H,D)
# imat=imat/math.sqrt(float(D))

# omat=np.random.rand(C,H)
# omat=omat/math.sqrt(float(H))

# # imat=np.loadtxt("imat.txt")
# # omat=np.loadtxt("omat.txt")

# bpimat=imat
# bpomat=omat

while vpartition:
	imat=np.loadtxt("imat.txt")
	omat=np.loadtxt("omat.txt")
	bpimat=imat
	bpomat=omat
	random.shuffle(vpartition)
	vpoint=vpartition[0]
	print "now validating the training process with subset %d."%(vpoint)
	vpartition.remove(vpoint)
	verr=float('inf')
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
				lab=[0,0,0,0]
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
				lab[0]=float(r[51])
				lab[1]=1-lab[0]
				lab[2]=float(r[53])
				lab[3]=1-lab[2]
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
				# print "current error:"
				# print err
				# if err<threshold:
				# 	print "accepted, epoch:%d"%(c)
				# 	break
				cverr=0.0
				v_count=0
				with open("%d.csv"%(vpoint),'rb') as g:
					vreader=csv.reader(g)
					for vr in vreader:
						viarr=[]
						vlab=[0,0,0,0]
						if vr[0]=='srch_id':
							continue
						if not len(vr)==54:
							continue
						for vi in range(0,len(vr)):
							if vi==0 or vi==1 or vi==7 or vi==14 or vi==51 or vi==52 or vi==53:
								continue
							if vr[vi]=="NULL" or vr[vi]=="":
								viarr.append(0.0)
							else:
								viarr.append(float(vr[vi]))
						vlab[0]=float(vr[51])
						vlab[1]=1-vlab[0]
						vlab[2]=float(vr[53])
						vlab[3]=1-vlab[2]
						vivec=np.array(viarr)
						vivec.shape=(D,1)
						vhvec=imat.dot(vivec)
						vfhvec=vhvec
						for vi in range(0,len(vhvec)):
							vfhvec[vi][0]=sigmoid(vhvec[vi][0])
						vovec=omat.dot(vfhvec)
						vfovec=vovec
						for vi in range(0,len(vovec)):
							vfovec[vi][0]=sigmoid(vovec[vi][0])
						cverr=cverr+LMS(vfovec,vlab)
						v_count=v_count+1
						if cverr>verr:
							break
						# print "validated %d items, error sum is %f."%(v_count,cverr)
				print "training set error:"
				print err
				print "validating set error:"
				print cverr/v_count
				if cverr>verr:
					print "accepted on training set %d, epoch:%d"%(point,c)
					imat=bpimat
					omat=bpomat
					gc=gc+c
					gerr=gerr+(verr/v_count)
					break
				else:
					verr=cverr
				c=c+1
				delta_K=[]
				for k in range(0,C):
					if k<2:
						tmp_delta=(lab[k]-fovec[k][0])*sigmoid_dif(ovec[k][0])
					else:
						tmp_delta=25.0*(lab[k]-fovec[k][0])*sigmoid_dif(ovec[k][0])
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
	gerr=float(gerr/len(partition))
	print "finish validating on subset %d, with totally %d epoches and error sum is %f"%(vpoint,gc,gerr)
	np.savetxt("imat.txt",imat)
	np.savetxt("omat.txt",omat)

np.savetxt("imat.txt",imat)
np.savetxt("omat.txt",omat)