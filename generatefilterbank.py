#!/usr/bin/python
import pywt, math, cmath
import numpy as np
pi=math.pi

def wavelet_value(k,T):
	value=0.5*(1+math.cos(pi*k/(T+1)))*math.sqrt(2-math.cos(pi*k/(T+1)))	#Daubechies function
	return value

def dec_low(length, n1, n2, P, S, T):
	H0_k=[]
	for i in range(length/2):
		if (i<=P):
			H0_k.append(1)
		elif (i<=P+T):
			H0_k.append(wavelet_value(i-P,T))
		else:
			H0_k.append(0)
	reversed_h0=H0_k[::-1]
	H0_k=H0_k+reversed_h0
	return H0_k

def dec_high(length, n1, n2, P, S, T):
	H1_k=[]
	for i in range(length/2):
		if (i<=P):
			H1_k.append(0)
		elif (i<=P+T):
			H1_k.append(wavelet_value(T+1+P-i,T))
		else:
			H1_k.append(1)
	reversed_h1=H1_k[::-1]
	H1_k=H1_k+reversed_h1
	return H1_k

def obtain_AlphaBeta(Q, r):
	beta=2/float(Q+1)
	alpha=1-(beta/float(r))
	return alpha,beta

# Call the following function from the main program 
# N0 and N1 are calculated here. We can also pass them as arguments
def passthroughfilters(X, Q, r):
	n=len(X)
	alpha,beta=obtain_AlphaBeta(Q,r)
	n0=2*int(round((alpha/2)*n))
	n1=2*int(round((beta/2)*n))
	P=int(round(n-n1)/2)
	S=int(round(n-n0)/2)
	T=int(round(((n0+n1-n)/2)-1))
	h0=dec_low(n,n0,n1,P,S,T)
	h1=dec_high(n,n0,n1,P,S,T)
	v0=[]
	v1=[]
	for i in range(n):
		v0.append(X[i]*h0[i])
		v1.append(X[i]*h1[i])
	return v0, v1