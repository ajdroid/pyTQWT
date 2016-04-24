from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import scipy, traceback, warnings, sys
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from scipy.ndimage import imread

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

im1 = imread('data/a01_1.tif')
im2 = imread('data/a01_2.tif')
#!/usr/bin/python
import math, cmath
import numpy as np
pi=math.pi

def AFB(X, N0, N1):
    N = len(X)
    assert N%2 == 0
    assert N0%2==0 and N1%2 ==0 
    assert N0+N1 > N
    P = int((N - N1)/2)
    T = int((N0 + N1 - N)/2 - 1)
    S = int((N - N0)/2)
    V0 = np.zeros((N0,), dtype=complex)
    V1 = np.zeros((N1,), dtype=complex)
    k = np.arange(1,T+1, dtype=int)
    th = np.zeros((T+1,), dtype=float)
    th[k] = 0.5*(1 + np.cos(k*pi/(T + 1)))*np.sqrt(2-np.cos(k*pi/(T + 1)))
    # Low pass sub-band
    V0[0] = X[0]
    V0[1:P+1] = X[1:P+1]
    V0[N0-P:N0] = X[N-P:N]
    V0[P+1:P+T+1] = X[P+1:P+T+1]*th[1:T+1]
    V0[N0-P-T:N0-P] = X[N-P-T:N-P]*th[1:T+1]
    V0[N0/2] = 0
    # High pass sub-band
    V1[0] = 0
    V1[1:T+1] = X[P+1:P+T+1]*th[T:0:-1]
    V1[N1-T:N1] = X[N-P-T:N-P]*th[1:T+1]
    V1[T+1:T+S+1] = X[P+T+1:P+T+S+1]
    V1[N1-T-S:N1-T] = X[N-P-T-S:N-P-T]
    V1[N1/2] = X[N/2]
    v0 = np.fft.ifft(V0)
    v1 = np.fft.ifft(V1)    
    return v0, v1

def SFB(V0, V1, N):
    N0 = len(V0)
    N1 = len(V1)
    assert N%2 == 0
    assert N0%2==0 and N1%2 ==0 
    assert N0+N1 > N
    P = int((N - N1)/2)
    T = int((N0 + N1 - N)/2 - 1)
    S = int((N - N0)/2)
    Y0 = np.zeros((N,), dtype=complex)
    Y1 = np.zeros((N,), dtype=complex)
    k = np.arange(1,T+1, dtype=int)
    th = np.zeros((T+1,), dtype=float)
    th[k] = 0.5*(1 + np.cos(k*pi/(T + 1)))*np.sqrt(2-np.cos(k*pi/(T + 1)))
    # Low pass sub-band
    Y0[0] = V0[0]
    Y0[1:P+1] = V0[1:P+1]
    Y0[N-P:N] = V0[N0-P:N0]
    Y0[P+1:P+T+1] = V0[P+1:P+T+1]*th[1:T+1]
    Y0[N-P-T:N-P] = V0[N0-P-T:N0-P]*th[T:0:-1]
    Y0[N/2] = 0
    # High pass sub-band
    Y1[0] = 0 # already 0 though
    Y1[1+P:T+P+1] = V1[1:T+1]*th[T:0:-1]
    Y1[N-T-P:N-P] = V1[N1-T:N1]*th[1:T+1]
    Y1[T+P+1:T+P+S+1] = V1[T+1:T+S+1]
    Y1[N-T-P-S:N-T-P] = V1[N1-T-S:N1-T]
    Y1[N/2] = V1[N1/2]
    Y = Y1 + Y0
    y = np.real(np.fft.ifft(Y))
    return y

def imAFB(im, n0, n1):
    nrows, ncols = im.shape
    imL = np.zeros((n0, ncols), dtype = float)
    imH = np.zeros((n1, ncols), dtype = float)
    imLL = np.zeros((n0, n0), dtype = float)
    imLH = np.zeros((n0, n1), dtype = float)
    imHL = np.zeros((n1, n0), dtype = float)
    imHH = np.zeros((n1, n1), dtype = float)

    # Filter columnwise first

    for icol in range(ncols):
        x = im[:,icol] 
        X = np.fft.fft(x)
        imL[:, icol], imH[:, icol] = AFB(X, n0, n1)

    # Filter the common part of the rows    

    for irow in range(min(n0,n1)):  
        x = imL[irow, :] 
        X = np.fft.fft(x)
        imLL[irow,:], imLH[irow,:] = AFB(X, n0, n1)
        x = imH[irow, :] 
        X = np.fft.fft(x)
        imHL[irow,:], imHH[irow,:] = AFB(X, n0, n1)


    # Filter remaining rows
    
    for irow in range(min(n0,n1), max(n0,n1)-min(n0,n1)):  
        x = imL[irow, :] 
        X = np.fft.fft(x)
        imLL[irow,:], imLH[irow, :] = AFB(X, n0, n1)

    return imLL, imLH, imHL, imHH

def imSFB(imLL, imLH, imHL, imHH, nrows=256, ncols=256):
    n0 = imLL.shape[0]
    n1 = imHH.shape[0]
    imL = np.zeros((n0, ncols), dtype = float)
    imH = np.zeros((n1, ncols), dtype = float)
    im = np.zeros((nrows, ncols), dtype = float)
    # Synth to imH
    for i in range(min(n0, n1)):
        imH[i,:] = SFB(np.fft.fft(imHL[i,:]), np.fft.fft(imHH[i,:]), ncols)
        imL[i,:] = SFB(np.fft.fft(imLL[i,:]), np.fft.fft(imLH[i,:]), ncols)
    
    # Synth to imL
    for i in range(min(n0,n1), max(n0,n1)-min(n0,n1)):
        imL[i,:] = np.real(SFB(np.fft.fft(imLL[i,:]), np.fft.fft(imLH[i,:]), ncols))

    # Synthesize image
    for i in range(ncols):
        im[:,i] = np.real(SFB(np.fft.fft(imL[:,i]), np.fft.fft(imH[:,i]), nrows))
                      
    return im

def TQWTfuse(im1, im2, n0=256, n1=256):
    imLL1, imLH1, imHL1, imHH1 = imAFB(im1, n0, n1)
    imLL2, imLH2, imHL2, imHH2 = imAFB(im2, n0, n1)
    
    imLL = (imLL1+imLL2)/2
    imLH = np.maximum(imLH1, imLH2)
    imHL = np.maximum(imHL1, imHL2)
    imHH = np.maximum(imHH1, imHH2)
    
    imfuse = imSFB(imLL, imLH, imHL, imHH)
    return imfuse

if __name__ =='__main__':
	n0 = n1 = 140
	imFuse = TQWTfuse(im1, im2, 140, 140)
	plt.figure()
	plt.imshow(imFuse, cmap='gray')
	plt.figure()
	plt.imshow(im1, cmap='gray')
	plt.figure()
	plt.imshow(im2, cmap='gray')
	plt.show()
	# imFuse = 256*(im1Rec-im1Rec.min())/(im1Rec.max()-im1Rec.min())