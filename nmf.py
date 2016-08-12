import numpy as np
from numpy.linalg import norm
from time import time
from sys import stdout
from numpy import *

def nmf(V,W,H,tol,maxiter):

     gradW = np.dot(W, np.dot(H, H.T)) - np.dot(V, H.T)
     gradH = np.dot(np.dot(W.T, W), H) - np.dot(W.T, V)
     initgrad = norm(np.r_[gradW, gradH.T])
     tolW = max(0.001,tol)*initgrad
     tolH = tolW

     for iter in xrange(1,maxiter):
          projnorm = norm(np.r_[gradW[np.logical_or(gradW<0, W>0)],
                                         gradH[np.logical_or(gradH<0, H>0)]])
          if projnorm < tol*initgrad: break

          (W, gradW, iterW) = nlssubprob(V.T,H.T,W.T,tolW,1000)
          W = W.T
          gradW = gradW.T

          if iterW==1: tolW = 0.1 * tolW

          (H,gradH,iterH) = nlssubprob(V,W,H,tolH,1000)
          if iterH==1: tolH = 0.1 * tolH

     print  'project gradient', norm(V - np.dot(W,H))


     return (W,H)

def nlssubprob(V,W,H,tol,maxiter):

     WtV = np.dot(W.T, V)
     WtW = np.dot(W.T, W)

     alpha = 1
     beta = 0.1
     for iter in xrange(1, maxiter):
          grad = np.dot(WtW, H) - WtV
          projgrad = norm(grad[np.logical_or(grad < 0, H >0)])
         # if projgrad < tol: break

          # search step size
          for inner_iter in xrange(1,20):
               Hn = H - alpha*grad
               Hn = np.where(Hn > 0, Hn, 0)
               d = Hn-H
               gradd = sum(grad * d)
               dQd = sum(np.dot(WtW,d) * d)
               suff_decr = 0.99*gradd + 0.5*dQd < 0
               if inner_iter == 1:
                    decr_alpha = not suff_decr
                    Hp = H
               if decr_alpha:
                    if suff_decr:
                         H = Hn
                         break
                    else:
                         alpha = alpha * beta
               else:
                  if not suff_decr or (Hp == Hn).all():
                       H = Hp
                       break
                  else:
                       alpha = alpha/beta
                       Hp = Hn

     return (H, grad, iter)


def nmf_just_grad(V,W,H,tol,maxiter):

     for iter in xrange(1, maxiter):

          WtV = np.dot(W.T, V)
          WtWH = np.dot(np.dot(W.T, W), H)

          VHt = np.dot(V, H.T)
          WHHt = np.dot(np.dot(W, H), H.T)

          for i in range(H.shape[0]):
              for j in range(H.shape[1]):
                   if WtWH[i][j]:
                         H[i][j] = H[i][j] * WtV[i][j] / WtWH[i][j]

          for i in range(W.shape[0]):
               for j in range(W.shape[1]):
                    if WHHt[i][j]:
                         W[i][j] = W[i][j] * VHt[i][j] / WHHt[i][j]

     print 'gradient descent', norm(V - np.dot(W,H))

     return (W, H)





f = open('ratings.dat').read().split('\n')

marks = []
for rate in f:
    try:
        one_mark = {}
        one_mark['user'] = int(rate.split('::')[0])
        one_mark['film'] = int(rate.split('::')[1])
        one_mark['rate'] = int(rate.split('::')[2])
        marks.append(one_mark)
    except: print rate

V = np.zeros((6040,3952))
for mark in marks:
    V[mark['user']-1][mark['film']-1] = mark['rate']

Num_of_factors = 100
Hinit = np.random.random((Num_of_factors, 3952))
Winit = np.random.random((6040, Num_of_factors))
print 'initial norm', norm(V - np.dot(Winit, Hinit))
W,H = nmf_just_grad(V,Winit,Hinit,0.000001,150)
W,H = nmf(V,Winit,Hinit,0.000001,150)