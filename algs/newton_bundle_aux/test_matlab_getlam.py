import os
import numpy as np
import matlab.engine

tol = 1e-16
max_iter = 1e4

threads   = 8
eng = matlab.engine.start_matlab()
eng.parpool('local',threads)
eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)

dfS = np.random.randn(10,50)
sub_vec = np.random.randn(50)

P = matlab.double(dfS.T.tolist())
sub_vec = matlab.double(sub_vec.tolist())
lam, delta = eng.WolfeAlg(P,tol,tol,tol,max_iter,sub_vec,nargout=2)
lam = np.asarray(lam).squeeze()
delta = np.asarray(delta).squeeze()