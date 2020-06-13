import os
import numpy as np
import matlab.engine

threads   = 8
eng = matlab.engine.start_matlab()
# eng = future.result()
# eng.parpool('local',threads,background=True)

dfS = np.random.randn(10,50)

tol = 1e-16
max_iter = 1e4

eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)
P = matlab.double(dfS.T.tolist())
_, lam, delta, _ = eng.WolfeAlg(P,tol,tol,tol,max_iter,nargout=4)

lam = np.asarray(lam).flatten()