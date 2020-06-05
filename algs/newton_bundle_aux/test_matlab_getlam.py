import os
import numpy as np
import matlab.engine

threads   = 8
future = matlab.engine.start_matlab(background=True)
eng = future.result()
eng.parpool('local',threads,background=True)

eng.addpath(os.getcwd() + '/algs/newton_bundle_aux', nargout=0)
a = matlab.double(A_list.tolist(), is_complex=True)
r_gap = np.asarray(eng.radius_gap(a)[0])