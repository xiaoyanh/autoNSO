tol = 1e-15
m_params = {'MSK_DPAR_INTPNT_QO_TOL_DFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_INFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_MU_RED': tol,
            'MSK_DPAR_INTPNT_QO_TOL_NEAR_REL': 10,
            'MSK_DPAR_INTPNT_QO_TOL_PFEAS': tol,
            'MSK_DPAR_INTPNT_QO_TOL_REL_GAP': tol,
            }
# m_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_INFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_MU_RED': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL': 10,
#             'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
#             'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
#             # 'MSK_IPAR_OPTIMIZER': 'CONIC',
#             # 'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1e3),
#             }


# GUROBI
g_params = {'BarConvTol': 1e-10,
            'BarQCPConvTol': 1e-10,
            'FeasibilityTol': 1e-9,
            'OptimalityTol': 1e-9,}