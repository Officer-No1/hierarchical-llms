import numpy
import ctypes

name = "FORCESNLPsolver"
requires_callback = True
lib = "lib/libFORCESNLPsolver.so"
lib_static = "lib/libFORCESNLPsolver.a"
c_header = "include/FORCESNLPsolver.h"
nstages = 2

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  2,   1),    2),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, ( 14,   1),   14),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, ( 58,   1),   58)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x1"                  , ""               , ctypes.c_double, numpy.float64,     (  7,),    7),
 ("x2"                  , ""               , ctypes.c_double, numpy.float64,     (  7,),    7)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(7, 2, 3, 29, 7, 7, 3, 0), 
	(7, 2, 3, 29, 7, 7, 3, 0)
]