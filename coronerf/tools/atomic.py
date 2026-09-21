######################################################
### atomic physics helpers
######################################################

import numpy as np

def compute_ccoef(log_ne, log_temp, log_r, ion, line):
	'''
	Given (ne, T, r) triplet, compute ccoef for a given emission line for a given ion
	'''
	thetab = np.degrees(np.arccos(1/np.sqrt(3))) # dummy placeholder, update when doing full stokes

	ne = 10**log_ne
	temp = 10**log_temp
	r = 10**log_r
	height = r - 1
	ion.calc_rho_sym(ne,temp,height,thetab)

	# recompute line C_coeff and upper_level_alignment
	rho00_u = ion.rho[line.upper_level_index, 0]
	#rho20_u = ion.rho[line.upper_level_index, 2]
	#upper_level_alignment = rho20_u / rho00_u if rho00_u != 0 else 0.0

	upper_level_pop_frac = np.sqrt(2.0 * line.Jupp + 1.0) * rho00_u
	C_coeff = (line.hnu / (4.0 * np.pi)) * line.Einstein_A * upper_level_pop_frac * ion.totn

	return C_coeff
