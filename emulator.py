import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.linalg import cholesky, toeplitz, solve_triangular
from math import comb
from operator import itemgetter
from collections import defaultdict
from itertools import product

from LISG import construct_psg
from kernels import maternKernel

def main():
	'''
	Implements 'fast prediciton' algorithm, an extension to Algorithm 1 in
	Plumlee 2014, 'Fast prediction of deterministic functions using sparse grid
	experimental designs,' generalised to LISGs.
	'''

	# Sparse grid parameters.
	dim = 50
	level = 10
	penalty = np.arange(0, dim+1, dtype=np.float64)
	penalty = np.array([1,1], dtype=np.float64)
	penalty = np.array(
		[np.ceil(np.log2(i)) for i in range(1,dim+2)],
		dtype=np.float64
	)

	# Emulator parameters.
	nu_list = np.repeat(0.5,dim+1)
	lengthscale_list = list(2**np.array(penalty[:dim]))
	sigma_list = np.ones(dim+1)

	# Define test function.													 
	def func(x):
		'''
		Test function used in place of simulator.
		Parameters:
			x: List of Floats, point in [-0.5,0.5]^dim.
		Returns:
			Float.
		'''
		return 1.0 #x[0]**2 + x[1]**2

	# Plotting parameters.
	resolution = 200
	
	# Error estimate parameter.
	no_mc_points = 100

	# Plot function in 1 or 2D.
	
	if dim == 1 or dim == 2:
		plot_interploant(
			dim,
			level,
			penalty,
			nu_list,
			lengthscale_list,
			sigma_list,
			resolution,
			func
		)

	error, N  = L2_error(
				dim,
				level,
				penalty,
				nu_list,
				lengthscale_list,
				sigma_list,
				no_mc_points,
				func
			)
	
	print('L2 Error')
	print(error)
	print('Num points')
	print(N)

	

class MaternPSGEmulator:
	'''
	Respresents Emulator for functions parameterised over [-0.5,0.5]^dim at
	points arranged in a penalised sparse grid using a separable matern kernel.
	'''

	def __init__(
		self,
		dim,
		level,
		penalty,
		func,
		nu_list,
		lengthscale_list,
		sigma_list
	):
		'''
		Attributes
		----------
		dim : int,
			Dimension.
		level: int
			Level of construction of the sparse grid.
		penalty : list of ints
			Penalty in each dimension.
		func : function
			Objective function for sampling.
		nu_list : list of floats
			Smoothness in each dimension.
		lengthscale_list : list of Floats
			Lengthscale in each dimension.
		sigma_list : list of Floats
			Standard deviation in each direction.
		kernel_list : list of maternKernel objects
			Represents separable Matern kernel.
		L_dict : dictionary
			Stores Cholesky matrices corresponding to multi-indices.
		dict : dictionary
			For each point in the correpsonding sparse grid stores function
			value and weight.
		'''

		# Store variables in object.
		self.dim = dim
		self.func = func
		self.nu_list = nu_list
		self.lengthscale_list = lengthscale_list
		self.level = level
		self.penalty = penalty
		self.sigma_list = sigma_list
		
		# Initialise kernels.
		self.construct_kernels()

		# Construct penalised sparse grid dictionary.
		self.construct_psg_dict()


	def construct_kernels(self):
		'''
		Create and store list of kernel objects for each dimension.
		'''
		kernelObject_list = []
		for i in range(self.dim):
			kernelObject_list.append(maternKernel(
				self.nu_list[i],
				self.lengthscale_list[i],
				self.sigma_list[i]
			))
		self.kernel_list = kernelObject_list
		# Reset matrix inverse dictionary. [MAKE THIS NICER]
		self.L_dict = {}
		
	def construct_psg_dict(self):
		'''
		Returns dictionary with sparse grid points as keys and 2-Lists as
		values, where the first entry of list is func evaluated at the grid
		point.
		'''
		print('Building Lengthscale-Informed Sparse Grid...')

		keys = construct_psg(self.level, self.penalty, self.dim)
		self.dict = {tuple(key): [self.func(key),0] for key in keys}
		# Reset matrix inverse dictionary. [MAKE THIS NICER]
		self.L_dict = {}

	def contribution_scaling(self, multi_index, ess_dim):
		'''
		Scales the contribution of each lengthscale-informed sparse grid
		component for inversion calculation.
		Parameters:
			multi_index : list 
				List of ess_dim-many non-negative integers.
			ess_dim : int
				Essential dimension.
		Returns:
			Float, scales contribution from given multi-index's sub-system.
		'''
		penalty = self.penalty[:self.dim]
		# Find list of penalties corresponding to the zero indices AND
		# find the sum total of all a_j + p_j for all non-zero indices.
		p_v = []
		ap_total = 0
		for j in range(ess_dim):
			if multi_index[j] == 0:
				p_v.append(penalty[j])
			else:
				ap_total += multi_index[j] + penalty[j]

		k = len(p_v)

		# Initialise loop parameters.
		m = 0
		total = 0

		# Sum up to sum(p), however all contributions m > level - ap_total are
		# zero due to binomial.
		m_max = min(sum(p_v), self.level - ap_total)

		# Sum from m = 0 to sum(p_v).
		while m <= m_max:
			# Count number of affected multi-indices when projected into
			# isotropic space.
			index_sum = count_multi_indices(m, k, p_v)
			if index_sum == 0:
				return total
			# Calculate contribution to sum.
			total += index_sum*(-1)**(self.level-m-ap_total)*comb(
				int(ess_dim-1),
				int(self.level-m-ap_total)
			)
			# Increment m.
			m += 1

		return total
		
	def ordered_index_to_design(self, multi_index, ess_dim):
		'''
		Order points in cartesian products to match order of Kronecker system.

		Parameters:
			multi_index: List
				ess-dim-many non-negative integers.
			ess_dim: int
				Esssential dimension of design
		Returns:
			List of grid-points , ordered with respect to Kronecker linear
			system.
		'''
		# Initialise list of 1D designs.
		design_list = []
		# Iterate through essential dimensions.
		for j in range(ess_dim):
			exponent = max(1, multi_index[j] + 1)
			design_list.append(np.arange(
				-0.5 + 0.5**exponent,
				0.5,
				0.5**exponent
			))
		# Iterate through remaining dimensions.
		for j in range(ess_dim, self.dim):
			design_list.append(np.zeros(1))

		return list(product(*design_list))	

	def calculate_weights(self):
		'''
		Calculate weight vector. Adaptation of 'Algorithm 1' in Plumlee 2014. 
		'''
		print('Calculating weights...')
		
		# Find essential dimension.
		# If the penalty is large enough such that there will only ever be one
		# point in the final dimension...
		if self.penalty[self.dim-1] > self.level:
			# Find first dimension, ess_dim, for which 
			# penalty[ess_dim] > level.
			ess_dim = 0
			while self.penalty[ess_dim] <= self.level:
				ess_dim += 1
		else: 
			ess_dim = self.dim

		# Create list of multi-indices required, P(L):
		multi_index_list = []
		
		for l in range(self.level+1):
			#multi_index_list += list(to_sum_k_rec(self.dim, l))
			multi_index_list += list(to_penalised_sum_k(
										ess_dim,
										l,
										self.penalty
									))

		# Iterate over all multi-indices.
		for multi_index in multi_index_list:
			if True:
				# Find subset of sparse grid points corresponding to
				# multi-index, arranged in Kronecker order.
				keys = self.ordered_index_to_design(multi_index, ess_dim)
				# Initilaise inverse Kronecker product matrix.
				L = np.ones(1)
				# Iterate over each index corresponding to a dimension.
				for j, index in enumerate(multi_index):
					# Find penalty in dimension j.
					penalty = self.penalty[j]
					# Check if 1D covar matrix Cholesky decomposition is stored
					# in matrix dictionary.
					if (j, max(0,index)) in self.L_dict:
						L_component = self.L_dict[(j, max(0,index))]
					else:
						# If inverse 1D covar matrix is not stored, calculate
						# using Cholesky decomposition.
						# Determine size of matrix.
						n = int(2**(index+1) - 1)
						# Determine 1D design.
						one_d_array = np.arange(
							-0.5+0.5**(index+1),
							0.5,
							0.5**(index+1)
						)
						# Initalise first row of Toeplitz matrix.
						column = [1]
						# Iterate through column.
						for i in range(1,n):
							column.append(self.kernel_list[j](
								one_d_array[i],
								one_d_array[0]
							))
						# Construct matrix.
						covar_mat = toeplitz(column,column)
						# Cholesky decomposition.
						L_component = cholesky(covar_mat, lower = True)
	
						# Store matrix in dictionary.
						self.L_dict[(j, index)] = L_component
	
					# Successively build Kronecker product of Cholesky matrices
					L = np.kron(L, L_component)
	
				# Solve triangular systems.
				# [THINK ABOUT KEYS][SPARSE PRODUCT]
	
				data, old_weights = np.array(itemgetter(*keys)(self.dict)).T

				if isinstance(data, float):
					data, old_weights = [data], [old_weights]
				b = solve_triangular(
					L,
					np.array(data),
					lower = True
				)
				weight_update = solve_triangular(
					L.T,
					b
				)
				# Calculate weight update.
				new_weights = old_weights + \
					self.contribution_scaling(multi_index, ess_dim) * \
					weight_update

				# Store new weights in dictionary.
				self.dict.update(list(zip(keys, list(zip(data, new_weights)))))
		print('Done.')


	def __call__(self, arg):
		'''
		Returns value (float) of interpolant at given point (array-like) in 
		domain
		'''

		# Evaluate separable Matern kernel at each point in sparse grid for
		# given argument, then scale by weighting.
		total = 0
		for point in self.dict.keys():
			log_subtotal = 0
			for j, kernel in enumerate(self.kernel_list):
				log_subtotal += np.log(kernel(point[j], arg[j]))
			total += self.dict[point][1] * np.exp(log_subtotal)

		return total

def count_multi_indices(m, k, sub_penalty):
	'''
	Count the number of multi-indices l in N_0^k such that |l| = m and
	l_i <= sub_penalty_i for all i.
	'''
	p = sub_penalty[:k]
	# dp[i][s] = number of ways to assign first i components to sum to s.
	dp = [defaultdict(int) for _ in range(k+1)]
	dp[0][0] = 1 # base case: zero sum with zero components.

	for i in range(1, k + 1):
		for s in range(m + 1):
			for l_i in range(int(min(p[i - 1], s) + 1)):
				dp[i][s] += dp[i - 1][s - l_i]

	return dp[k][m]

def to_penalised_sum_k(dim, level, penalty, prefix=(), j=0):
	'''
	Yields all dim-dimensional non-negative integer tuples l such that:
	sum(l_j + penalty_i * 1_{l_j > 0}) == level
	'''
	if j == dim:
		if level == 0:
			yield prefix
		return

	max_val = int(level)  # conservative upper bound
	for a in range(max_val + 1):
		p = penalty[j] if a > 0 else 0
		total = a + p
		if total <= level:
			yield from to_penalised_sum_k(
				dim,
				level - total,
				penalty,
				prefix + (a,),
				j + 1
			)

def plot_interploant(
		dim,
		level,
		penalty,
		nu_list,
		lengthscale_list,
		sigma_list,
		resolution,
		func
	):
	'''
	Creates MaternPSGEmulator object and plots inteprolant for 1 and 2
	dimensions.
	
	Parameters
	----------
	dim : int
		Dimension of domain of funciton.
	level : int
		Level of construction of sparse grid.
	penalty : list of ints
		Penalty parameter in each dimension.
	nu_list : list of floats
		Smoothness in each dimension.
	lengthscale_list : list of floats
		Lengthscale in each dimension.
	sigma_list : list of floats
		Standard deviation in each dimension.
	resolution : int
		Resolution of plots; no. of points in each axial direction.
	func : function
		Function to be evaluated on a sparse grid.
	'''
	print('Plotting Interpolant...')

	# Construct MaternPSGEmulator object.
	PSGEmulatorObject = MaternPSGEmulator(
		dim,
		level,
		penalty,
		func,
		nu_list,
		lengthscale_list,
		sigma_list
	)

	PSGEmulatorObject.calculate_weights()

	if dim == 1:
		fig, ax = plt.subplots()
		points = list(PSGEmulatorObject.dict.keys())
		weights = [PSGEmulatorObject.dict[point][1] for point in points]
		
		def	kernel_approximation(MaternObject, arg, points, weights):
			'''
			Kernel approximation
			'''
			total = 0
			for i in range(len(points)):
				total +=  weights[i]*MaternObject.kernel_list[0](
								points[i][0],
								arg
								)
			return total
	
		x_list = np.linspace(-0.5,0.5,resolution)
		y_list = [kernel_approximation(
				PSGEmulatorObject,
				x,
				points, 
				weights
			) for x in x_list]

		ax.plot(x_list,y_list)
		plt.show()

	elif dim == 2:
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		points = list(PSGEmulatorObject.dict.keys())
		weights = [PSGEmulatorObject.dict[point][1] for point in points]
		def kernel_approximation(
				MaternObject,
				arg1,
				arg2,
				points,
				weights
			):
			'''
			Kernel approximation.
			'''
			total = 0
			for i in range(len(points)):
				total += weights[i] *\
								MaternObject.kernel_list[0](
									points[i][0],
									arg1
								) *\
								MaternObject.kernel_list[1](
									points[i][1],
									arg2
								)
			return total

		x = y = np.linspace(-0.5,0.5,resolution)
		X, Y = np.meshgrid(x, y)
		z = np.array(kernel_approximation(
						PSGEmulatorObject,
						np.ravel(X),
						np.ravel(Y),
						points,
						weights
					)
				)
				
		Z = z.reshape(X.shape)

		ax.plot_surface(X, Y, Z)
		plt.show()
	
def L2_error(
		dim,
		level,
		penalty,
		nu_list,
		lengthscale_list,
		sigma_list,
		no_mc_points,
		func
	):
	'''
	Caluclates relative L2 Error between Matern kernel interpolant defined on a
	lengthscale-informed sparse grid and the true function.
	
	Parameters
	----------
	dim : int
		Dimension of domain of function.
	level : int
		Level of construction of sparse grid.
	penalty : list of ints
		Penalty parameter in each dimension.
	nu_list : list of floats
		Smoothness in each dimension.
	lengthscale_list : list of floats
		Lengthscale in each dimension.
	sigma_list : list of floats
		Standard deviation in each dimension.
	no_mc_points : int
		Number of Monte Carlo points used to evaluate L2 integral.
	func : function
		Function to be evaluated on a sparse grid.

	Returns
	-------
	L2_error : float.
		L2 Error for given function and interpolation scheme.
	N : int.
		Number of points in PSG.
	'''

	# Construct MaternPSGEmulator object.
	PSGEmulatorObject = MaternPSGEmulator(
		dim,
		level,
		penalty,
		func,
		nu_list,
		lengthscale_list,
		sigma_list
	)

	PSGEmulatorObject.calculate_weights()

	print('Begin error testing:')
	# Generate MC points for sampling.
	mc_points = np.random.uniform(-0.5, 0.5, (no_mc_points, dim))

	print('Sampling simulator...')
	# Sample func at MC points.
	mc_func_evals = np.zeros(no_mc_points)
	for i in range(no_mc_points):
		mc_func_evals[i] = func(mc_points[i])
	
	print('Sampling emulator...')
	# Sample emulator at MC points.
	mc_emulator_evals = np.zeros(no_mc_points)
	for i in range(no_mc_points):
		mc_emulator_evals[i] = PSGEmulatorObject(mc_points[i])

	# Approximate L2 norm of f.
	f_norm = np.sqrt(sum(mc_func_evals**2))

	# Approximate absolute L2 integral.
	absolute_error = np.sqrt(sum((mc_func_evals-mc_emulator_evals)**2))
	
	# Return relative error.
	return absolute_error / f_norm, len(PSGEmulatorObject.dict.keys())


def error_data(
		dim,
		level,
		penalty,
		nu_list,
		lengthscale_list,
		sigma_list,
		no_mc_points,
		func
	):
	'''
	Returns data needed to calculate error between Matern kernel interpolant
	defined on a lengthscale-informed sparse grid and the true function.
	
	Parameters
	----------
	dim : int
		Dimension of domain of function.
	level : int
		Level of construction of sparse grid.
	penalty : list of ints
		Penalty parameter in each dimension.
	nu_list : list of floats
		Smoothness in each dimension.
	lengthscale_list : list of floats
		Lengthscale in each dimension.
	sigma_list : list of floats
		Standard deviation in each dimension.
	no_mc_points : int
		Number of Monte Carlo points used to evaluate L2 integral.
	func : function
		Function to be evaluated on a sparse grid.

	Returns
	-------
	L2_error : float.
		L2 Error for given function and interpolation scheme.
	N : int.
		Number of points in PSG.
	'''

	# Construct MaternPSGEmulator object.
	PSGEmulatorObject = MaternPSGEmulator(
		dim,
		level,
		penalty,
		func,
		nu_list,
		lengthscale_list,
		sigma_list
	)

	PSGEmulatorObject.calculate_weights()

	print('Begin error testing:')
	# Generate MC points for sampling.
	mc_points = np.random.uniform(-0.5, 0.5, (no_mc_points, dim))

	print('Sampling simulator...')
	# Sample func at MC points.
	mc_func_evals = np.zeros(no_mc_points)
	for i in range(no_mc_points):
		mc_func_evals[i] = func(mc_points[i])
	
	print('Sampling emulator...')
	# Sample emulator at MC points.
	mc_emulator_evals = np.zeros(no_mc_points)
	for i in range(no_mc_points):
		mc_emulator_evals[i] = PSGEmulatorObject(mc_points[i])
	
	# Return monte carlo points and evaluations of f and emulator.
	return(
		mc_points,
		mc_func_evals,
		mc_emulator_evals,
		len(PSGEmulatorObject.dict.keys())
	)

if __name__ == '__main__':
	main()
