## PACKAGES ##
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from datetime import datetime

## MODULES ##
from kernels import maternKernel
from LISG import construct_psg
from emulator import L2_error, error_data
from scipy.special import comb

def main():
	'''
	Calculate approximate L2 error against N for different dimensions.
	'''
	## PARAMETERS ##

	# Universal parameters.
	iso_penalty = lambda j : np.zeros(len(j)) # Isotropic sparse grid.
	lin_penalty = lambda j : j # Linear penalty; exp lengthscle growth.
	log_penalty = lambda j : np.ceil(np.log2(j+1)) # Log penality; linear 

	# lengthscale growth.
	num_centres = 20 # Number of kernels to be used in random kernel function.
	res = 100 # Number of points used to approximate RMSE.
	
	# Parameters for Experiment 1.
	dim_list1 = [2,3,5,10,25,50,100] # Dimensions at which to run experiment.
	nu_list1 = [1.5] # Kernels considered.
	level_list1 = [1,2,3,4,5,6,7]
	scaling_sd1 = 5

	print('Initiating experiment: Beep Boop')
	experiment1(
		lin_penalty, # Penalty associated to kernel and grid.
		lin_penalty, # Correct penalty associated to function.
		dim_list1,
		nu_list1,
		num_centres,
		scaling_sd1,
		level_list1,
		res
	)


class random_kernel_func:
	'''
	Represents a function in a chosen stretched Native space to be
	interpolated.
	'''

	def __init__(
		self,
		dim,
		nu_array,
		penalty,
		num_centres,
		scaling_sd
	):
		'''
	
		Attributes
		----------
		
		dim : int
			Dimension of function domain.
		nu_array : array-like
			Smoothness parameter of constituent kernels in each dimension.
		penalty : function
			Given input dimension, returns corresponding penalty.
		num_centres : int
			Number of kernels in linear combination.
		kernel_array : array-like of maternKernel objects.
			Represents separable Matern kernel.
		scale_array : array-like of floats.
			Scaling of each kernel in linear combination.
		scaling_sd : float.
			Standard deviation for scalings.
		'''
		
		# Store parameters as attributes
		self.dim = dim
		self.nu_array = nu_array
		self.penalty = penalty
		self.num_centres = num_centres
		self.scaling_sd = scaling_sd

		# Construct and store kernel_array
		self.construct_kernel()

		# Draw and store random scalings and centres of kernels.
		self.random_scales()
		self.random_centres()

	def random_scales(self):
		'''
		Draws scalings for each kernel from normal disctribution.
		'''
		self.scale_array = np.random.normal(
							0.0,
							scale=self.scaling_sd,
							size=self.num_centres
						)
	
	def random_centres(self):
		'''
		Draws centres for each kernel uniformly over domain.
		'''
		self.centre_array = np.random.uniform(
											-0.5,
											0.5,
											(self.num_centres, self.dim)
										)
		
	
	def construct_kernel(self):
		'''
		Construct np.array of maternKernel objects representign separable
		Matern kernel.
		'''
		kernel_array = np.zeros(self.dim, dtype=object)
		# Loop through each dimension.
		for j in range(self.dim):
			kernel_array[j] = maternKernel(
									self.nu_array[j],
									2**self.penalty(j),
									1
								)
		self.kernel_array = kernel_array

	def __call__(self, x):
		'''
		Return value of random kernel function at x \in (-0.5,0.5)^dim. 
		'''
		value_array = np.ones((self.num_centres, self.dim))
		for i in range(self.num_centres):
			for j in range(self.dim):
				value_array[i,j] = self.kernel_array[j](
												x[j],
												self.centre_array[i,j]
											)	
		return np.dot(self.scale_array, np.prod(value_array,axis=1))
	
def experiment1(
	penalty,
	func_penalty,
	dim_list,
	nu_list,
	num_centres,
	scaling_sd,
	level_list,
	res,
):
	'''
	Returns array of approximate L2 errors for approximating random 
	kernel function for different parameterisations.
	'''
	
	filename = f'experiment1_data/linp_{num_centres}_basis_funcs_{scaling_sd}_scaling_{res}_errorres'

	# Initialize an empty dictionary or load the existing one
	if os.path.exists(filename):
		try:
			with open(filename, 'rb') as file:
				data = pickle.load(file)  # Try to load data.
				print("File loaded successfully.")
		except (EOFError, pickle.UnpicklingError):
			print("File is empty or corrupted. Initializing new data.")
			data = {}  # Start fresh if the file is empty or corrupted.
	else:
		print("File does not exist. Creating a new one.")
		data = {}  # Create a new dictionary.
	
	# For each kernel type...
	for nu in nu_list:
		print(f'#### nu = {nu} ####')
		# For each dimension...
		for dim in dim_list:
			print(f'~~ dim = {dim} ~~')
			# Create array of nu in each dimension.
			nu_array = np.repeat([nu], dim) #+1?
			# Construct random kernel function.
			test_func = random_kernel_func(
											dim,
											nu_array,
											func_penalty, #start at 1?
											num_centres,
											scaling_sd
										)
			# Create array of penalties.
			penalty_array = penalty(np.arange(0,dim+1,dtype=np.float64))
			print('penalty')
			print(penalty_array)
			# For each level up to max_level...
			for level in level_list:
				print(f'level = {level}')
				mc_points, mc_func_evals, mc_emulator_evals, N = error_data(
					dim,
					level,
					penalty_array,
					np.repeat(nu,dim),
					2**penalty_array,
					np.ones(dim),
					res,
					test_func
				)


				# Open and read data dictionary file
				print('Saving...')
				with open(filename, 'rb') as file:
					data_dict = pickle.load(file)
				# Initiate key if not used previously.
				if (nu, dim, level) not in data_dict:
					data_dict[(nu, dim, level)] = []
				# Store data in dictionary.
				data_dict[(nu, dim, level)].append((
					mc_func_evals,
					mc_emulator_evals,
					N,
					datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				))
				# Save file.
				with open(filename, 'wb') as file:
					pickle.dump(
								data_dict,
								file,
								protocol=pickle.HIGHEST_PROTOCOL
							)
			print('Done!')
	

if __name__ == '__main__':
	main()
