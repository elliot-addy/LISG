## PACKAGES ##
import numpy as np
import matplotlib.pyplot as plt

## MODULES ##
from kernels import maternKernel

def main():
	'''
	Generates random function made of a linear combination of chosen Matern
	kernels for testing and plots for dim = 1,2.
	'''
	# Parameters.
	dim = 2
	nu_list = [.5,.5]
	lengthscale_list = [1,1]
	sigma_list = [1,1]
	num_kernels = 20

	# Plotting parameters.
	resolution = 20

	# Generate randomKernelFunc object.
	RandomKernelFunc = randomKernelFunc(
							dim,
							nu_list,
							lengthscale_list,
							sigma_list,
							num_kernels
						)

	# Plot if dim = 1 or 2,
	if dim == 1:
		fig, ax = plt.subplots()
		x_array = np.linspace(-0.5, 0.5, resolution)
		y_array = [RandomKernelFunc([x]) for x in x_array]
		ax.plot(x_array, y_array)
		plt.show()

	if dim == 2:
		fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
		x_array = y_array = np.linspace(-0.5, 0.5, resolution)
		X, Y = np.meshgrid(x_array, y_array)
		xy_array = np.array([np.ravel(X),np.ravel(Y)]).T
		z_array = np.array([RandomKernelFunc(xy) for xy in xy_array])
		Z = np.reshape(z_array, X.shape)
		ax.plot_surface(X, Y, Z)
		plt.show()


class randomKernelFunc:
	'''
	Represents a random Matern kernel function defined in [-0.5,0.5]^d.

	Attributes
	----------
	dim : int
		Dimension of domain of function.
	nu_list : list of floats (or 'inf')
		Smoothness in each axial direction.
	lengthscale_list : list of floats
		Lengthscale in each axial direction.
	sigma_list : list of floats
		Standard deviation in each dimension.
	num_kernels : int
		Number of kernels in linear combination.
	kernel_list : list of maternKernel objects.
		Represents the parameterised separable matern kernel.
	random_points : array-like
		Array of n-many dim-dimensional points at which kernels are centred.
	weights : list of floats
		Array of n-many real numbers that scale the kernels at each point.
	'''

	def __init__(
		self,
		dim,
		nu_list,
		lengthscale_list,
		sigma_list,
		num_kernels
	):
		self.dim = dim
		self.nu_list = nu_list
		self.lengthscale_list = lengthscale_list
		self.sigma_list = sigma_list
		self.num_kernels = num_kernels

		# Construct list of 1D Matern kernels representing separable Matern
		# kernel.
		kernel_list = []
		for i in range(dim):
			kernel_list.append(
				maternKernel(
					nu_list[i],
					lengthscale_list[i],
					sigma_list[i]
				)
			)
		self.kernel_list = kernel_list
		
		# Generate random points and weights.
		self.randomise()


	def randomise(self):
		'''
		Generate random points and weights to locate and scale random kernels.
		'''
		self.random_points = np.random.uniform(
								-0.5,
								0.5,
								(self.num_kernels, self.dim)
							)
		# Weights normally distributed around 0 according to total variance.
		self.random_weights = np.random.normal(
								0,
								np.prod(self.sigma_list),
								self.num_kernels
							)

	def __call__(self, arg):
		'''
		Given arg in [-0.5,0.5]^d, returns value (float) of Matern kernel
		function.
		'''
		values_array = np.zeros((self.num_kernels, self.dim))
		for i in range(self.num_kernels):
			for j in range(self.dim):
				values_array[i,j] = self.kernel_list[j](
										self.random_points[i,j],
										arg[j]
									)

		return self.random_weights.T @ np.prod(values_array, axis=1)

if __name__ == '__main__':
	main()
