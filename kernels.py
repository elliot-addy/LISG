'''
Script containing 1D kernel objects, used in constructing emulators. For sparse
grids we only consider product kernels, hence 1D kernels are sufficient.
'''

import numpy as np
import matplotlib.pyplot as plt

def main():
	'''
	Construct Matern kernel object given parameterisation and plot kernel
	with a single argument fixed at zero.
	'''
	
	# Kernel parameters.
	nu = 1.5
	sigma = 2
	lengthscale = 3

	# Plotting parameters.
	resolution = 100
	start = -3
	stop = 3

	# Function.
	maternKernelObject = maternKernel(nu, sigma, lengthscale)
	x_array = np.linspace(start, stop, resolution)
	zero_array = np.zeros(resolution)

	# Plotting.
	fig, ax = plt.subplots()
	plt.plot(x_array, maternKernelObject(x_array, zero_array))
	plt.show()

class maternKernel:
	'''
	1D Matern kernel object.

	Attributes
	----------
	nu : float
		Regularity hyperparameter.
	lengthscale : float
		Lengthscale hyperparameter.
	sigma : float
		Scaling / standard deviation hyperparameter.
	'''

	def __init__(
		self,
		nu,
		lengthscale,
		sigma: float = 1
	):
		
		# Save parameters to object.
		self.nu = nu
		self.lengthscale = lengthscale
		self.sigma = sigma

		# Initilaise parameterisation of kernel object.
		self.construct(nu)

	def construct(
		self,
		nu
	):
		'''
		Parameterise Matern kernel for use in call function.

		Args:
			nu: Int, smoothness parameter.
			lengthscale: Float, lengthscale parameter.
		'''
	
		# Explicitly define common kernels.
		if nu == 0.5:
			def unscaled_kernel(dist):
				'''
				Unscaled 1D Matern 1/2 kernel (or exponential kernel).
				'''
				return np.exp(-dist/self.lengthscale)

		elif nu == 1.5:
			def unscaled_kernel(dist):
				'''
				Unscaled 1D Matern 3/2 kernel.
				'''
				return (1 + np.sqrt(3)*dist/self.lengthscale)\
					*np.exp(-np.sqrt(3)*dist/self.lengthscale)
		
		elif nu == 2.5:
			def unscaled_kernel(dist):
				'''
				Unscaled 1D Matern 5/2 kernel.
				'''
				return (1 + (np.sqrt(5)*dist/self.lengthscale)\
					+ (5*dist**2)/(3*self.lengthscale**2))\
					*np.exp(-(np.sqrt(5)*dist)/self.lengthscale)
				
		elif nu == 'inf':
			def unscaled_kernel(dist):
				'''
				Unscaled 1D squared exponential / Gaussian kernel.
				'''
				return np.exp(-(dist/self.lengthscale)**2/2)

		self.unscaled_kernel = unscaled_kernel

	def __call__(
		self,
		arg1,
		arg2
	):
		# Stationary kernel; function of distance of arguments.
		dist = abs(arg1 - arg2)
		# Return unscaled kernel, vairance still to be accounted for.
		return self.unscaled_kernel(dist)

if __name__ == '__main__':
	main()
