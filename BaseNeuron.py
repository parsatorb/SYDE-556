import numpy as np

#Base class, should be subclassed for specific neurons

#Not forced to be abstract cuz this is still a work in progress and we'll probably change up
#the design fundamentally really soon.

#x represents real world signal trying to modelled as a vector
class BaseNeuron:

	"""
	Initializers: Parameters all neurons should have, used for defining input current
	@param e: stands for encoder, or preferred stimulus 
	"""
	def __init__(self, gain, bias, e):
		self._gain = gain
		self._bias = bias
		self._e = e;

	"""
	Returns activity of neuron in hertz. 
	"""
	def tuningSimple(self, x):
		#Yeah I know this is sloppy OOP, but I don't even know the final spec or scope of this library
		#so Im leaving it flexible for now. 
		#TODO: Make this less slop city
		return self.model(self.current(x))

	#Define in subclass
	def model():
		pass

	"""
	Returns current to the neuron. 
	@param x:  Vector input, must be either a scalar, array, or numpy array
	"""
	def current(self, x):
		x = np.array(x, float)
		gainedX = (self._gain)*x
		unbiased = np.dot(gainedX, np.array(self._e))
		total = unbiased + self._bias
		return total


	"""
	Decode, assuming no noise.
	@param A: The neuron activity being decoded, must be array or numpy array.
	@param X: The original vector.
	Returns decoders array D.
	"""
	def decodeIdeal(self, A, x):
		A = np.array(A)
		A_t = A.transpose()
		D = np.dot(np.linalg.pinv(np.dot(A_t, A)), np.dot(A_t, x))
		return D

	#def decodeNoisy(A, x, epsilon):





