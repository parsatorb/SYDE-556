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
	Decode
	@param A: The neuron activity being decoded, must be array or numpy array.
	@param X: The original vector.
	@param sigma: Standard deviation of the noise
	@param s: Sample steps
	Returns decoders array D.
	"""
	@classmethod
	def decode(cls, A, X, sigma=0):
		A = np.array(A)
		A_t = A.transpose()
		S = np.size(X)
		num_neurons = len(A[1,:])
		gamma = ( np.divide((np.dot(A_t, A)), S) + (sigma*sigma)*(np.identity(num_neurons)) )
		D = np.dot(np.linalg.inv(gamma), np.divide(np.dot(A_t, X), S))
		return D


	"""
	Decode Ideal (assuming no noise)
	@param A: The neuron activity being decoded, must be array or numpy array.
	@param X: The original vector.
	@param sigma: Standard deviation of the noise
	@param s: Sample steps
	Returns decoders array D.
	"""
	@classmethod
	def decodeIdeal(cls, A, X):
		return np.dot(np.linalg.pinv(np.dot(A.transpose(), A)), np.dot(A.transpose(), X))
		# A = np.array(A)
		# A_t = A.transpose() 
		# D = np.dot(np.linalg.inv(np.dot(A_t, A)), np.dot(A_t, X))
		# return D

class RectifiedLinear(BaseNeuron):

	"""
	See BaseNeuron.py for param descriptions
	xInt stands for x intercept, the point at which this neuron stops firing
	"""
	def __init__(self, gain, bias, e):
		super().__init__(gain, bias, e)


	"""
	Takes current and returns activity.
	@param current: Must be an array, scalar, or numpy array
	"""
	def model(self, current):
		#element-wise maximum
		eMax = lambda t: max(t, 0)
		vMax = np.vectorize(eMax)
		return vMax(current);

class LIF(BaseNeuron):
	"""
	See BaseNeuron.py for param descriptions
	"""
	def __init__(self, gain, bias, e, tRef=0.002, tRC=0.02):
		super().__init__(gain, bias, e)
		#Reference and RC time constants (in seconds)
		self._tRef = tRef
		self._tRC = tRC

	"""
	Takes current and returns activity.
	@param current: Must be an array, scalar, or numpy array
	"""
	def model(self, current):
		#element-wise maximum
		f = np.vectorize(self.__modelHelper)
		return f(current)


	"""
	Helper for the model function
	@param j: Must be scalar
	"""
	def __modelHelper(self, j):
		if j <= 1:
			return 0
		return 1./(self._tRef - ( self._tRC*np.log(1 - 1./j) ))

	"""
	Set gain and bias for a max point xMax, aMax, and an 'x-int'.
	aMax must be scalar. All other inputs must be array_like of the same dimension 
	"""
	def setParams(self, xMax, aMax, xInt):
		c = np.exp( (self._tRef - (1. / aMax)) / self._tRC ) #just some constant to make the math nicer
		k = np.dot(xInt, self._e) - 1	#some other constant
		self._gain = (1 - 1. / (1 - c))/k
		self._bias = 1 - (self._gain * np.dot(xInt, self._e))

	"""
	Set gain and bias based on resting and stimulated frequencies in Hz
	"""
	def setParams(self, resting, stim):
		a = self._tRef
		b = self._tRC
		self._bias = 1./(1 - np.exp((resting*a - 1)/(resting*b)))
		self._gain = (1./(1 - np.exp((stim*a - 1.)/(stim*b))) - self._bias)


	"""
	x is array-like representing the signal over time
	step is the constant time step between each value in the x array
	init_cond is the initial voltage level, assumed 0 usually
	"""
	def voltageBuildup(self, x, step, init_cond=0):
		spikes = np.zeros(len(x))
		voltage = np.zeros(len(x))
		voltage[0] = init_cond
		refractory_period = int(self._tRef//step)
		i = 0
		while i < len(voltage)-1:
			if voltage[i] < 0:
				voltage[i] = 0

			if voltage[i] >= 1:
				lower = i+1
				upper = lower + refractory_period - 1
				voltage[lower:upper] = [0] * (upper-lower)
				spikes[i+1] = 1
				i = upper
			else:
				J = self.current(x[i])
				v_change = step*(1./self._tRC)*(J - voltage[i])
				voltage[i+1] = voltage[i] + v_change
				i += 1
				

		return (spikes, voltage)

