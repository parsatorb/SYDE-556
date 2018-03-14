from BaseNeuron import BaseNeuron
import numpy as np

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
	def setParamsMax (self, xMax, aMax, xInt):
		c = np.exp( (self._tRef - (1. / aMax)) / self._tRC ) #just some constant to make the math nicer
		k = np.dot(xInt/(np.abs(xMax)), self._e) - 1	#some other constant
		self._gain = (1 - 1. / (1 - c))/k
		self._bias = 1 - (self._gain * np.dot(xInt/(np.abs(xMax)), self._e))
		
		# self._gain = (1 + xInt) / np.dot(self._e, xInt)
		# self._bias = (1. / ( 1 - np.exp((aMax*self._tRef - 1)/(aMax*self._tRC)) )) - self._gain*np.dot(self._e, xMax)

		# x = 1. / (1 - np.exp((self._tRef - (1. / aMax)) / self._tRC))
		# self._gain = (1 - x) / (xInt - 1)
		# self._bias = 1 - (self._gain * xInt)
		

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



