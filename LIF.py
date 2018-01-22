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
	def setParams(self, xMax, aMax, xInt):
		c = np.exp( (self._tRef - (1. / aMax)) / self._tRC ) #just some constant to make the math nicer
		k = np.dot(xInt, self._e) - 1	#some other constant
		self._gain = (1 - 1. / (1 - c))/k
		self._bias = 1 - (self._gain * np.dot(xInt, self._e))





