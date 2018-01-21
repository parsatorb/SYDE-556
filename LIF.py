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
		f = np.vectorize(__modelHelper)
		return f(current)


	"""
	Helper for the model function
	@param j: Must be scalar
	"""
	def __modelHelper(j):
		if j > 1:
			return 0

		return 1/(self._tRef - self._tRC*np.log(1 - 1/j))

	def setParams(max_rates, xrange, samplingRate=0.05):






