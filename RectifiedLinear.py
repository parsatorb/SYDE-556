from BaseNeuron import BaseNeuron
import numpy as np

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
