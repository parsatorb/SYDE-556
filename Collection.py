from LIF import LIF
from BaseNeuron import BaseNeuron
import numpy as np

class Collection:

	def __init__(self, N, es, max_rates=[100, 200], radius=1, dx=0.05, f=lambda t: t):
		dx = dx/radius
		xInts = np.random.uniform(-1, 1, N)
		max_rates = np.random.uniform(max_rates[0], max_rates[1], N)

		#TODO: delete this chunk
		slope = max_rates / (radius*es - xInts);
		biases = -1*slope*xInts;
		gains = abs(slope)

		x = np.arange(-1, 1, dx)
		A = np.zeros(shape=(np.size(x),N))

		zero_index = np.where(np.abs(x)<dx/10)
		selectedNeuron = None

		for i in range(0, N):
			neuron = LIF(gains[i], biases[i], es[i])
			neuron.setParamsMax(es[i], max_rates[i], xInts[i])
			gains[i] = neuron._gain
			biases[i] = neuron._bias
			a = neuron.tuningSimple(f(x))
			if (a[zero_index] > 20 and a[zero_index] < 50): #I know its sloppy hardcoding this in, but im short on time
				selectedNeuron = neuron
			A[:, i] = a

		d = BaseNeuron.decodeIdeal(A, f(x))
		x = x*radius

		self._A = A
		self._x = x
		self._d = d
		self._N = N
		self._gains = gains
		self._biases = biases
		self._es = es
		self._selected = selectedNeuron


	def spikingBehaviour(self, xt, dt, h):
		spikes = np.zeros(shape=(np.size(xt), self._N))
		activity = np.zeros(shape=(np.size(xt), self._N))
		for i in range(0, self._N):
			neuron = LIF(self._gains[i], self._biases[i], self._es[i])
			neuralSpikes, voltages = neuron.voltageBuildup(xt, dt)
			spikes[:, i] = neuralSpikes*self._es[i]
			activity[:, i] = np.convolve(neuralSpikes, h, mode='same')
			self._A = activity
		return spikes, activity

	def decodeForFunction(self, A, f, dt):
		self._d = BaseNeuron.decodeIdeal(A, f)
		return self._d

