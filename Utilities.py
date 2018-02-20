import numpy as np
def generate_signal(T, dt, rms, limit, seed=None):
	if seed is None: 
		np.random.seed(None)
	else: 
		np.random.seed(seed)


	t = np.arange(0, T, dt)
	a = np.zeros(len(t), dtype=complex)

	step = 2*np.pi/T;
	steps = int(np.ceil(2*np.pi*limit/step))
	
	freqRange = np.arange(0, 2*np.pi*limit, step)
	reals = np.random.rand(1, steps)[0]
	realsReversed = reals[::-1]
	realsReversed = realsReversed[:-1].copy()
	
	imag = np.random.rand(1, steps)[0]
	imag = np.multiply(imag, np.complex(0,1))
	imagReversed = np.multiply(imag[::-1], -1)
	imagReversed = imagReversed[:-1].copy()
	imag[0] = 0

	a[0:len(reals)] = np.add(reals, imag)
	a[-len(realsReversed):] = np.add(realsReversed, imagReversed)
	timeSignal = np.fft.ifft(a)
	timeSignal = scale(timeSignal, rms)

	#making them hermitian symmetrical so that time signal is purely real
	reals = np.append(realsReversed, reals)
	imag = np.append(imagReversed, imag)
	freqRange = np.append(np.multiply(freqRange[::-1], -1), freqRange[1:])

	freqSignal = np.add(reals, imag)
	freqSignal = scale(freqSignal, rms)
	return ((t, timeSignal), (freqRange, freqSignal))

def scale(f, desired_rms):
	rms = np.sqrt(np.mean(np.square(f.real)))
	scaling = desired_rms/rms
	f = np.multiply(f, scaling)
	return f