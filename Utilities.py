import numpy as np

def generate_signal(T, dt, rms, limit, seed=None, gaussian=False):
    b = 2*np.pi*limit
    if seed is None: 
        np.random.seed(None)
    else: 
        np.random.seed(seed)
    
    step = 2*np.pi/T
    w = np.arange(step, step*((T/dt)//2), step)
    reals = np.random.rand(len(w)) - 0.5
    imaginaries = np.multiply(np.random.rand(len(w))-0.5, np.complex(0,1))
    
    if gaussian:
        gaussian_scaling = np.exp((-1)*w**2/(2*b**2))    
    else:
        gaussian_scaling = np.ones(len(w))
        gaussian_scaling[np.where(np.abs(w)>b)] = 0

    reals = np.multiply(reals, gaussian_scaling)
    imaginaries = np.multiply(imaginaries, gaussian_scaling)
    
    realsReversed = reals[::-1]
    imagReversed = (-1)*imaginaries[::-1]
    
    freqReversed = realsReversed + imagReversed
    freqForward = reals + imaginaries
    
    freqSignal = np.append(freqReversed, [0])
    freqSignal = np.append(freqSignal, freqForward)
    
    timeSignal = np.fft.ifft(np.fft.ifftshift(freqSignal))

    timeSignal = scale(timeSignal, rms)
    freqSignal = scale(freqSignal, rms)

    freqRange = np.append((-1)*w[::-1], [0])
    freqRange = np.append(freqRange, w)
    t = np.arange(0, len(freqRange))*dt
    return ((t, timeSignal), (freqRange, freqSignal))


def scale(f, desired_rms):
	rms = np.sqrt(np.mean(np.square(f.real)))
	scaling = desired_rms/rms
	f = np.multiply(f, scaling)	
	return f

#TODO: Delete this
def generate_signal_old(T, dt, rms, limit, seed=None):
	if seed is None: 
		np.random.seed(None)
	else: 
		np.random.seed(seed)


	t = np.arange(0, T-dt, dt)
	w = np.arange(0, T-dt, dt)-(T/2)
	a = np.zeros(len(t), dtype=complex)
	X = np.zeros(len(w))

	step = 2*np.pi/T;
	steps = int(np.ceil(2*np.pi*limit/step))
	
	freqRange = np.arange(0, 2*np.pi*limit, step)
	reals = np.random.rand(steps)
	reals[0] = 0
	realsReversed = reals[::-1]
	realsReversed = realsReversed[:-1].copy()
	
	imag = np.random.rand(steps) - 0.5
	imag = np.multiply(imag, np.complex(0,1))
	imag[0] = 0
	imagReversed = np.multiply(imag[::-1], -1)
	imagReversed = imagReversed[:-1].copy()

	a[0:len(reals)] = np.add(reals, imag)
	a[-len(realsReversed):] = np.add(realsReversed, imagReversed)


	timeSignal = np.fft.ifft(a)
	timeSignal = scale(timeSignal, rms)

	#making them symmetrical so that time signal is purely real
	reals = np.append(realsReversed, reals)
	imag = np.append(imagReversed, imag)
	freqRange = np.append(np.multiply(freqRange[::-1], -1), freqRange[1:])

	freqSignal = np.add(reals, imag)
	freqSignal = scale(freqSignal, rms)
	
	diff = len(w) - len(freqSignal)
	padding = np.zeros(int(diff/2))
	freqSignal = np.append(padding, freqSignal)
	freqSignal = np.append(freqSignal, padding)

	return ((t, timeSignal), (w, freqSignal))