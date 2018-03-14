import numpy as np
import nengo

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
	rms = np.sqrt(np.mean(np.square(f[np.where(f > 0)].real)))
	scaling = desired_rms/rms
	f = np.multiply(f, scaling)	
	return f

def _filter(n, tau):
    t_h = (np.arange(200)*dt)-0.1
    h = np.power(t_h, n)*np.exp(-t_h/tau)
    h[np.where(t_h<0)]=0
    h = h/norm(h,1)
    return (t_h, h)



def compute_lif_decoder(n_neurons, dimensions, encoders, max_rates, intercepts, tau_ref, tau_rc, radius, x_pts, function):
    """
    Parameters:
        n_neurons: number of neurons (integer)
        dimensions: number of dimensions (integer)
        encoders: the encoders for the neurons (array of shape (n_neurons, dimensions))
        max_rates: the maximum firing rate for each neuron (array of shape (n_neurons))
        intercepts: the x-intercept for each neuron (array of shape (n_neurons))
        tau_ref: refractory period for the neurons (float)
        tau_rc: membrane time constant for the neurons (float)
        radius: the range of values the neurons are optimized over (float)
        x_pts: the x-values to use to solve for the decoders (array of shape (S, dimensions))
        function: the function to approximate
    Returns:
        A (the tuning curve matrix)
        dec (the decoders)
    """
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(n_neurons=n_neurons, dimensions=dimensions, encoders=encoders, max_rates=max_rates, intercepts=[x/radius for x in intercepts], neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),radius=radius)
    sim = nengo.Simulator(model)
    
    x_pts = np.array(x_pts)
    if len(x_pts.shape) == 1:
        x_pts.shape = x_pts.shape[0], 1
    _, A = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=x_pts)
    
    target = [function(xx) for xx in x_pts]
    solver = nengo.solvers.LstsqL2()
    dec, info = solver(A, target)
    return A, dec

#convienence method
def compute_decoders(N, max_rates, x, dimensions=1, tau_ref=0.002, tau_rc=0.02, radius=1, function=lambda t: t):
	if dimensions==1:
		es = np.random.choice([-1, 1], size=(N, 1))
	elif dimensions==2:
		angles = np.random.uniform(0, 2*np.pi, N)
		es = np.array([np.cos(angles), np.sin(angles)]).transpose()

	max_rates = np.random.uniform(max_rates[0], max_rates[1], N)
	xInts = np.random.uniform(-1*radius, 1*radius, N)

	return compute_lif_decoder(n_neurons=N, dimensions=dimensions, encoders=es, max_rates=max_rates, intercepts=xInts, tau_ref=tau_ref, tau_rc=tau_rc, radius=radius, x_pts=x, function=function)





