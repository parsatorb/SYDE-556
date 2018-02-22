from Utilities import *
from LIF import LIF
import numpy as np
import sys

def two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref):
    neuron1 = LIF(alpha, Jbias, 1, tau_ref, tau_rc)
    neuron2 = LIF(alpha, Jbias, -1, tau_ref, tau_rc)
    voltages1, spikes1 = neuron1.voltageBuildup(x, dt)
    voltages2, spikes2 = neuron2.voltageBuildup(x, dt)
    return (spikes1, spikes2)

def rms(signal):
	return np.sqrt(np.mean(np.square(signal)))

T = 4.0         # length of signal in seconds
dt = 0.001      # time step size
limit = int(sys.argv[1])
h_only = bool(sys.argv[2])

# Generate bandlimited white noise (use your own function from part 1.1)
x, X = generate_signal(T, dt, rms=0.5, limit=limit, seed=3)
#My function returns the ranges too, so just need a little more sifting
u, x = x
u, X = X

Nt = len(x)             #Number of time steps
t = np.arange(Nt) * dt  #The range on the time signal

# Neuron parameters
tau_ref = 0.002          # Reference time constant
tau_rc = 0.02            # RC time constant 
x0 = 0.0                 # firing rate at x=x0 is a0
a0 = 40.0
x1 = 1.0                 # firing rate at x=x1 is a1
a1 = 150.0

# Calculating neuron parameters
eps = tau_rc/tau_ref
r1 = 1.0 / (tau_ref * a0)
r2 = 1.0 / (tau_ref * a1)
f1 = (r1 - 1) / eps
f2 = (r2 - 1) / eps
alpha = (1.0/(np.exp(f2)-1) - 1.0/(np.exp(f1)-1))/(x1-x0) 
x_threshold = x0-1/(alpha*(np.exp(f1)-1))              
Jbias = 1-alpha*x_threshold;   

# Simulate the two neurons (use your own function from part 3)
# Own function is implemented in the jupyter notebook
# I'll place it somewhere better if this functionality is needed again sometime
spikes = two_neurons(x, dt, alpha, Jbias, tau_rc, tau_ref)


freq = np.arange(Nt)/T - Nt/(2.0*T)   #Range of frequencies 
omega = freq*2*np.pi                  #Range of frequencies in rad

r = spikes[0] - spikes[1]               #Response of both the neurons
R = np.fft.fftshift(np.fft.fft(r)) 		#Response of neurons in frequency domain

sigma_t = 0.025                          #Standard deviation of gaussian used for smoothing function in the window
W2 = np.exp(-omega**2*sigma_t**2)     	#Function we're going to use to smooth our window
W2 = W2 / sum(W2)                        #Normalizing

CP = X*R.conjugate()                  #
WCP = np.convolve(CP, W2, 'same')  #
RP = R*R.conjugate()                  #
WRP = np.convolve(RP, W2, 'same')  #
XP = X*X.conjugate()                  #
WXP = np.convolve(XP, W2, 'same')  #

H = WCP / WRP                         #Optimal filter, in freq domain

h = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H))).real  #Optimal filter, in time domain

XHAT = H*R                            #Decoded x, in freq domain

xhat = np.fft.ifft(np.fft.ifftshift(XHAT)).real  #Decoded x
xhat = xhat*(rms(x)/rms(xhat)) #normalize

import pylab

if not h_only:

	pylab.figure(1)
	pylab.subplot(1,2,1)
	pylab.plot(freq, np.sqrt(XP))  #Squared X in freq domain 
	pylab.legend()
	pylab.title('Square Magnitude of X')
	pylab.xlabel('Freq(rad)')
	pylab.ylabel('$|X(\omega)|^2$')
	pylab.xlim(-20, 20)

	pylab.subplot(1,2,2)
	pylab.plot(freq, np.sqrt(RP))  #Response spikes in freq domain
	pylab.legend()
	pylab.title('Square Magnitude of Spike Spectrum')
	pylab.xlabel('Freq(rad)')
	pylab.ylabel('$|R(\omega)|^2$')


	pylab.figure(2)
	pylab.subplot(1,2,1)
	pylab.plot(freq, H.real)   #Optimal filter in freq domain
	pylab.xlabel('Freq(rad)')
	pylab.title('Optimal Filter (Freq Domain)')
	pylab.xlim(-50, 50)
	
	pylab.subplot(1,2,2)
	pylab.plot(t-T/2, h)       #Optimal Filter in time domain
	pylab.title('Optimal Filter (Time Domain)')
	pylab.xlabel('Time (s)')
	pylab.xlim(-0.5, 0.5)

	pylab.figure(4)
	pylab.plot(freq, np.abs(R))   #Spikes in freq Domain
	pylab.xlabel('Freq(rad)')
	pylab.ylabel('$|R(\omega)|$')
	pylab.title('Spike Response Spectrum')
	# pylab.xlim(-100, 100)

	pylab.figure(5)
	pylab.plot(freq, np.abs(X))   #Spikes in freq Domain
	pylab.xlabel('Freq(rad)')
	pylab.ylabel('$|X(\omega|$')
	pylab.title('X Response Spectrum')
	pylab.xlim(-10, 10)

	pylab.figure(6)
	pylab.plot(freq, np.abs(XHAT))   #Spikes in freq Domain
	pylab.xlabel('Freq(rad)')
	pylab.ylabel('$|XHAT(\omega|$')
	pylab.title('XHAT Response Spectrum')
	pylab.xlim(-10, 10)


if not h_only:
	pylab.figure(3)
	pylab.plot(t, r, color='k', label='Neural Spikes', alpha=0.2)  #Plot of neural spikes
	pylab.plot(t, x, linewidth=2, label='Signal (x)')           #Original white signal
	pylab.plot(t, xhat, label='Decoded Signal')                     #Approximated signal
	pylab.title('Neural Spikes and Approximation of Signal')
	pylab.legend(loc='best')
	pylab.xlabel('Time (s)')

pylab.show()
