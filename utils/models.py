import numpy as np
from numpy.fft import fft, ifft, fftfreq
from numpy.random import normal
import scipy.constants as const
from tqdm.notebook import tqdm

def mzm(Ai, Vπ, u, Vb):
    """
    MZM modulator 
    
    :param Vπ: Vπ-voltage
    :param Vb: bias voltage
    :param u:  input driving signal
    :param Ai: input optical signal 
    
    :return Ao: output optical signal
    """
    π  = np.pi
    Ao = Ai*np.cos(0.5/Vπ*(u+Vb)*π)
    
    return Ao


def edfa(Ei, Fs, G, NF, Fc):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]    

    :return: amplified noisy optical signal [nparray]
    """
    assert G > 0, 'EDFA gain should be a positive scalar'
    assert NF >= 3, 'The minimal EDFA noise figure is 3 dB'
    
    NF_lin   = 10**(NF/10)
    G_lin    = 10**(G/10)
    nsp      = (G_lin*NF_lin - 1)/(2*(G_lin - 1))
    N_ase    = (G_lin - 1)*nsp*const.h*Fc
    p_noise  = N_ase*Fs    
    noise    = normal(0, np.sqrt(p_noise), Ei.shape) + 1j*normal(0, np.sqrt(p_noise), Ei.shape)
    return Ei*np.sqrt(G_lin) + noise

def ssfm(Ei, Fs, Ltotal, Lspan, hz, alpha, gamma, D, Fc, amp, NF):      
    """
    Split-step Fourier method (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param amp: 'edfa', 'ideal', or 'None. [default:'edfa']
    :param NF: edfa noise figure [dB] [default: 4.5 dB]
    :param Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: propagated signal
    """             

    λ  = const.c/Fc
    α  = 1e-3*alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*const.c)
    γ  = gamma
            
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    
    Nspans = int(np.floor(Ltotal/Lspan))
    Nsteps = int(np.floor(Lspan/hz))
    
    Ech = Ei.reshape(len(Ei),)  
      
    linOperator = np.exp(-(α/2)*(hz/2) + 1j*(β2/2)*(ω**2)*(hz/2))
    
    for spanN in tqdm(range(1, Nspans+1)):   
        Ech = fft(Ech) #single-polarization field
        
        # fiber propagation step
        for stepN in range(1, Nsteps+1):            
            # First linear step (frequency domain)
            Ech = Ech*linOperator            

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech*np.exp(1j*γ*(Ech*np.conj(Ech))*hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)       
            Ech = Ech*linOperator           

        # amplification step
        Ech = ifft(Ech)
        if amp =='edfa':
            Ech = edfa(Ech, Fs, alpha*Lspan*1e-3, NF, Fc)
        elif amp =='ideal':
            Ech = Ech*np.exp(α/2*Nsteps*hz)
        elif amp == None:
            Ech = Ech*np.exp(0);         
          
    return Ech.reshape(len(Ech),)