import numpy as np

from scipy.fftpack     import fft, ifft
from scipy.constants   import pi
from scipy.signal      import hilbert
from scipy.interpolate import interp1d

from optic.modulation  import modulateGray, demodulateGray, GrayMapping
from optic.dsp         import pulseShape, lowPassFIR, firFilter, pnorm, decimate
from optic.metrics     import signal_power
from optic.models      import mzm
from optic.core        import parameters

from commpy.utilities  import upsample

def PAPR(N, sig):
    """
    PAPR (Peak to Average Power Ratio).
    Parameters
    ----------
    N : scalar
        number of symbols in a OFDM block
    Returns
    -------
    PAPR : scalar
           PAPR of a signal
    """
    
    L = 1000
    return 10*np.log10( max(sig[0:L*N])**2/signal_power(sig[0:L*N]) )


def hermit(V):
    """
    Hermitian simmetry block.
    Parameters
    ----------
    V : complex-valued np.array
        input array
        
    Returns
    -------
    Vh : complex-valued np.array
         vector with hermitian simmetry
    """
    
    Vh = np.zeros(2*len(V) + 2, complex)
    
    Vh[1:len(V)+1] = V 
    
    for j in range(len(V)):
        Vh[len(Vh) - j - 1] = np.conjugate(V[j])
    
    return Vh


def modulateOFDM(Nfft, Ns, N, Nz, G, K, pilot, symbTx):
    """
    OFDM symbols modulator.
    Parameters
    ----------
    Nfft   : scalar
             size of IFFT
    Ns     : scalar
             number of subcarriers
    N      : scalar
             number of information subcarriers
    Nz     : scalar
             number of nulls subcarriers
    G      : scalar
             cyclic prefix length
    K      : scalar
             number of pilot carriers per OFDM block
    pilot  : complex-valued scalar
             symbol chose for pilot carriers
    symbTx : complex-valued array
             symbols sequency transmitted
    
    Returns
    -------
    symbTx      : complex-valued np.array
                  symbols sequency transmitted
    symbTx_OFDM : complex-valued np.array
                  OFDM symbols sequency transmitted
    """
    
    Int = int(N/K)
    carriers = np.arange(0, N)
    pilot_carriers = np.append(carriers[0::Int], carriers[-1])

    symbTx_P = np.zeros( (int(len(symbTx)/N), Nfft + G), complex)
    aux = 0
    
    for i in range(len(symbTx_P)):
        # Adição das portadoras piloto
        symbTx[int(aux):int(aux + N)][pilot_carriers] = pilot
        
        symbTx_P[i, G : G + Nfft] = hermit( np.concatenate( (symbTx[int(aux):int(aux + N)], np.zeros(int(Nz)) ) ) )
        aux = aux + N
        
        # Aplicação da IFFT
        symbTx_P[i, G : Nfft + G] = ifft(symbTx_P[i, G:Nfft + G])#*np.sqrt(Nfft)

        # Adição do prefixo cíclico
        symbTx_P[i,:] = np.concatenate((symbTx_P[i,Nfft:Nfft + G],symbTx_P[i,G:Nfft + G]))

    # Conversão Paralelo -> Serial
    symbTx_OFDM = np.squeeze(symbTx_P.reshape(1,len(symbTx_P[0])*len(symbTx_P)))
    
    return symbTx_OFDM, symbTx


def demodulateOFDM(Nfft, Ns, N, Nz, G, K, pilot, symbRx_OFDM):
    """
    OFDM symbols demodulator.
    Parameters
    ----------
    Nfft        : scalar
                  size of IFFT
    Ns          : scalar
                  number of subcarriers
    N           : scalar
                  number of information subcarriers
    Nz          : scalar
                  number of nulls subcarriers
    G           : scalar
                  cyclic prefix length
    K           : scalar
                  number of pilot carriers per OFDM block
    pilot       : complex-valued scalar
                  symbol chose for pilot carriers
    symbRx_OFDM : complex-valued array
                  OFDM symbols sequency received

    Returns
    -------
    symbRx     : complex np.array
                 symbols sequency received equalized
    symbRx_neq : complex np.array
                 symbols sequency received without equalization
    H_abs      : np.array
                 channel amplitude estimated 
    H_pha      : np.array
                 channel phase estimated
    """
    
    Int = int(N/K)
    carriers = np.arange(0, N)
    pilot_carriers = np.append(carriers[0::Int], carriers[-1])

    symbRx_P = np.zeros((int(len(symbRx_OFDM)/(Nfft+G)), Nfft), complex)

    aux = 0
    H_abs_F = 0
    H_pha_F = 0

    for i in range(len(symbRx_P)):
        # Extração do prefixo cíclico
        symbRx_P[i,0:Nfft] = symbRx_OFDM[aux+G:aux+Nfft+G]
        aux = aux + Nfft + G

        # Aplicação da FFT
        symbRx_P[i,:] = fft(symbRx_P[i,:])#/np.sqrt(Nfft)

    for i in range(len(symbRx_P)):
        H_est = symbRx_P[i,1:1 + N][pilot_carriers] / pilot

        H_abs = interp1d(carriers[pilot_carriers], np.abs(H_est), kind = 'linear')(carriers)
        H_pha = interp1d(carriers[pilot_carriers], np.angle(H_est), kind = 'linear')(carriers)

        H_abs_F += H_abs
        H_pha_F += H_pha

    H_abs = H_abs_F/len(symbRx_P)
    H_pha = H_pha_F/len(symbRx_P)

    # Conversão P/S
    symbRx_S     = np.zeros(int(len(symbRx_P)*N), complex)  # Símbolos equalizados
    symbRx_S_neq = np.zeros(int(len(symbRx_P)*N), complex)  # Símbolos não-equalizados
    aux = 0

    for i in range(len(symbRx_P)):
        # Retirada da simetria hermitiana e equalização
        symbRx_S[int(aux):int(aux + N)]     = symbRx_P[i,1:1 + N]/(H_abs*np.exp(1j*H_pha))
        symbRx_S_neq[int(aux):int(aux + N)] = symbRx_P[i,1:1 + N]
        aux = aux + N

    symbRx     = symbRx_S
    symbRx_neq = symbRx_S_neq

    return symbRx, symbRx_neq, H_abs, H_pha


def Tx(paramTx):
    """
    OFDM transmissor (Tx).
    Parameters
    ----------
    paramTx : parameter object (struct), optional
        Parameters of the OFDM transmissor.
        
    paramTx.SpS: samples per symbol [default: 32]
    paramTx.Rs: symbols rate [default: 1.5e9]
    paramTx.Fa: sampling frequency [default: 48e9]
    paramTx.Fc: optical carrier frequency [Hz] [default: 193.4e12 Hz]
    paramTx.Scheme: OFDM scheme ["CE-DDO-OFDM", "DDO-OFDM"] [default: "CE-DDO-OFDM"]
    
    paramTx.M: number of constellation symbols [default: 4]
    paramTx.Nfft: size of IFFT [default: 512]
    paramTx.Ns: number of subcarriers [default: 255]
    paramTx.N: number of information subcarriers [default: 255]
    paramTx.Nz: number of null subcarriers [default: 0]
    paramTx.G: cyclic prefix length [default: 4]
    paramTx.K: number of pilot carriers per OFDM block [default: 8]
    
    paramTx.g:  gain in the signal before the MZM [default: 1.0]
    paramTx.Vπ: MZM switching voltage [V] [default: 4.4 V]
    paramTx.Vb: MZM bias voltage [V] [default: -2.2 V]
    paramTx.Pi_dBm: optical signal power [dBm] [default: 0 dBm]
    
    paramTx.H: phase modulation parameter [default: 0.35/(2*pi)]
    paramTx.fc: electrical carrier frequency [Hz] [default: 1e9 Hz]
    paramTx.A: electrical carrier amplitude [default: 1]
    
    Returns
    -------
    sigTxo : np.array
             optical signal.
    sigTx  : np.array
             time-domain baseband OFDM signal
    sigSig : np.array
             time-domain modulated signal (CE-DDO-OFDM or DDO-OFDM)
    symbTx : complex-valued np.array
             symbols sequency transmitted
    t      : np.array
             time vector
    pulse  : np.array
             pulse chose in transmission
    pilot  : complex-valued scalar
             symbol chose for pilot carriers
    """
    
    # Parâmetros da transmissão
    paramTx.SpS = getattr(paramTx, "SpS", 32)
    paramTx.Rs  = getattr(paramTx, "Rs", 1.5e9)
    paramTx.Fa  = getattr(paramTx, "Fa", 48e9)
    paramTx.Fc  = getattr(paramTx, "Fc", 193.4e12)
    paramTx.Scheme = getattr(paramTx, "Scheme", "CE-DDO-OFDM")
    
    # Parâmetros do esquema OFDM
    paramTx.M    = getattr(paramTx, "M", 4)
    paramTx.Nfft = getattr(paramTx, "Nfft", 512)
    paramTx.Ns   = getattr(paramTx, "Ns", 255)
    paramTx.N    = getattr(paramTx, "N", 255)
    paramTx.Nz   = getattr(paramTx, "Nz", 0)
    paramTx.G    = getattr(paramTx, "G", 4)
    paramTx.K    = getattr(paramTx, "K", 8)
    
    # Parâmetros da portadora óptica
    paramTx.g      = getattr(paramTx, "g", 1.0)
    paramTx.Vπ     = getattr(paramTx, "Vπ", 4.4)
    paramTx.Vb     = getattr(paramTx, "Vb", -2.2)
    paramTx.Pi_dBm = getattr(paramTx, "Pi_dBm", 0)
    
    # Parâmetros da portadora elétrica
    paramTx.H  = getattr(paramTx, "H", 0.35/(2*pi))
    paramTx.fc = getattr(paramTx, "fc", 1e9)
    paramTx.A  = getattr(paramTx, "A", 1)

    Scheme = paramTx.Scheme
    SpS = paramTx.SpS
    Rs  = paramTx.Rs
    Fc  = paramTx.Fc
    Fa  = paramTx.Fa       # Sampling frequency
    Ta  = 1/Fa             # Sampling period
    
    M    = paramTx.M
    Nfft = paramTx.Nfft
    Ns   = paramTx.Ns
    N    = paramTx.N
    Nz   = paramTx.Nz
    G    = paramTx.G
    K    = paramTx.K
    
    H  = paramTx.H
    fc = paramTx.fc
    A  = paramTx.A
    
    g      = paramTx.g
    Vπ     = paramTx.Vπ
    Vb     = paramTx.Vb
    Pi_dBm = paramTx.Pi_dBm
    Pi     = 10**(Pi_dBm/10)*1e-3
    
    # Symbols constellation definiton
    constType = 'qam'
    constSymb = GrayMapping(M, constType)
    bitMap = demodulateGray(constSymb, M, constType)
    bitMap = bitMap.reshape(-1, int(np.log2(M)))

    # Random bits sequency
    bits = np.random.randint(2, size = Ns*2**9) #((Nfft-K-2)//2)*2**7
    
    # Maping bits - symbols
    symbTx = modulateGray(bits, M, constType)
    symbTx = pnorm(symbTx)
    
    # Pilot carriers
    pilot = max(symbTx.real) + 1j*max(symbTx.imag)
    
    # OFDM symbols generation
    symbTx_OFDM, symbTx = modulateOFDM(Nfft, Ns, N, Nz, G, K, pilot, symbTx)
    
    # Pulse choice
    pulse = pulseShape('rrc', SpS, alpha = 0.15)
    pulse = pulse/max(abs(pulse))
    
    # CE-DD-OFDM
    if(Scheme == "CE-DDO-OFDM"):
        # Pulse formatation
        sigTx = firFilter(pulse, upsample(symbTx_OFDM.real, SpS))
        sigTx = pnorm(sigTx)
        t = np.arange(0, sigTx.size)*Ta
        # Tirei o pnorm de sigTx_CE e adicionei o /SpS no sigTx
        
        # Optical modulation
        sigTx_CE = A*np.cos(2*pi*fc*t + 2*pi*H*(sigTx.real))
        Ai = np.sqrt(Pi) * np.ones(len(sigTx))
        sigTxo = mzm(Ai, sigTx_CE, Vπ, Vb)
        sigSig = sigTx_CE
    
    # DDO-OFDM
    else:
        # Pulse formatation
        sigTx = firFilter(pulse, upsample(symbTx_OFDM.real, SpS))
        sigTx = pnorm(sigTx)
        t = np.arange(0, sigTx.size)*Ta
        
        # Optical modulation
        sigTx_DD = g*sigTx.real*np.cos(2*pi*fc*t)
        Ai = np.sqrt(Pi) * np.ones(len(sigTx_DD))
        sigTxo = mzm(Ai, sigTx_DD, Vπ, Vb)
        sigSig = sigTx_DD
    
    return sigTxo, sigTx, sigSig, symbTx, t, pulse, pilot


def Rx(ipd, pilot, pulse, t, paramRx):
    """
    OFDM receiver (Rx).
    Parameters
    ----------
    paramRx : parameter object (struct), optional
        Parameters of the OFDM receiver.
        
    paramRx.SpS: samples per symbol [default: 32]
    paramRx.Fa: sampling frequency [default: 48e9]
    paramRx.H: phase modulation parameter [default: 0.35/(2*pi)]
    paramRx.fc: electrical carrier frequency [Hz] [default: 1e9 Hz]
    paramRx.Scheme: OFDM scheme ["CE-DDO-OFDM", "DDO-OFDM"] [default: "CE-DDO-OFDM"]

    paramTx.Nfft: size of IFFT [default: 512]
    paramTx.Ns: number of subcarriers [default: 255]
    paramTx.N: number of information subcarriers [default: 255]
    paramTx.Nz: number of null subcarriers [default: 0]
    paramRx.G: cyclic prefix length [default: 4]
    paramRx.K: number of pilot carriers per OFDM block [default: 8]
    
    Returns
    -------
    symbRx     : complex-valued np.array
                 symbols sequency received
    symbRx_neq : complex-valued np.array
                 symbols sequency received with no equalization
    sigRx      : np.array
                 received signal after processing
    H_abs      : np.array
                 channel amplitude estimated 
    H_pha      : np.array
                 channel phase estimated
    """
        
    paramRx.SpS = getattr(paramRx, "SpS", 32)
    paramRx.Fa  = getattr(paramRx, "Fa", 48e9)
    paramRx.H   = getattr(paramRx, "H", 0.35/(2*pi))
    paramRx.fc  = getattr(paramRx, "fc", 1e9)
    paramRx.Scheme = getattr(paramRx, "Scheme", "CE-DDO-OFDM")  
    
    paramRx.Nfft = getattr(paramRx, "Nfft", 512)
    paramRx.Ns   = getattr(paramRx, "Ns", 255)
    paramRx.N    = getattr(paramRx, "N", 255)
    paramRx.Nz   = getattr(paramRx, "Nz", 0)
    paramRx.G = getattr(paramRx, "G", 4)
    paramRx.K = getattr(paramRx, "K", 8)
    
    # Receiver parameters
    SpS = paramRx.SpS
    Scheme = paramRx.Scheme
    Fa  = paramRx.Fa
    H   = paramRx.H
    fc  = paramRx.fc
    
    Nfft = paramRx.Nfft
    Ns   = paramRx.Ns
    N    = paramRx.N
    Nz   = paramRx.Nz
    G = paramRx.G
    K = paramRx.K
    
    # Decimation parameters
    paramDec = parameters()
    paramDec.SpS_in = SpS
    paramDec.SpS_out = 1
    
    # DC level extraction
    I_Rx = ipd - ipd.mean()
    I_Rx = I_Rx/np.std(I_Rx)
    
    # CE-DDO-OFDM
    if(Scheme == "CE-DDO-OFDM"):
        signal_a = hilbert(I_Rx) * np.exp(-1j*2*pi*fc*t)
        sigRx = np.unwrap((np.arctan2(signal_a.imag, signal_a.real)), axis = 0)/(2*pi*H)
        sigRx = sigRx - sigRx.mean()
        
        # Seleção das amostras do sinal recebido 
        sigRx = firFilter(pulse, sigRx)
        sigRx = pnorm(sigRx)
        symbRx_OFDM = decimate(sigRx.reshape(-1,1), paramDec) # downsampling to 1 sample per symbol
        symbRx_OFDM = np.squeeze(symbRx_OFDM)
        symbRx_OFDM = pnorm(symbRx_OFDM)
    
    # DD-OFDM
    else:
        sigRx = firFilter(pulse, I_Rx*np.cos(2*pi*fc*t))
        sigRx = pnorm(sigRx)
        symbRx_OFDM = decimate(sigRx.reshape(-1,1), paramDec) # downsampling to 1 sample per symbol
        symbRx_OFDM = np.squeeze(symbRx_OFDM)
        symbRx_OFDM = pnorm(symbRx_OFDM)
    
    # OFDM demodulation
    symbRx, symbRx_neq, H_abs, H_pha = demodulateOFDM(Nfft, Ns, N, Nz, G, K, pilot, symbRx_OFDM)
    
    return symbRx, symbRx_neq, sigRx, H_abs, H_pha