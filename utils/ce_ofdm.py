import numpy as np
from numpy.random      import normal
from utils.dsp         import highPassFIR

from scipy.fftpack     import fft, ifft
from scipy.constants   import e, k, pi
from scipy.signal      import hilbert
from scipy.interpolate import interp1d

from optic.modulation  import modulateGray, demodulateGray, GrayMapping
from optic.dsp         import pulseShape, lowPassFIR, firFilter
from optic.metrics     import signal_power
from optic.models      import mzm

from commpy.utilities  import upsample

def modulateOFDM(N, G, K, pilot, symbTx):
    Int = int((N/2)/K) 
    carriers = np.arange(0, int(N/2))
    pilot_carriers = np.append(carriers[1::Int], carriers[-1])
    
    symbTx_P = np.zeros((int(len(symbTx)/(N/2)), N + G), complex)
    aux = 0

    for i in range(len(symbTx_P)):
        # Adição das portadoras piloto
        symbTx[int(aux):int(aux+N/2)][pilot_carriers] = pilot
        symbTx_P[i,G:G+int(N/2)] = symbTx[int(aux):int(aux+N/2)]
        aux = aux + N/2
                
        # Simetria hermitiana
        v = symbTx_P[i,G]
        symbTx_P[i,G] = v.real
        symbTx_P[i,G + int(N/2)] = v.imag 

        for j in range(int(N/2)-1):
            j = j + 1
            symbTx_P[i,G + int(N) - j] = np.conjugate(symbTx_P[i,G + j])
        
        # Aplicação da IFFT
        symbTx_P[i,G:N+G] = ifft(symbTx_P[i,G:N+G])*np.sqrt(N)

        # Adição do prefixo cíclico
        symbTx_P[i,:] = np.concatenate((symbTx_P[i,N:N+G],symbTx_P[i,G:N+G]))
    
    # Conversão Paralelo -> Serial
    symbTx_OFDM = np.squeeze(symbTx_P.reshape(1,len(symbTx_P[0])*len(symbTx_P)))
    
    return symbTx_OFDM, symbTx


def demodulateOFDM(N, G, K, pilot, symbRx_OFDM):
    Int = int((N/2)/K) 
    carriers = np.arange(0, int(N/2))
    pilot_carriers = np.append(carriers[1::Int], carriers[-1])
    
    # Conversão S/P
    symbRx_P = np.zeros((int(len(symbRx_OFDM)/(N+G)), N), complex)
    aux = 0
    
    for i in range(len(symbRx_P)):
        # Extração do prefixo cíclico
        symbRx_P[i,0:N] = symbRx_OFDM[aux+G:aux+N+G]
        aux = aux + N + G

        # Aplicação da FFT
        symbRx_P[i,:] = fft(symbRx_P[i,:])/np.sqrt(N)

    # Conversão P/S
    symbRx_S = np.zeros(int(len(symbRx_P[0])*len(symbRx_P)/2), complex)  # Com eq.
    symbRx_S_neq = np.zeros(int(len(symbRx_P[0])*len(symbRx_P)/2), complex)  # Sem eq.
    aux = 0

    for i in range(len(symbRx_P)):
        # Estimação do canal
        H_est = symbRx_P[i,0:int(N/2)][pilot_carriers] / pilot
        
        H_abs = interp1d(carriers[pilot_carriers], np.abs(H_est), kind = 'cubic')(carriers[1:int(N/2)])
        H_pha = interp1d(carriers[pilot_carriers], np.angle(H_est), kind = 'cubic')(carriers[1:int(N/2)])
        
        H_abs = np.pad(H_abs, (1,0), 'constant', constant_values = H_abs[0])
        H_pha = np.pad(H_pha, (1,0), 'constant', constant_values = H_pha[0])
        
        # Retirada da simetria hermitiana e equalização
        symbRx_S[int(aux):int(aux + N/2)] = symbRx_P[i,0:int(N/2)] / (H_abs * np.exp(1j*H_pha) )
        symbRx_S_neq[int(aux):int(aux + N/2)] = symbRx_P[i,0:int(N/2)] #/ (H_abs * np.exp(1j*H_pha) )
        aux = aux + N/2

    symbRx = symbRx_S
    symbRx_neq = symbRx_S_neq
    
    return symbRx, symbRx_S_neq, H_abs, H_pha

def Tx(paramTx):
    # Parâmetros da transmissão
    paramTx.SpS = getattr(paramTx, "SpS", 32)       # Amostras por símbolo
    paramTx.Rs  = getattr(paramTx, "Rs", 1.5e9)     # Taxa de símbolo
    paramTx.Ts  = getattr(paramTx, "Ts", 1/1.5e9)   # Tempo de símblo
    paramTx.Fa  = getattr(paramTx, "Fa", 48e9)      # Frequência de amostragem
    paramTx.Ta  = getattr(paramTx, "Ta", 20.83e-12) # Período de amostragem
    paramTx.Fc  = getattr(paramTx, "Fc", 193.4e12)  # Frequência da portadora óptica
    
    # Parâmetros do esquema OFDM
    paramTx.M = getattr(paramTx, "M", 16)   # Número de símbolos da constelação QAM
    paramTx.N = getattr(paramTx, "N", 1024) # Dobro do número de sub-portadoras
    paramTx.G = getattr(paramTx, "G", 8)    # Tamanho do prefixo cíclico
    paramTx.K = getattr(paramTx, "K", 16)   # Número de portadoras piloto por bloco OFDM

    # Parâmetros da portadora elétrica
    paramTx.H  = getattr(paramTx, "H", 0.35/(2*pi)) # Índice de modulação
    paramTx.fc = getattr(paramTx, "fc", 1e9)        # Frequência da portadora [Hz]
    paramTx.A  = getattr(paramTx, "A", 1)           # Amplitude da portadora

    # Parâmetros da portadora óptica
    paramTx.Vπ    = getattr(paramTx, "Vπ", 400)    # Tensão Vπ do MZM
    paramTx.Vb    = getattr(paramTx, "Vb", 400)    # Tensão Vbias do MZM
    paramTx.Pi_dBm = getattr(paramTx, "Pi_dBm", 0)  # Potência óptica do sinal [dBm]

    SpS = paramTx.SpS
    Rs  = paramTx.Rs
    Ts  = paramTx.Ts
    Fa  = paramTx.Fa
    Ta  = paramTx.Ta
    Fc  = paramTx.Fc

    M = paramTx.M
    N = paramTx.N
    G = paramTx.G
    K = paramTx.K
    
    H  = paramTx.H
    fc = paramTx.fc
    A  = paramTx.A
    
    Vπ     = paramTx.Vπ
    Vb     = paramTx.Vb
    Pi_dBm = paramTx.Pi_dBm
    
    # Definição da constelação de símbolos
    constType = 'qam'
    constSymb = GrayMapping(M, constType)
    bitMap = demodulateGray(constSymb, M, constType)
    bitMap = bitMap.reshape(-1, int(np.log2(M)))

    # Geração de sequência aleatória de bits
    bits = np.random.randint(2, size = 6*2**15)

    # Mapeamento bits - símbolos
    symbTx = modulateGray(bits, M, constType)
    symbTx = symbTx/np.sqrt(signal_power(symbTx))
    
    # Portadoras piloto
    pilot = -0.7071067811865475+1j*0.7071067811865475 #0.9475658169809407-0.9475658169809407j
    
    # Geração dos símbolos OFDM
    symbTx_OFDM, symbTx = modulateOFDM(N, G, K, pilot, symbTx)
    
    # Formatação de pulso
    pulse = pulseShape('rrc', SpS, alpha = 0.15)
    pulse = pulse/max(abs(pulse))

    sigTx = firFilter(pulse, upsample(symbTx_OFDM, SpS))
    t = np.arange(0, sigTx.size)*Ta
    
    # Geração do sinal CE-DD-OFDM
    sigTx_CE = A*np.cos(2*pi*fc*t + 2*pi*H*sigTx)
    
    # Modulação óptica
    Pi     = 10**(Pi_dBm/10)*1e-3
    Ai     = np.sqrt(2*Pi) * np.ones(len(sigTx))
    sigTxo = mzm(Ai, sigTx_CE, Vπ, Vb)
    
    return sigTxo, symbTx, t, pulse, pilot

def Rx(sigRxo, pilot, pulse, t, paramRx):
    paramRx.Tc = getattr(paramRx, "Tc", 25)   # Temperatura [°C]
    paramRx.Rd = getattr(paramRx, "Rd", 0.85) # Responsividade
    paramRx.Id = getattr(paramRx, "Id", 5e-9) # Corrente de escuro [A]
    paramRx.RL = getattr(paramRx, "RL", 50)   # Resistência [Ω]
    paramRx.B  = getattr(paramRx, "B", 10e9)  # Largura de banda [Hz]

    paramRx.Fa  = getattr(paramRx, "Fa", 48e9)       # Frequência de amostragem
    paramRx.SpS = getattr(paramRx, "SpS", 32)        # Amostras por símbolo
    paramRx.H   = getattr(paramRx, "H", 0.35/(2*pi)) # Índice de modulação
    paramRx.fc  = getattr(paramRx, "fc", 1e9)        # Frequência da portadora elétrica
    
    paramRx.N = getattr(paramRx, "N", 1024)
    paramRx.G = getattr(paramRx, "G", 8)
    paramRx.K = getattr(paramRx, "K", 16)
    
    # Parâmetros do receptor
    Tc = paramRx.Tc
    Rd = paramRx.Rd
    Id = paramRx.Id
    RL = paramRx.RL
    B  = paramRx.B

    Fa  = paramRx.Fa
    SpS = paramRx.SpS
    H   = paramRx.H
    fc  = paramRx.fc
    
    N = paramRx.N
    G = paramRx.G
    K = paramRx.K
    
    Pin = (np.abs(sigRxo)**2).mean()   # Potência óptica média recebida
    Ip  = Rd*np.abs(sigRxo)**2         # Fotocorrente livre de ruído do receptor

    # Ruído de disparo 
    σ2_s = 2*e*(Rd*Pin + Id)*B  # Variância do ruído de disparo 

    # Ruído térmico
    T    = Tc + 273.15   # Temperatura em Kelvin
    σ2_T = 4*k*T*B/RL    # Variância do ruído térmico

    # Adiciona ruído do receptor p-i-n aos sinais
    Is = normal(0, np.sqrt(Fa*(σ2_s/(2*B))), Ip.size)
    It = normal(0, np.sqrt(Fa*(σ2_T/(2*B))), Ip.size)  
    
    I = Ip + Is + It

    # FPB (Resposta do fotodetector)
    I_Rx = firFilter(lowPassFIR(B, Fa, 8000, 'rect'), I)

    # FPA (Retirada do nível DC)
    I_Rx = firFilter(highPassFIR(0.1e9, Fa, 8001), I_Rx)
    
    # Demodulação da fase
    signal_a = firFilter(pulse/SpS, hilbert(I_Rx) * np.exp(-1j*2*pi*fc*t))
    Θ        = (np.arctan(signal_a.imag/signal_a.real))/(2*pi*H)
    
    # Seleção das amostras do sinal recebido
    symbRx_OFDM = Θ[0::SpS]
    
    # Demodulação OFDM
    symbRx, symbRx_neq, H_abs, H_pha = demodulateOFDM(N, G, K, pilot, symbRx_OFDM)
    
    return symbRx, symbRx_neq, H_abs, H_pha