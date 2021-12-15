import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy
import statistics
from scipy.signal import butter,lfilter
import math

audio1, sr= librosa.load('trial_audio2.wav', sr=41000)

#filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b)
    #print(a)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
low_freq=100
high_freq=4000
fil_audio1=butter_bandpass_filter(audio1, low_freq,high_freq,sr,order=5)
#print(fil_audio)
#starting 100 ms
n0=0
n1=int(sr*(0.1))

#finding zero crossing rate for starting 100 ms(silence)
zc=librosa.zero_crossings(fil_audio1[n0:n1], pad=False)
zcr=librosa.feature.zero_crossing_rate(fil_audio1[n0:n1], pad=False)
zcr=zcr*n1
print("zero crossing rates for 100ms silence")
print(zcr)

#selecting threshold for zcr
min_zcr=250
mean_zcr=statistics.mean(zcr[0])
stdev_zcr=statistics.stdev(zcr[0])

zcr_threshold=min(min_zcr,(mean_zcr+2*stdev_zcr))
#print("zero crossing rate threshold: "+ str(zcr_threshold))


#10 ms
n2=int(0.01*sr)
rmse_audio=librosa.feature.rms(fil_audio1, frame_length=n2, hop_length=n2)[0]

#calculating lower and upper threshold
IMX=max(rmse_audio)
IMN=min(rmse_audio)
print(IMX)
print(IMN)
I1=0.03*(IMX-IMN) + IMN
I2=4*IMN
ITL=min(I1,I2)
ITU=5*(ITL)
print("Lower threshold: "+ str(ITL))
print("Upper threshold: "+ str(ITU))


#detecting beginning point of utterance
N1=0
print(len(rmse_audio))
for m in range(0, len(rmse_audio)):
    if(N1!=0):
        break
    e_m=rmse_audio[m]
    if(e_m>=ITL):
        for n in range(m,len(rmse_audio)):
            e_n=rmse_audio[n]
            if(e_n<ITU):
                break
            elif(e_n>=ITU):
                N1=n+1
                print("N1 found")
                if(n==m):
                    N1=N1-1
                    break
                else:
                    break  
            else:
                continue
    else:
        continue

m1=0
for i in range(N1,N1-26,-1):
    zcr_temp=librosa.feature.zero_crossing_rate(fil_audio1[N1*n2:N1*n2+1], pad=False)
    if(zcr_temp>=zcr_threshold):
        m1+=1
        zcr_last_index=i
if(m1>=3):
    N1=zcr_last_index        

print("Beginning of utterance is at:" + str(N1*n2/sr) + " seconds, and at frame: "+ str(N1*n2))


#Detecting endpoint of utterance
N2=0
for m in range(len(rmse_audio)-1,-1,-1):
    if(N2!=0):
        break
    e_m=rmse_audio[m]
    if(e_m>=ITL):
        for n in range(m,-1,-1):
            e_n=rmse_audio[n]
            if(e_n<ITL):
                break
            elif(e_n>=ITU):
                N2=n
                print("N2 found")
                if(n==m):
                    N2=N2+1
                    break
                else:
                    break
            else:
                continue            
    else:
        continue 

m2=0
for i in range(N2,N2+26):
    zcr_temp=librosa.feature.zero_crossing_rate(fil_audio1[N2*n2:N2*n2+1], pad=False)
    if(zcr_temp>=zcr_threshold):
        m2+=1
        zcr_last_index=i
if(m2>=3):
    N2=zcr_last_index

print("Endpoint of utterance is at:" + str(N2*n2/sr) + " seconds, and at frame: "+str(N2*n2))


#calculating stop closure threshold
sc_signal=0
square=0
for i in range(N1*n2,(N2*n2)+1):
    square+=fil_audio1[i]**2

mean_signal=square/((N2-N1)*n2 + 1)
sc_signal=math.sqrt(mean_signal)
sc_threshold=0.2*sc_signal
print("Stop closure threshold: " + str(sc_threshold))


#Detecting stop closure

S1=S2=0
arr=[]

samples_per_ms=int(sr/1000)
window_length_in_ms=10
samples_per_frame=samples_per_ms*window_length_in_ms

nFrames=int(len(fil_audio1)/samples_per_frame)

short_term_energies=[]
for k in range(nFrames):
    start_index=k*samples_per_frame
    end_index=start_index+samples_per_frame
    window=np.zeros(len(audio1))
    window[start_index:end_index]=1
    STE=sum((fil_audio1**2)*(window**2))
    short_term_energies.append(STE)


for p in range(N1,N2+1):
    if(short_term_energies[p]<=sc_threshold):
        arr.append(0)
    else:
        arr.append(1)    


temp=0
perm=0
for i in range(0,len(arr)):
    if(arr[i]==0):
        temp+=1
    else:
        if(temp>=perm):
            S1=i-temp+N1
            S2=i+N1
            perm=temp
            temp=0

print("Beginning of stop closure: "+ str(S1*samples_per_frame/sr)+"s, at frame no.: "+str(S1*samples_per_frame))
print("Endpoint of stop closure: "+ str(S2*samples_per_frame/sr) + "s, at frame no.: "+str(S2*samples_per_frame))

print("Amplitude of audio from S1 to S2:")
print(audio1[S1*samples_per_frame:S2*samples_per_frame])



#plotting graphs
plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.plot(short_term_energies)
plt.xlabel("Window number(10ms)")
plt.ylabel("Short term Energy")
plt.grid()

plt.subplot(2,2,2)
plt.plot(rmse_audio)
plt.xlabel("Window number(10ms)")
plt.ylabel("RMS energy")
plt.grid()

plt.subplot(2,2,3)
plt.plot(audio1)
plt.xlabel("Frames")
plt.ylabel("Amplitude")


plt.grid()
plt.show()