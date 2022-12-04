import sounddevice
from scipy.io.wavfile import write

fs = 45000

second = int(input("Enter the time duration in seconds: "))
print("Recording...\n")
record_voice = sounddevice.rec(int(second*fs), samplerate=fs, channels=2)
sounddevice.wait()
write("recoding.wav", fs, record_voice)
print("Finished...\nCheck It Out...\n")