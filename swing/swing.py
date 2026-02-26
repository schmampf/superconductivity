import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# -----------------------
# Load .m4a (requires ffmpeg installed)
# -----------------------
path = "Duke Ellington - It don_t mean a thing (1943).m4a"
audio = AudioSegment.from_file(path)  # auto-detects format

# Convert to mono + get sample rate
audio = audio.set_channels(1)
fs = audio.frame_rate  # Hz

# pydub samples are integers; convert to float in [-1, 1]
x = np.array(audio.get_array_of_samples())
x = x.astype(np.float32)
x /= (
    np.iinfo(x.dtype).max
    if np.issubdtype(x.dtype, np.integer)
    else np.max(np.abs(x) + 1e-12)
)

# Optional: analyze only first N seconds to keep FFT size reasonable
seconds = 10
N = min(len(x), int(seconds * fs))
x = x[:N]

# -----------------------
# FFT (with windowing)
# -----------------------
w = np.hanning(len(x))
xw = x * w

X = np.fft.rfft(xw)
f = np.fft.rfftfreq(len(xw), d=1.0 / fs)

# Magnitude spectrum (linear) or dB
mag = np.abs(X)
mag_db = 20 * np.log10(mag + 1e-12)

# -----------------------
# Plot
# -----------------------
plt.figure()
plt.plot(f, mag_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"FFT magnitude: {path} (first {seconds}s, fs={fs} Hz)")
plt.xlim(0, min(20000, fs / 2))  # typical audio band
plt.grid(True, which="both", ls=":")
plt.show()
