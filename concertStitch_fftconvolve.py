import os
import subprocess

import imageio
import librosa
import numpy as np
from scipy.signal import fftconvolve

# Gather all the videos from the server. Here, gather all the videos inside the folder "videos"
v_source = []
for file in os.listdir("shimon/"):
    if file.endswith(".MOV"):
        v_source.append(file)
v_source.sort()
a_source = []

print("Source_Videos: ", v_source)

# Extract the audio from all the videos and save it inside a folder called "audio_extracts"
if "Audio_Extracts" not in os.listdir("."):
    os.mkdir("Audio_Extracts")

for i in range(len(v_source)):
    a_source.append("aSource_" + str(i) + ".wav")
    command = "ffmpeg -i shimon/" + v_source[i] + " -ab 160k -ac 1 -ar 44100 -vn Audio_Extracts/aSource_" + str(i) + ".wav"
    subprocess.call(command, shell=True)

print("Extracted audios", a_source)

# import the High Quality Audio from the mixer output and down sample it to reduce dimensionality

# command = "ffmpeg -i shimon/mainCut.mp4 -ab 160k -ac 1 -ar 44100 -vn HQ_Audio.wav"
# subprocess.call(command, shell=True)
HQ_audio = "HQ_Audio.wav"
sr = 8000

print("loading audio...")
audio, fs = librosa.load(HQ_audio, sr=44100, mono=True)
audio = librosa.resample(audio, fs, sr)
print('no of samples of HQ_Audio:', len(audio))

if "Audio_Sync_Outputs" not in os.listdir("."):
    os.mkdir("Audio_Sync_Outputs")

source_sync_point = []


def correlate(signal, signal_1):
    Axy = fftconvolve(signal, signal_1[::-1], 'full')
    M = signal_1.shape[0]
    return np.argmax(Axy) - M + 1
    # N + M - 1


for i in range(len(a_source)):
    x, _ = librosa.load("Audio_Extracts/" + a_source[i], sr=fs)
    x_low = librosa.resample(x, fs, sr)
    max_index = correlate(audio, x_low)
    source_sync_point.append(max_index)
    max_index = int(np.round(max_index * fs / sr))
    zeros = np.zeros(np.abs(max_index))
    if max_index >= 0:
        out = np.concatenate((zeros, x), axis=0)
    else:
        out = x[np.abs(max_index):]
    string = 'Audio_Sync_Outputs/sync_' + str(i) + '.wav'
    librosa.output.write_wav(string, out, fs)
    print("Axy", i, "done!")
    print(max_index / float(fs))

FRAMES_PER_SECOND = 25
source_sync_point = np.array(source_sync_point) / float(sr)
source_sync_frame = np.round(source_sync_point * FRAMES_PER_SECOND).astype(int)
print("sync frames:", source_sync_frame)

source = []

for s in v_source:
    source.append(np.array(imageio.mimread("shimon/" + s, memtest=False)))

width = source[0].shape[2]
height = source[0].shape[1]

sr = 8000

TOTAL_LENGTH = int(np.round(len(audio) / float(sr)) * FRAMES_PER_SECOND)
print("Total length:", TOTAL_LENGTH)

temp = []
for i in range(len(source)):
    temp.append((source[i].shape[0] + source_sync_frame[i]))
source_end_sync = np.array(temp)
print("end sync frames:", source_end_sync)

frame_width = 640
frame_height = 360

black_video = np.zeros((TOTAL_LENGTH, frame_height, frame_width, 3), dtype=np.uint8)

print("black video created")
###########################################################################
# This part is not required for release... just for the demo
# BEWARE!!! This part takes the most time and energy...

source_full = []

if "Full_Videos" not in os.listdir("."):
    os.mkdir("Full_Videos")

for i in range(len(source_sync_frame)):
    temp = np.array(black_video[:source_sync_frame[i]])
    temp = np.append(temp, source[i], axis=0)
    source_full.append(np.append(temp, black_video[source_end_sync[i]:], axis=0))
    imageio.mimwrite("Full_Videos/source_full_" + str(i) + ".mov", source_full[i], fps=FRAMES_PER_SECOND, macro_block_size=None)

###########################################################################

# The dice algorithm

NUMBER_OF_SECONDS_CUT = 2

output = np.copy(black_video)

print("sources read successfully")


def return_space(frame_count):
    samples = []
    for j in range(len(source_sync_frame)):
        if source_end_sync[j] >= (frame_count + (FRAMES_PER_SECOND * NUMBER_OF_SECONDS_CUT)):
            if source_sync_frame[j] < frame_count:
                samples.append(j)
    return samples


dice = 0
for i in range(TOTAL_LENGTH):
    space = return_space(i)
    if len(space) != 0:
        if i % (NUMBER_OF_SECONDS_CUT * FRAMES_PER_SECOND) == 0:
            dice = np.random.choice(space)
            print(dice)
        output[i] = source_full[dice][i]
    i += 1

print("Shape of stitched video:", output.shape)
imageio.mimwrite('stitched.mov', output.astype(np.uint8), fps=FRAMES_PER_SECOND, macro_block_size=None)

# Fades
FADE_TIME = 2  # seconds

end = TOTAL_LENGTH - 1
while np.count_nonzero(output[end]) == 0:
    end -= 1

start = 0
while np.count_nonzero(output[start]) == 0:
    start += 1

fadeFrames = FADE_TIME * FRAMES_PER_SECOND
fade_func = np.linspace(0, 1, fadeFrames)

for i in range(end, (end - fadeFrames), -1):
    output[i] = output[i] * fade_func[end - i]

for i in range(start, (start + fadeFrames)):
    output[i] = output[i] * fade_func[i - start]

imageio.mimwrite('stitched_fade.mov', output, fps=FRAMES_PER_SECOND)

cmd = 'ffmpeg -y -i HQ_Audio.wav  -r 30 -i stitched.mov  -filter:a aresample=async=1 -c:a aac -c:v copy stitched_av.mov'
subprocess.call(cmd, shell=True)
