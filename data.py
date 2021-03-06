import os
import random
from scipy.io import wavfile
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_categories(path):
	categories = dict.fromkeys(os.listdir(path))

	for c in categories:
		c_abspath = os.path.join(path, c)
		audio_files = filter(lambda n: n.endswith(".wav"), os.listdir(c_abspath))
		categories[c] = [os.path.join(c_abspath, au) for au in audio_files]

	return categories

def random_wav_slice(path, slice_size, debug=False, category="bla", downsample=7):
    sample_rate, data = wavfile.read(path)
    data = data[::7]
    slice_start = random.randint(0, data.shape[0] - slice_size - 1)
    slice_end = slice_start + slice_size
    convert_16_bit = float(2**15)
    data = data[slice_start:slice_end]/(convert_16_bit+1.0)
    data = data.astype(np.float32)
    return data

def random_sample(categories, slice_size=20000, debug=False, batch_size=128):
    sample_tensors = []
    target_tensors = []

    for batch_i in xrange(batch_size):

        category = random.choice(list(categories.keys()))				
        category_index = sorted(list(categories.keys())).index(category)

        random_file = random.choice(list(categories[category]))			
        sample_data = random_wav_slice(random_file, slice_size, category=category)

        target_tensors.append(category_index)
        sample_tensors.append(sample_data)

    sample_tensors = torch.from_numpy(np.array(sample_tensors, dtype=np.float32))
    target_tensors = torch.from_numpy(np.array(target_tensors, dtype=np.int64))

    return target_tensors, sample_tensors


def load_wav(style, num):
    wav_path = os.path.join("../../../datasets/gtzan/", style, style+"."+str(num).zfill(5)+".wav")
    arr = pywav.read(wav_path)[1].astype(np.float32)
    arr /= max(abs(arr.max()), abs(arr.min()))
    return arr

