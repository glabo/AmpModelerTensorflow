import pickle
from scipy.io import wavfile
import argparse
import numpy as np

from tf_model import PedalNetTF
from tqdm import tqdm
from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
import os


def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def test(args):
    model = load_model(args.model)
    train_data = pickle.load(open(os.path.dirname(args.model) + "/data.pickle", "rb"))

    mean, std = train_data["mean"], train_data["std"]

    in_rate, in_data = wavfile.read(args.input)
    assert in_rate == 44100, "input data needs to be 44.1 kHz"
    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    # split into samples
    in_data = in_data[:length].reshape((-1, sample_size, 1)).astype(np.float32)

    # standardize
    in_data = (in_data - mean) / std

    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=1)

    pred = []
    batches = pad_in_data.shape[0]
    for x in tqdm(np.array_split(pad_in_data, batches)):
        prediction = model.predict(convert_to_tensor(x))
        pred.append(prediction)

    pred = np.concatenate(pred)
    pred = pred[:, -in_data.shape[1] :, :]

    save(args.output, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pedalnet_tf/pedalnet_model.keras")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    test(args)
