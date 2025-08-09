import pickle
from scipy.io import wavfile
import argparse
import numpy as np

from tf_model import PedalNetTF
from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
import os


def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def test(args):
    model = load_model(args.model)
    data = pickle.load(open(os.path.dirname(args.model) + "/data.pickle", "rb"))

    x_test = data["x_test"]
    prev_sample = np.concatenate((np.zeros_like(x_test[0:1]), x_test[:-1]), axis=0)
    pad_x_test = np.concatenate((prev_sample, x_test), axis=1)

    y_pred = []
    for x in np.array_split(pad_x_test, 10):
        prediction = model.predict(convert_to_tensor(x))
        y_pred.append(prediction)

    y_pred = np.concatenate(y_pred)
    y_pred = y_pred[:, -x_test.shape[1] :, :]

    save(os.path.dirname(args.model) + "/y_pred.wav", y_pred)
    save(os.path.dirname(args.model) + "/x_test.wav", data["x_test"] * data["std"] + data["mean"])
    save(os.path.dirname(args.model) + "/y_test.wav", data["y_test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pedalnet_tf/pedalnet_model.keras")
    args = parser.parse_args()
    test(args)
