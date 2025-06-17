import struct
import argparse
import os
import string
import traceback
import numpy

from sklearn.model_selection import train_test_split
from tensorflow import math as tf_math
from tensorflow import reduce_mean
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

WINDOW_SIZE = 64

def read_3d_int_slice_binary(filename):
    data = []
    with open(filename, "rb") as f:
        # Read number of outer slices
        outer_len_bytes = f.read(4)
        if not outer_len_bytes:
            raise EOFError("Unexpected end of file while reading outer length")
        outer_len = struct.unpack("<i", outer_len_bytes)[0]

        for _ in range(outer_len):
            # Read number of inner slices
            inner_len_bytes = f.read(4)
            if not inner_len_bytes:
                raise EOFError("Unexpected end of file while reading inner length")
            inner_len = struct.unpack("<i", inner_len_bytes)[0]

            inner_list = []

            for _ in range(inner_len):
                # Read number of elements in this slice
                leaf_len_bytes = f.read(4)
                if not leaf_len_bytes:
                    raise EOFError("Unexpected end of file while reading leaf length")
                leaf_len = struct.unpack("<i", leaf_len_bytes)[0]

                # Read leaf_len int32 values
                leaf_data_bytes = f.read(4 * leaf_len)
                if len(leaf_data_bytes) != 4 * leaf_len:
                    raise EOFError("Unexpected end of file while reading leaf data")

                leaf = list(struct.unpack(f"<{leaf_len}f", leaf_data_bytes))
                inner_list.append(leaf)

            data.append(inner_list)

    return data



def get_input_target_pairs_from_binary(binaryPath: str):
    # binary file 
    allData = read_3d_int_slice_binary(binaryPath)
    # [[[input], [target]],
    #  [[input], [target]],
    # ]
    # we're going to use the front half of the data for training, the back half for learning initially
    midPoint = len(allData) // 2

    allInput = [i[0] for i in allData]
    allOutput = [i[1] for i in allData]

    allInput_np = numpy.array(allInput)
    allOutput_np = numpy.array(allOutput)
    return allInput_np, allOutput_np


# given that the sum of values in y_true might be some number < 1, the output of this function
# can be a very large number. After that, if it locks on 1.00, it means that it has 0'd the y_pred
# because that's the best it can figure out
def noiseToSignalLoss(y_true, y_pred):
    losses = tf_math.divide(
        tf_math.reduce_sum(
            tf_math.pow(
                tf_math.abs(
                    tf_math.subtract(
                        y_true,
                        y_pred
                    )
                ),
                2
            )
        ),
        tf_math.reduce_sum(
            tf_math.pow(tf_math.abs(y_true),2)
        )
    )
    return reduce_mean(losses)

def main():
    parser = argparse.ArgumentParser(description="Actually generate a model.")
    parser.add_argument("numEpochs", help="Number of epochs")
    parser.add_argument("binaryPath", help="Path to learning data binary")
    
    args = parser.parse_args()

    binaryPath = args.binaryPath
    numEpochs = int(args.numEpochs)

    if not os.path.isfile(binaryPath):
        print(f"Error: '{binaryPath}' does not exist or is not a file.")

    # unpack and split testing and training data
    X, y = get_input_target_pairs_from_binary(binaryPath)
    X_train, X_test, y_train, y_test = train_test_split(X,
													y,
													test_size=.2)

    # Generate the network
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=WINDOW_SIZE))
    model.add(Dense(units=64, activation='relu'))

    # ideally this returns -1 to 1 instead of 0 to 1, but we can likely normalize outputs ourself
    model.add(Dense(units=WINDOW_SIZE, activation='sigmoid')) 

    # Compile the model duh
    model.compile(loss=noiseToSignalLoss, optimizer='sgd', metrics=['accuracy'])
    # Training
    try:
        model.fit(X_train, y_train, epochs=numEpochs, batch_size=32)
    except Exception as e:
        # Errors can take the form of printing the entire dataset, so we print them to a file
        # so that we can actually see the traceback
        with open('./failure_output/exception', mode='w') as file:
            # Get the traceback information as a string
            traceback_str = traceback.format_exc()
            # Write the exception type, message, and traceback to the file
            file.write(f"Exception Type: {type(e).__name__}\n")
            file.write(f"Exception Message: {e}\n")
            file.write(f"Traceback:\n{traceback_str}\n")
            file.write("-" * 20 + "\n") 
        print("we hit a error :( ", type(e))

    y_hat = model.predict(X_test)
    print(y_hat[0][0])
    #print(max(y_hat.all()))
    #accuracy_score(y_test, y_hat)

    model.save("./model/model.keras")
    return


if __name__ == "__main__":
    main()