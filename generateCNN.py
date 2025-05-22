import struct
import argparse
import os
import string

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import accuracy_score

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
    return allInput, allOutput

def main():
    parser = argparse.ArgumentParser(description="Actually generate a model.")
    parser.add_argument("numEpochs", help="Number of epochs")
    parser.add_argument("binaryPath", help="Path to learning data binary")
    
    args = parser.parse_args()

    binaryPath = args.binaryPath
    numEpochs = args.numEpochs

    if not os.path.isfile(binaryPath):
        print(f"Error: '{binaryPath}' does not exist or is not a file.")

    # unpack and split testing and training data
    X, y = get_input_target_pairs_from_binary(binaryPath)
    print("X[0]: ", X[0])
    print("y[0]: ", y[0])
    # X_train, X_test, y_train, y_test = train_test_split(X,
	# 												y,
	# 												test_size=.2)

    # # Generate the network
    # model = Sequential()
    # model.add(Dense(units=32, activation='relu', input_dim=WINDOW_SIZE))
    # model.add(Dense(units=64, activation='relu'))

    # # ideally this returns -1 to 1 instead of 0 to 1, but we can likely normalize outputs ourself
    # model.add(Dense(units=WINDOW_SIZE, activation='sigmoid')) 

    # # Compile the model duh
    # model.compile(loss='ESR', optimizer='sgd', metrics='accuracy')
    # # Training
    # model.fit(X_train, y_train, epochs=numEpochs, batch_size=32)

    # y_hat = model.predict(X_test)
    # accuracy_score(y_test, y_hat)
    return


if __name__ == "__main__":
    main()