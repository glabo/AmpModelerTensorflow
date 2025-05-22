import struct
import argparse
import os
import string

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

                leaf = list(struct.unpack(f"<{leaf_len}i", leaf_data_bytes))
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

    X_train = [i[0] for i in allData[:midPoint]]
    y_train = [i[1] for i in allData[:midPoint]]
    x_test = [i[0] for i in allData[midPoint:]]
    y_test = [i[1] for i in allData[midPoint:]]
    return X_train, x_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description="Actually generate a model.")
    parser.add_argument("binaryPath", help="Path to learning data binary")
    
    args = parser.parse_args()

    binaryPath = args.binaryPath

    if not os.path.isfile(binaryPath):
        print(f"Error: '{binaryPath}' does not exist or is not a file.")

    X_train, x_test, y_train, y_test = get_input_target_pairs_from_binary(binaryPath)
    return


if __name__ == "__main__":
    main()