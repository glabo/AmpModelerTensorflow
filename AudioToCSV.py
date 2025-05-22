
# Need a csv library, and to take 2 file paths as input, DI and Output
import argparse
import os
import sys
import numpy as np
from scipy.io import wavfile
import csv

# we're going to use blocks of WINDOW_SIZE samples for learning comparison
WINDOW_SIZE = 64

def wav_to_csv(di_wav_path, output_wav_path, csv_path):
    # Read the .wav file
    try:
        di_sample_rate, di_data = wavfile.read(di_wav_path)
    except Exception as e:
        raise ValueError(f"Failed to read WAV file: {e}")
	
    try:
        output_sample_rate, output_data = wavfile.read(output_wav_path)
    except Exception as e:
        raise ValueError(f"Failed to read WAV file: {e}")

    # Print some info
    print(f"Sample Rate: {di_sample_rate} Hz")
    print(f"Data shape: {di_data.shape}")
    print(f"Data type: {di_data.dtype}")

	# Verify this stuff all matches
	assert(di_sample_rate == output_sample_rate)
	assert(di_data.shape == output_data.shape)
	assert(di_data.dtype == output_data.dtype)
	assert(len(di_data) == len(output_data))
	assert(di_data.ndim == 1 and output_data.ndim == 1)

	# divide by 32768 to normalize samples to (-1,1)

    # Write to CSV
    try:
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
			for i in len(di_data)-WINDOW_SIZE:
				windowEndpoint = i + WINDOW_SIZE
				writer.writerow([di_data[i:windowEndpoint],
								 output_data[i:windowEndpoint]])
    except Exception as e:
        raise IOError(f"Failed to write CSV file: {e}")

    print(f"CSV written to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Process two file paths.")
    parser.add_argument("DI", help="Path to guitar DI .wav file")
    parser.add_argument("Output", help="Path to post-amp .wav file")

    args = parser.parse_args()

    diFile = args.DI
    outputFile = args.Output

    if not os.path.isfile(file1):
        print(f"Error: '{file1}' does not exist or is not a file.")
        sys.exit(1)

    if not os.path.isfile(file2):
        print(f"Error: '{file2}' does not exist or is not a file.")
        sys.exit(1)

	wav_to_csv(diFile.path, outputFile.path, "./data")

if __name__ == "__main__":
    main()
