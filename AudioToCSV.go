package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"slices"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

var WINDOW_SIZE = 64

func parseWavFile(filepath string) (*audio.IntBuffer, *wav.Decoder, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file %s: %v", filepath, err)
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)

	if !decoder.IsValidFile() {
		return nil, nil, fmt.Errorf("invalid WAV file: %s", filepath)
	}

	buffer, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode WAV file: %v", err)
	}

	return buffer, decoder, nil
}

func printWavInfo(buffer *audio.IntBuffer, decoder *wav.Decoder, name string) {
	fmt.Printf("File: %s\n", name)
	fmt.Printf("  NumSamples: %d\n", len(buffer.Data))
	fmt.Printf("  maxVal: %d\n", slices.Max(buffer.Data))
	fmt.Printf("  Sample Rate: %d\n", decoder.SampleRate)
	fmt.Printf("  Channels: %d\n", decoder.NumChans)
	fmt.Printf("  Bit Depth: %d\n", decoder.BitDepth)
	fmt.Printf("  Duration: %.2f seconds\n", float64(len(buffer.Data))/float64(decoder.SampleRate)/float64(decoder.NumChans))
	fmt.Println()
}

func generateIOPairs(outFilePath string, inputBuf *audio.IntBuffer, targetBuf *audio.IntBuffer) {
	numSamples := len(inputBuf.Data)
	numPairs := numSamples - WINDOW_SIZE
	outArray := make([][][]int, numPairs)
	for i := range numPairs {
		row := make([][]int, 2)
		winEnd := i + WINDOW_SIZE
		row[0] = inputBuf.Data[i:winEnd]
		row[1] = targetBuf.Data[i:winEnd]
		outArray[i] = row
	}

	write3DIntSliceBinary(outArray, outFilePath)
}

func write3DIntSliceBinary(data [][][]int, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	normalizeVal := math.Pow(2, 23) - 1

	// Write number of outer slices
	err = binary.Write(writer, binary.LittleEndian, int32(len(data)))
	if err != nil {
		return err
	}

	for _, twoD := range data {
		// Write number of inner slices
		err = binary.Write(writer, binary.LittleEndian, int32(len(twoD)))
		if err != nil {
			return err
		}

		for _, oneD := range twoD {
			// Write length of this slice
			err = binary.Write(writer, binary.LittleEndian, int32(len(oneD)))
			if err != nil {
				return err
			}

			// Write actual float values
			for _, val := range oneD {
				// Normalize 24bit range to -1/+1
				floatVal := float32(val) / float32(normalizeVal)
				err = binary.Write(writer, binary.LittleEndian, floatVal)
				if err != nil {
					return err
				}
			}
		}
	}

	// Ensure all buffered data is written to file
	return writer.Flush()
}

// Unused currently
func read3DIntSliceBinary(filename string) ([][][]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var outerLen int32
	err = binary.Read(file, binary.LittleEndian, &outerLen)
	if err != nil {
		return nil, err
	}

	result := make([][][]int, outerLen)

	for i := int32(0); i < outerLen; i++ {
		var innerLen int32
		err = binary.Read(file, binary.LittleEndian, &innerLen)
		if err != nil {
			return nil, err
		}

		innerSlice := make([][]int, innerLen)

		for j := int32(0); j < innerLen; j++ {
			var leafLen int32
			err = binary.Read(file, binary.LittleEndian, &leafLen)
			if err != nil {
				return nil, err
			}

			leaf := make([]int, leafLen)
			for k := int32(0); k < leafLen; k++ {
				var val int32
				err = binary.Read(file, binary.LittleEndian, &val)
				if err != nil {
					return nil, err
				}
				leaf[k] = int(val)
			}
			innerSlice[j] = leaf
		}
		result[i] = innerSlice
	}

	return result, nil
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("Usage: go run main.go <outputFilePath> <file1.wav> <file2.wav>")
		return
	}

	out := os.Args[1]
	file1 := os.Args[2]
	file2 := os.Args[3]

	buf1, dec1, err := parseWavFile(file1)
	if err != nil {
		log.Fatalf("Error with file 1: %v", err)
	}

	buf2, dec2, err := parseWavFile(file2)
	if err != nil {
		log.Fatalf("Error with file 2: %v", err)
	}

	printWavInfo(buf1, dec1, file1)
	printWavInfo(buf2, dec2, file2)

	generateIOPairs(out, buf1, buf2)
}
