// package mnist handles the mnist data set
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

const numLabels = 10
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	// Width of the input tensor / picture
	Width = 28
	// Height of the input tensor / picture
	Height = 28
)

func readLabelFile(r io.Reader, e error) (labels []Label, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

func readImageFile(r io.Reader, e error) (imgs []RawImage, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err /*os.ErrInvalid*/
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != int(m) {
			return nil, os.ErrInvalid
		}
	}
	return imgs, nil
}

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

// Label is a digit label in 0 to 9
type Label uint8

func loadOriginal(typ, loc string, as tensor.Dtype) (inputs, targets tensor.Tensor, err error) {
	const (
		trainLabel = "train-labels.idx1-ubyte"
		trainData  = "train-images.idx3-ubyte"
		testLabel  = "t10k-labels.idx1-ubyte"
		testData   = "t10k-images.idx3-ubyte"
	)

	var labelFile, dataFile string
	switch typ {
	case "train", "dev":
		labelFile = filepath.Join(loc, trainLabel)
		dataFile = filepath.Join(loc, trainData)
	case "test":
		labelFile = filepath.Join(loc, testLabel)
		dataFile = filepath.Join(loc, testData)
	}

	var labelData []Label
	var imageData []RawImage

	if labelData, err = readLabelFile(os.Open(labelFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read Labels")
	}

	if imageData, err = readImageFile(os.Open(dataFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read image data")
	}

	inputs = prepareX(imageData, as)
	targets = prepareY(labelData, as)
	return
}

func loadAll(loc string, as tensor.Dtype) (trainInputs, trainTargets, validInputs, validTargets, testInputs, testTargets tensor.Tensor, err error) {
	const (
		trainLabel = "train-labels-idx1-ubyte"
		trainData  = "train-images-idx3-ubyte"
		testLabel  = "t10k-labels-idx1-ubyte"
		testData   = "t10k-images-idx3-ubyte"
	)

	// Load training and validation data
	trainLabelFile := filepath.Join(loc, trainLabel)
	trainDataFile := filepath.Join(loc, trainData)

	trainLabelData, err := readLabelFile(os.Open(trainLabelFile))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "Unable to read Training Labels")
	}

	trainImageData, err := readImageFile(os.Open(trainDataFile))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "Unable to read Training image data")
	}

	allTrainInputs := prepareX(trainImageData, as)
	allTrainTargets := prepareY(trainLabelData, as)

	numExamples := allTrainInputs.Shape()[0]
	numValid := int(float64(numExamples) * *validSplit)
	numTrain := numExamples - numValid

	trainInputs, err = allTrainInputs.Slice(sli{0, numTrain})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	trainTargets, err = allTrainTargets.Slice(sli{0, numTrain})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	validInputs, err = allTrainInputs.Slice(sli{numTrain, numExamples})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	validTargets, err = allTrainTargets.Slice(sli{numTrain, numExamples})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	// Load test data
	testLabelFile := filepath.Join(loc, testLabel)
	testDataFile := filepath.Join(loc, testData)

	testLabelData, err := readLabelFile(os.Open(testLabelFile))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "Unable to read Test Labels")
	}

	testImageData, err := readImageFile(os.Open(testDataFile))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "Unable to read Test image data")
	}

	testInputs = prepareX(testImageData, as)
	testTargets = prepareY(testLabelData, as)

	return trainInputs, trainTargets, validInputs, validTargets, testInputs, testTargets, nil
}

func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func reversePixelWeight(px float64) byte {
	return byte((pixelRange*px - pixelRange) / 0.9)
}

func prepareX(M []RawImage, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(M)
	cols := len(M[0])

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, pixelWeight(M[i][j]))
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, float32(pixelWeight(M[i][j])))
			}
		}
		backing = b
	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func prepareY(N []Label, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(N)
	cols := 10

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b

	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func printLabelCounts(trainTargets, validTargets, testTargets tensor.Tensor) {
	// Assuming the targets are one-hot encoded
	labelNames := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

	// Convert tensor.Tensor to *tensor.Dense and then extract the indices of the maximum values (i.e., the labels)
	trainLabels, _ := trainTargets.(*tensor.Dense).Argmax(1)
	validLabels, _ := validTargets.(*tensor.Dense).Argmax(1)
	testLabels, _ := testTargets.(*tensor.Dense).Argmax(1)

	fmt.Println("Label\tTrain\tValid\tTest")
	fmt.Println("-----\t-----\t-----\t-----")

	for i, label := range labelNames {
		trainCount := countInstancesOfLabel(trainLabels, i)
		validCount := countInstancesOfLabel(validLabels, i)
		testCount := countInstancesOfLabel(testLabels, i)

		fmt.Printf("%s\t%d\t%d\t%d\n", label, trainCount, validCount, testCount)
	}
}

func countInstancesOfLabel(labels tensor.Tensor, label int) int {
	count := 0
	data := labels.Data().([]int)
	for _, l := range data {
		if l == label {
			count++
		}
	}
	return count
}
