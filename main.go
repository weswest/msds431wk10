package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
)

// This is related to GoMNIST
// Print the image to the console
func printImage(image GoMNIST.RawImage) {
	scaleFactor := 255.0 / 8.0
	nRow := 28
	nCol := 28

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			// Get the pixel value at the current position
			pixel := image[i*nCol+j]

			// Scale the pixel value
			scaledPixel := int(math.Round(float64(pixel) / scaleFactor))

			// Make sure that only 0 scales to 0
			if pixel != 0 && scaledPixel == 0 {
				scaledPixel = 1
			}

			// Print a space if the pixel value is 0, otherwise print the scaled pixel value
			if scaledPixel == 0 {
				fmt.Print(" ")
			} else {
				fmt.Print(scaledPixel)
			}
		}
		// Start a new line after each row
		fmt.Println()
	}
}

// This takes all of the images and converts them to float64s
func convertMNISTForModeling(images []GoMNIST.RawImage) [][]float64 {
	var floatImages [][]float64

	for _, image := range images {
		var floatImage []float64
		for _, pixel := range image {
			floatImage = append(floatImage, float64(pixel))
		}
		floatImages = append(floatImages, floatImage)
	}

	return floatImages
}
func createCNN() *cnns.WholeNet {
	// Create a new neural network
	net := cnns.WholeNet{
		LP: cnns.NewLearningParametersDefault(),
	}

	// First convolutional layer: 3x3 with 32 filters
	conv1 := cnns.NewConvLayer(&tensor.TDsize{X: 28, Y: 28, Z: 1}, 32, 3, 1)
	net.Layers = append(net.Layers, conv1)

	// ReLU activation after convolution
	relu1 := cnns.NewReLULayer(conv1.GetOutputSize())
	net.Layers = append(net.Layers, relu1)

	// Max pooling layer: 2x2
	maxpool1 := cnns.NewPoolingLayer(relu1.GetOutputSize(), 2, 2, "max", "valid")
	net.Layers = append(net.Layers, maxpool1)

	// Second convolutional layer: 3x3 with 64 filters
	conv2 := cnns.NewConvLayer(maxpool1.GetOutputSize(), 64, 3, 1)
	net.Layers = append(net.Layers, conv2)

	// ReLU activation after convolution
	relu2 := cnns.NewReLULayer(conv2.GetOutputSize())
	net.Layers = append(net.Layers, relu2)

	// Max pooling layer: 2x2
	maxpool2 := cnns.NewPoolingLayer(relu2.GetOutputSize(), 2, 2, "max", "valid")
	net.Layers = append(net.Layers, maxpool2)

	// Fully connected (dense) layer
	fc1 := cnns.NewFullyConnectedLayer(maxpool2.GetOutputSize(), 128)
	fc1.SetActivationFunc(cnns.ActivationSygmoid)
	fc1.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)
	net.Layers = append(net.Layers, fc1)

	// Output layer: Dense with 10 classes (digits)
	fc2 := cnns.NewFullyConnectedLayer(fc1.GetOutputSize(), 10)
	fc2.SetActivationFunc(cnns.ActivationSygmoid)
	fc2.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)
	net.Layers = append(net.Layers, fc2)

	return &net
}

func convertToTensor(images []GoMNIST.RawImage) []*tensor.Tensor {
	tensors := make([]*tensor.Tensor, len(images))
	for i, img := range images {
		data := make([]float64, len(img))
		for j, pixel := range img {
			data[j] = float64(pixel) / 255.0 // Normalize to [0,1]
		}
		tensors[i] = tensor.NewTensor(data, 28, 28, 1)
	}
	return tensors
}

func tensorToMatrix(t *tensor.Tensor) *mat.Dense {
	data := t.GetData()
	return mat.NewDense(t.Dims[0], t.Dims[1]*t.Dims[2], data)
}

func trainModel(cnn *cnns.WholeNet, trainImages []*tensor.Tensor, trainLabels []*mat.Dense, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i, img := range trainImages {
			// Convert tensor to matrix
			imgMatrix := tensorToMatrix(img)

			// Feedforward
			err := cnn.FeedForward(imgMatrix)
			if err != nil {
				log.Printf("Feedforward caused error: %s", err.Error())
				return
			}

			// Get the desired output for the current image
			desiredOutput := trainLabels[i]

			// Backpropagate
			err = cnn.Backpropagate(desiredOutput)
			if err != nil {
				log.Printf("Backpropagate caused error: %s", err.Error())
				return
			}

			// Accumulate loss (for demonstration purposes, using MSE here)
			prediction := cnn.Layers[len(cnn.Layers)-1].(*cnns.FullyConnectedLayer).GetOutput() // Assuming the last layer is a FullyConnectedLayer
			loss := 0.0
			for j := 0; j < 10; j++ {
				diff := prediction.At(0, j) - desiredOutput.At(0, j)
				loss += diff * diff
			}
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainImages))
		fmt.Printf("Epoch %d: Average Loss: %f\n", epoch+1, avgLoss)
	}
}

// ... [Rest of your code]

func main() {
	rng := rand.New(rand.NewSource(431)) //Obvi.
	fmt.Println("Random number: ", rng.Intn(100))

	//////////////////////////
	// GoMNIST time			//
	//////////////////////////
	// Load the dataset
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("First Train label: ", train.Labels[0])
	printImage(train.Images[0])

	// This code returns the train and test MNIST.Set types
	// Set has NRow, NCol, Images ([]RawImage), Labels ([]Label)

	fmt.Println("MNIST Rows: ", train.NRow, test.NRow)
	fmt.Println("MNIST Columns: ", train.NCol, test.NCol)
	inputData := convertMNISTForModeling(train.Images)
	fmt.Println(inputData)

	// Convert MNIST data to suitable format
	trainTensors := convertToTensor(train.Images)
	trainMatrix := convertToMatrix(train.Labels)

	// Create the CNN model
	cnn := createCNN()

	// Train the model
	trainModel(cnn, trainTensors, trainMatrix, 10) // Training for 10 epochs as an example
}
