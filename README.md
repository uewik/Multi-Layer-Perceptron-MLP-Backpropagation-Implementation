# Multi-Layer Perceptron (MLP) Backpropagation Implementation

## Overview

This project implements the backpropagation algorithm for training Multi-Layer Perceptrons (MLPs) to approximate various mathematical functions. The implementation follows the backpropagation algorithm as described on pages 11-26 of Neural Network Design, featuring both feedforward and backward propagation steps with network parameter updates.

## Features

- **Custom MLP Implementation**: From-scratch implementation of multi-layer perceptron with configurable architecture
- **Backpropagation Algorithm**: Complete implementation of forward pass, backward pass, and parameter updates
- **Multiple Target Functions**: Support for approximating four different mathematical functions
- **Visualization**: Real-time plotting of network response vs target function and SSE convergence
- **Configurable Parameters**: Adjustable learning rate, network architecture, and convergence criteria

## Target Functions

The program can approximate the following target functions:

1. **f(p) = sin(p)** for p ∈ [-2π, 2π] with a two-layered MLP [7,1]
2. **f(p) = p²** for p ∈ [-2,2] with a two-layered MLP [3,1]  
3. **f(p) = eᵖ** for p ∈ [0,2] with a two-layered MLP [3,1]
4. **f(p) = sin²(p) + cos³(p)** for p ∈ [-2π, 2π] with a three-layered MLP [10,5,1]

## Implementation Details

### Network Architecture
- **Input Layer**: Single input neuron for p values
- **Hidden Layer(s)**: Configurable number of neurons with sigmoid activation
- **Output Layer**: Single neuron with linear activation

### Key Components

#### MLP Class
- `__init__()`: Initializes weights and biases with small random values
- `forward()`: Implements feedforward propagation through all layers
- `backward()`: Implements backpropagation algorithm for gradient computation
- `update_params()`: Updates weights and biases using computed gradients

#### Training Function
- `train_mlp()`: Complete training loop with convergence monitoring
- Supports early stopping based on Sum of Squared Errors (SSE) threshold
- Generates visualization plots for network performance

### Activation Functions
- **Sigmoid**: Used in hidden layers with derivative for backpropagation
- **Linear**: Used in output layer for regression tasks

## Usage

### Running the Code

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd miniProject2
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the program**:
   ```bash
   python Question2.py
   ```

### Switching Between Target Functions

The code includes all four target functions. To run different functions:

1. **Function 1 (sin(p))** - Currently active by default
2. **Functions 2, 3, 4** - Uncomment the respective code blocks and comment out others

Example for Function 2:
```python
# Uncomment this block:
# p_values = np.linspace(-2, 2, num_samples)
# targets = p_values ** 2
# inputs = np.array([p_values]).T
# mlp = MLP([3, 1], 1)
# train_mlp(mlp, inputs, targets, learning_rate=0.02, max_iterations=200000, sse_cutoff=0.01)
```

## Configuration Parameters

### Default Settings
- **Learning Rate**: 0.02
- **Maximum Iterations**: 200,000
- **SSE Cutoff**: 0.01
- **Number of Samples**: 100
- **Random Seed**: 5806 (for reproducible results)

### Network Architectures
- **Function 1**: [7, 1] - 7 hidden neurons, 1 output
- **Function 2**: [3, 1] - 3 hidden neurons, 1 output  
- **Function 3**: [3, 1] - 3 hidden neurons, 1 output
- **Function 4**: [10, 5, 1] - 10 and 5 hidden neurons, 1 output

## Output and Visualization

The program generates two types of plots:

### 1. Network Response Plot
- **Blue dots**: Target function values
- **Red line**: Network output after training
- Shows how well the network approximates the target function
- Includes network architecture information in the title

### 2. SSE Convergence Plot
- **Log-log scale**: Number of iterations vs Sum of Squared Errors
- Demonstrates training convergence
- Includes final SSE value and training parameters

## Results

The implementation successfully demonstrates:
- Effective function approximation for all four target functions
- Convergence monitoring with early stopping
- Visual comparison between network output and target functions
- Training progress tracking through SSE plots

## Technical Implementation Notes

### Backpropagation Algorithm
1. **Forward Pass**: Compute activations layer by layer
2. **Error Calculation**: Calculate output error using target values
3. **Backward Pass**: Propagate errors backward through network layers
4. **Parameter Update**: Update weights and biases using gradient descent

### Key Features
- **Vectorized Operations**: Efficient numpy-based computations
- **Proper Gradient Computation**: Correct implementation of chain rule
- **Flexible Architecture**: Easy modification of network structure
- **Convergence Monitoring**: Automatic stopping when SSE threshold is met

## Code Structure

```
Question2.py
├── Activation Functions
│   ├── sigmoid() & sigmoid_derivative()
│   └── linear() & linear_derivative()
├── MLP Class
│   ├── __init__() - Network initialization
│   ├── forward() - Forward propagation
│   ├── backward() - Backpropagation
│   └── update_params() - Parameter updates
├── Training Function
│   └── train_mlp() - Complete training loop
└── Target Function Examples
    ├── Function 1: sin(p)
    ├── Function 2: p²
    ├── Function 3: eᵖ
    └── Function 4: sin²(p) + cos³(p)
```

## Dependencies

- `numpy`: For numerical computations and array operations
- `matplotlib`: For plotting and visualization

## Performance Notes

- The implementation uses early stopping to prevent overfitting
- SSE threshold of 0.01 provides good balance between accuracy and training time
- Different network architectures are optimized for each target function's complexity

## Author

This implementation was created as part of a neural networks course mini-project, demonstrating practical understanding of the backpropagation algorithm and MLP training procedures.

## License

This project is for educational purposes as part of coursework.
