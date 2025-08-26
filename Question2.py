import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5806)


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def sigmoid_derivative(n):
    return sigmoid(n) * (1 - sigmoid(n))


def linear(n):
    return n


def linear_derivative(n):
    return np.ones_like(n)


class MLP:
    def __init__(self, layer_sizes, input_dim):
        self.layer_sizes = [input_dim] + layer_sizes
        # initialize initial weights and biases with small random numbers in each layer
        # self.weights = [np.random.randn(y, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        # self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]

        self.weights = []
        self.biases = []
        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(np.random.randn(y, x))
            self.biases.append(np.random.randn(y, 1))

    def forward(self, x):
        # x: a0 = p
        a_list = [x]
        n_list = []
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            n = w @ a_list[-1] + b
            n_list.append(n)
            if i < len(self.weights) - 1:
                a_list.append(sigmoid(n))
            else:
                a_list.append(linear(n))
        return a_list, n_list

    def backward(self, a_list, n_list, target):
        list_sensitivity = []
        sensitivity_last_layer = (-2) * linear_derivative(n_list[-1]) * (target - a_list[-1])
        list_sensitivity.append(sensitivity_last_layer)

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            # initialize F^m(n^m)
            F = []
            for j in range(len(n_list[i - 1])):
                temp = []
                for k in range(len(n_list[i - 1])):
                    temp.append(0)
                F.append(temp)

            for l in range(len(n_list[i - 1])):
                F[l][l] = sigmoid_derivative(n_list[i - 1][l]).item()
            F_ndarray = np.array(F)

            sensitivity = F_ndarray @ self.weights[i].T @ list_sensitivity[0]

            list_sensitivity.insert(0, sensitivity)

        return list_sensitivity

    def update_params(self, list_sensitivity, a_list, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * list_sensitivity[i] @ a_list[i].T
            self.biases[i] -= learning_rate * list_sensitivity[i]


def train_mlp(mlp, inputs, targets, learning_rate, max_iterations, sse_cutoff):
    sse_history = []
    final_responses = []
    num_of_iterations = 0
    for i in range(max_iterations):
        e_list = []
        responses = []
        for x, y in zip(inputs, targets):
            x = x.reshape(-1, 1)  # Ensure that x is a column vector
            y = y.reshape(-1, 1)  # Ensure that y is a column vector

            # Forward Propagation
            a_list, n_list = mlp.forward(x)

            e = y - a_list[-1]
            e_list.append(e)

            responses.append(a_list[-1])

            # Backward Propagation
            list_sensitivity = mlp.backward(a_list, n_list, y)

            # Weight and bias update
            mlp.update_params(list_sensitivity, a_list, learning_rate)

        # Calculate and store the sum of squared errors
        sse = sum([e ** 2 for e in e_list])
        sse_history.append(sse)

        final_responses = responses

        num_of_iterations += 1

        # Check for convergence
        if sse < sse_cutoff:
            print(f"Training stopped after {i + 1} iterations with SSE: {sse}")
            break

    # Plot the network response against the target function
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)  # Subplot 1
    plt.scatter(inputs, targets, color='blue', label='Target')

    final_responses_ndarray = np.array(final_responses).squeeze()
    plt.plot(inputs, final_responses_ndarray, color='red', linestyle='-', label='Network after training')
    plt.xlabel('p')
    plt.ylabel('Magnitude')
    plt.title(f'Number of iterations = {num_of_iterations} \n' +
              f'Number of layers = {len(mlp.layer_sizes) - 1} \n' +
              f'Number of neurons = {mlp.layer_sizes[1:]}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot the SSE vs number of iterations in log scale
    # plt.subplot(1, 2, 2)  # Subplot 2
    sse_history_formatted = np.array(sse_history).squeeze()
    plt.semilogx(range(len(sse_history)), sse_history_formatted, '-b')
    plt.semilogy(range(len(sse_history)), sse_history_formatted, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Magnitude')
    plt.title(f'sum of square of errors {sse_history[-1][0][0]:.2f} \n' +
              f'Number of iterations = {num_of_iterations} \n' +
              f'Learning ratio = {learning_rate} \n' +
              f'Number of neurons = {mlp.layer_sizes[1:]} \n' +
              f'SSE error cut off = {sse_cutoff}')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_samples = 100

# target function 1:
p_values = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
targets = np.sin(p_values)
inputs = np.array([p_values]).T  # Generalize for actual input dimension

# Define the network
mlp = MLP([7, 1], 1)

# Train the network
train_mlp(mlp, inputs, targets, learning_rate=0.02, max_iterations=200000, sse_cutoff=0.01)

# Please comment out each of the following code blocks to run the different target function

# target function 2:
# p_values = np.linspace(-2, 2, num_samples)
# targets = p_values ** 2
# inputs = np.array([p_values]).T
#
# # Define the network
# mlp = MLP([3, 1], 1)
#
# # Train the network
# train_mlp(mlp, inputs, targets, learning_rate=0.02, max_iterations=200000, sse_cutoff=0.01)


# target function 3:
# p_values = np.linspace(0, 2, num_samples)
# targets = np.exp(p_values)
# inputs = np.array([p_values]).T
#
# # Define the network
# mlp = MLP([3, 1], 1)
#
# # Train the network
# train_mlp(mlp, inputs, targets, learning_rate=0.02, max_iterations=200000, sse_cutoff=0.01)


# target function 4:
# p_values = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
# targets = (np.sin(p_values)) ** 2 + (np.cos(p_values)) ** 3
# inputs = np.array([p_values]).T
#
# # Define the network
# mlp = MLP([10, 5, 1], 1)
#
# # Train the network
# train_mlp(mlp, inputs, targets, learning_rate=0.02, max_iterations=200000, sse_cutoff=0.01)
