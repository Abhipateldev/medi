import numpy as np

# Prepare the data: numbers from 1 to 100, scaled to 0.01 to 1.00
X = np.arange(1, 101).reshape(-1, 1) / 100.0
# Labels: 0 for even, 1 for odd
y = (np.arange(1, 101) % 2).reshape(-1, 1)

# Define sigmoid function and its derivative for activation and backpropagation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases randomly, using seed for reproducibility
np.random.seed(42)
W1 = np.random.uniform(-1, 1, (1, 4))  # Input to hidden layer (1x4)
b1 = np.random.uniform(-1, 1, (1, 4))  # Biases for hidden layer
W2 = np.random.uniform(-1, 1, (4, 1))  # Hidden to output layer (4x1)
b2 = np.random.uniform(-1, 1, (1, 1))  # Bias for output layer

# Set training parameters
learning_rate = 0.1
epochs = 10000

# Training loop: forward pass, compute loss, backpropagation, and update weights
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    y_pred = sigmoid(output_input)
    
    # Compute binary cross-entropy loss
    loss = -np.mean(y * np.log(y_pred + 1e-7) + (1 - y) * np.log(1 - y_pred + 1e-7))
    
    # Backpropagation
    d_loss_y_pred = - (y / (y_pred + 1e-7) - (1 - y) / (1 - y_pred + 1e-7))
    d_y_pred = d_loss_y_pred * sigmoid_derivative(y_pred)
    
    d_W2 = np.dot(hidden_output.T, d_y_pred)
    d_b2 = np.sum(d_y_pred, axis=0, keepdims=True)
    
    d_hidden_output = np.dot(d_y_pred, W2.T)
    d_hidden_input = d_hidden_output * sigmoid_derivative(hidden_output)
    
    d_W1 = np.dot(X.T, d_hidden_input)
    d_b1 = np.sum(d_hidden_input, axis=0, keepdims=True)
    
    # Update weights and biases using gradient descent
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    # Print loss every 1000 epochs for monitoring
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the network on numbers 1 to 10
test_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1) / 100.0
hidden_input_test = np.dot(test_numbers, W1) + b1
hidden_output_test = sigmoid(hidden_input_test)
output_input_test = np.dot(hidden_output_test, W2) + b2
y_pred_test = sigmoid(output_input_test)
predictions = (y_pred_test > 0.5).astype(int)

# Print results
for num, pred in zip(np.arange(1, 11), predictions):
    print(f"Number {num}: {'odd' if pred else 'even'}")