import numpy as np
import matplotlib.pyplot as plt
import time
import os

start_time_total = time.perf_counter()

# Clear old weight files
step_start = time.perf_counter()
for file in ['W_f.npy', 'W_i.npy', 'W_c.npy', 'W_o.npy', 'V.npy']:
    if os.path.exists(file):
        os.remove(file)
print(f"Weight file clearing time: {time.perf_counter() - step_start:.6f} seconds")

# Generate noisy cosine data
step_start = time.perf_counter()
seq_length = 100
x = np.linspace(0, 2 * np.pi, seq_length)
data = np.cos(x) + 0.01 * np.random.randn(seq_length)
print(f"Data generation time: {time.perf_counter() - step_start:.6f} seconds")

# Create delay vector
step_start = time.perf_counter()
delay = 2
inputs = np.array([data[i:i + delay] for i in range(len(data) - delay)])
targets = np.array([data[i + delay] for i in range(len(data) - delay)])
print(f"Delay vector creation time: {time.perf_counter() - step_start:.6f} seconds")

# Initialize weights
step_start = time.perf_counter()
hidden_size, input_size, output_size = 20, delay, 1
W_f, W_i, W_c, W_o = [np.random.randn(hidden_size, input_size + hidden_size) * 0.05 for _ in range(4)]
V = np.random.randn(output_size, hidden_size) * 0.05
b_f, b_i, b_c, b_o = [np.zeros((hidden_size, 1)) for _ in range(4)]
b_y = np.zeros((output_size, 1))
print(f"Weight initialization time: {time.perf_counter() - step_start:.6f} seconds")

# Training loop
step_start = time.perf_counter()
lr, epochs = 0.001, 200
sigmoid = lambda x: 1 / (1 + np.exp(-x))

for epoch in range(epochs):
    total_loss = 0
    for t in range(len(inputs)):
        x_t, y_t = inputs[t].reshape(-1, 1), targets[t].reshape(-1, 1)
        h_prev, C_prev = np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1))
        hx = np.vstack((h_prev, x_t))
        f_t = sigmoid(np.dot(W_f, hx) + b_f)
        i_t = sigmoid(np.dot(W_i, hx) + b_i)
        c_tilde = np.tanh(np.dot(W_c, hx) + b_c)
        C_t = f_t * C_prev + i_t * c_tilde
        o_t = sigmoid(np.dot(W_o, hx) + b_o)
        h_t = o_t * np.tanh(C_t)
        y_pred = np.dot(V, h_t) + b_y

        loss = np.mean((y_t - y_pred) ** 2)
        total_loss += loss
        grad = -2 * (y_t - y_pred)
        V -= lr * np.dot(grad, h_t.T)
        b_y -= lr * np.mean(grad)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")

print(f"Training time: {time.perf_counter() - step_start:.6f} seconds")

# Save weights
step_start = time.perf_counter()
for name, matrix in zip(['W_f', 'W_i', 'W_c', 'W_o', 'V'], [W_f, W_i, W_c, W_o, V]):
    np.save(name, matrix)
print(f"Weight saving time: {time.perf_counter() - step_start:.6f} seconds")

# Inference
step_start = time.perf_counter()
W_f, W_i, W_c, W_o, V = [np.load(f'{n}.npy') for n in ['W_f', 'W_i', 'W_c', 'W_o', 'V']]
b_f, b_i, b_c, b_o = [np.zeros((hidden_size, 1)) for _ in range(4)]
b_y = np.zeros((output_size, 1))
h, C = np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1))
y_pred_list = []

inference_start = time.perf_counter()
for t in range(len(inputs)):
    x_t = inputs[t].reshape(-1, 1)
    hx = np.vstack((h, x_t))
    f_t = sigmoid(np.dot(W_f, hx) + b_f)
    i_t = sigmoid(np.dot(W_i, hx) + b_i)
    c_tilde = np.tanh(np.dot(W_c, hx) + b_c)
    C = f_t * C + i_t * c_tilde
    o_t = sigmoid(np.dot(W_o, hx) + b_o)
    h = o_t * np.tanh(C)
    y = np.dot(V, h) + b_y
    y_pred_list.append(y.flatten())

inference_time = time.perf_counter() - inference_start
print(f"Inference time: {inference_time:.6f} seconds")
print(f"Time per timestep: {inference_time / len(inputs):.6f} seconds")


# Plot results
step_start = time.perf_counter()
plt.figure(figsize=(10, 4))
plt.plot(range(delay, seq_length), targets, label="True Cosine Data")
plt.plot(range(delay, seq_length), np.array(y_pred_list).flatten(), label="LSTM Predictions")
plt.legend()
plt.title("LSTM Prediction vs True Cosine Wave")
plt.show()
print(f"Plotting time: {time.perf_counter() - step_start:.6f} seconds")

# Total execution time
print(f"Total Execution Time: {time.perf_counter() - start_time_total:.6f} seconds")
