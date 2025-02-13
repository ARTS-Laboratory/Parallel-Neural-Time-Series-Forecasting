import numpy as np
import matplotlib.pyplot as plt
import time
#The goal is to implement an RNN-LSTM cell from strach using Linear algebra principles.
#The main features of an LSTM CELL:Input gate,Output gate , forget gate and cell state.

#Measure total runtime start
total_start_time = time.time()

#Generate Noisy Cosine Data
def generate_noisy_cosine(seq_length, noise_factor=0.1):
    x = np.linspace(0, 2 * np.pi, seq_length)
    cosine_wave = np.cos(x)
    noise = noise_factor * np.random.randn(seq_length)
    return cosine_wave + noise

seq_length = 100
data = generate_noisy_cosine(seq_length)

def create_delay_vector(data, delay=2):
    X, y = [], []
    for i in range(len(data) - delay):
        X.append(data[i:i + delay])
        y.append(data[i + delay])
    return np.array(X), np.array(y)

# Set delay (embedding dimension)
delay = 2
inputs, targets = create_delay_vector(data, delay=delay)

#Define LSTM Class with Backpropagation
class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # LSTM Weights Initialization
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.1

        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

        self.V = np.random.randn(output_size, hidden_size) * 0.1
        self.b_y = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        Forward pass for LSTM
        """
        self.h, self.C = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
        self.y_pred, self.cache = [], []

        for t in range(len(inputs)):
            x_t = inputs[t].reshape(-1, 1)
            hx = np.vstack((self.h, x_t))

            f_t = self.sigmoid(np.dot(self.W_f, hx) + self.b_f)
            i_t = self.sigmoid(np.dot(self.W_i, hx) + self.b_i)
            c_tilde = np.tanh(np.dot(self.W_c, hx) + self.b_c)
            self.C = f_t * self.C + i_t * c_tilde
            o_t = self.sigmoid(np.dot(self.W_o, hx) + self.b_o)
            self.h = o_t * np.tanh(self.C)

            y = np.dot(self.V, self.h) + self.b_y
            self.y_pred.append(y.flatten())

            self.cache.append((hx, f_t, i_t, c_tilde, o_t, self.h, self.C))

        return np.array(self.y_pred).flatten()

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, inputs, targets):
        """
        Backpropagation Through Time (BPTT)
        """
        dW_f, dW_i, dW_c, dW_o = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c), np.zeros_like(self.W_o)
        db_f, db_i, db_c, db_o = np.zeros_like(self.b_f), np.zeros_like(self.b_i), np.zeros_like(self.b_c), np.zeros_like(self.b_o)
        dV, db_y = np.zeros_like(self.V), np.zeros_like(self.b_y)

        dh_next, dC_next = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(inputs))):
            hx, f_t, i_t, c_tilde, o_t, h_t, C_t = self.cache[t]
            y_pred_t = self.y_pred[t].reshape(-1, 1)
            y_true_t = targets[t].reshape(-1, 1)

            dy = y_pred_t - y_true_t
            dV += np.dot(dy, h_t.T)
            db_y += dy

            dh = np.dot(self.V.T, dy) + dh_next
            do = dh * np.tanh(C_t) * o_t * (1 - o_t)
            dC = dh * o_t * (1 - np.tanh(C_t) ** 2) + dC_next
            df = dC * self.cache[t-1][6] * f_t * (1 - f_t)
            di = dC * c_tilde * i_t * (1 - i_t)
            dc_tilde = dC * i_t * (1 - c_tilde ** 2)

            dW_f += np.dot(df, hx.T)
            dW_i += np.dot(di, hx.T)
            dW_c += np.dot(dc_tilde, hx.T)
            dW_o += np.dot(do, hx.T)
            db_f += df
            db_i += di
            db_c += dc_tilde
            db_o += do

            dh_next = np.dot(self.W_f[:, :self.hidden_size].T, df) + \
                      np.dot(self.W_i[:, :self.hidden_size].T, di) + \
                      np.dot(self.W_c[:, :self.hidden_size].T, dc_tilde) + \
                      np.dot(self.W_o[:, :self.hidden_size].T, do)

            dC_next = f_t * dC

        # Gradient Descent Updates
        self.W_f -= self.learning_rate * dW_f
        self.W_i -= self.learning_rate * dW_i
        self.W_c -= self.learning_rate * dW_c
        self.W_o -= self.learning_rate * dW_o
        self.b_f -= self.learning_rate * db_f
        self.b_i -= self.learning_rate * db_i
        self.b_c -= self.learning_rate * db_c
        self.b_o -= self.learning_rate * db_o
        self.V -= self.learning_rate * dV
        self.b_y -= self.learning_rate * db_y

#Train LSTM Model with Backpropagation
lstm = SimpleLSTM(input_size=delay, hidden_size=10, output_size=1, learning_rate=0.01)
epochs = 200

train_start_time = time.time()
for epoch in range(epochs):
    predictions = lstm.forward(inputs)
    loss = lstm.compute_loss(targets, predictions)
    lstm.backward(inputs, targets)  # Perform backpropagation
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
train_end_time = time.time()
train_time = train_end_time - train_start_time

#Compute Computational Time per Timestep
time_per_timestep = train_time / (epochs * len(inputs))
print(f"\nComputational Time per Timestep: {time_per_timestep:.6f} seconds ({time_per_timestep * 1e6:.2f} µs)")

#Total Runtime
total_end_time = time.time()
print(f"Total Runtime: {total_end_time - total_start_time:.4f} seconds")

#Plot Results
plt.figure(figsize=(10,4))
plt.plot(targets, label="True Data")
plt.plot(predictions, label="LSTM Predictions")
plt.legend()
plt.show()
