import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set seed
torch.manual_seed(42)


class HebbianRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, alpha=0.2, eta=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha  # Plasticity coefficient
        self.eta = eta      # Learning rate for Hebbian trace
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Fixed weights
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1, requires_grad=False)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1, requires_grad=False)
        self.W_out = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1, requires_grad=True)

        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))

        # Hebbian trace: initialized later based on batch
        self.hebb = None

    def reset_hebb(self, batch_size):
        self.hebb = torch.zeros(batch_size, self.hidden_size, self.hidden_size)
    def plastic(self, h):
        return torch.bmm(self.hebb, h.unsqueeze(2)).squeeze(2)

    def forward(self, x_seq, mod_signal_seq=None):

      batch_size, seq_len, _ = x_seq.size()
      device = x_seq.device

      h = torch.zeros(batch_size, self.hidden_size).to(device)
      self.hebb = torch.zeros(batch_size, self.hidden_size, self.hidden_size).to(device)

      outputs = []

      for t in range(seq_len):
          x_t = x_seq[:, t, :]  # [B, input_size]
          mod_t = mod_signal_seq[:, t].view(-1, 1, 1) if mod_signal_seq is not None else 1.0

          h_unsq = h.unsqueeze(2)  # [B, H, 1]

        # Compute new hidden state with Hebbian plasticity
          h_new = torch.tanh(self.i2h(x_t) + self.h2h(h) + self.plastic(h))

          h_new_unsq = h_new.unsqueeze(1)  # [B, 1, H]

        # Hebbian rule with neuromodulation
          outer = torch.bmm(h_unsq, h_new_unsq)  # [B, H, H]
          self.hebb = (1 - self.eta) * self.hebb + self.eta * mod_t * outer

          h = h_new
          outputs.append(h)

      outputs = torch.stack(outputs, dim=1)  # [B, T, H]
      return self.out(outputs)              # Final output projection
    
# Generate toy sequence data
def generate_batch(batch_size, seq_len):
    x = torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1, :] = 0  # Last target is dummy
    return x, y

input_size = 1
hidden_size = 32
output_size = 1
seq_len = 10
batch_size = 32
n_epochs = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HebbianRNN(input_size, hidden_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

for epoch in range(n_epochs):
    model.train()
    x_batch, y_batch = generate_batch(batch_size, seq_len)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    # Create synthetic neuromodulatory signal: reward signal for correct prediction
    with torch.no_grad():
        mod_signal = torch.rand(batch_size, seq_len).to(device)  # random modulation (placeholder)

    y_pred = model(x_batch, mod_signal_seq=mod_signal)
    loss = F.mse_loss(y_pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f}")

plt.plot(losses)
plt.title("Training Loss over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.show()

model.eval()
x_test, y_test = generate_batch(1, seq_len)
x_test = x_test.to(device)

# No modulation during test
mod_signal_test = torch.ones(1, seq_len).to(device)

with torch.no_grad():
    y_pred = model(x_test, mod_signal_seq=mod_signal_test)

print("Input Sequence: ", x_test.squeeze().cpu().numpy())
print("Target Output : ", y_test.squeeze().cpu().numpy())
print("Predicted     : ", y_pred.squeeze().cpu().numpy().round(2))