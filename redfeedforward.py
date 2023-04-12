import torch
import torch.nn as nn

# Definir la arquitectura de la red
class FeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Definir parámetros de entrada
input_size = 10
hidden_size = 20
output_size = 5

# Crear instancia de la red
model = FeedforwardNet(input_size, hidden_size, output_size)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Ejemplo de entrenamiento
# Entradas y etiquetas de ejemplo
x = torch.randn(1, input_size)
y = torch.randn(1, output_size)

# Propagar hacia adelante
output = model(x)

# Calcular la pérdida
loss = criterion(output, y)

# Propagar hacia atrás y actualizar los pesos
optimizer.zero_grad()
loss.backward()
optimizer.step()