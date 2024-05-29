
import nonlinear_benchmarks

# Load data: 
#     - You are free to change this, e.g. split train into train and val sets
train_sets, (test_1, test_2) = nonlinear_benchmarks.CED()
n = test_1.state_initialization_window_length

# Train model:
#     - Feel free to change anything in this
from simple_model import train_model, apply_model
model = train_model(train_sets)

# Apply model on test data using only u and y[:n] returning y_models:
#     - Feel to change this but make sure that you only use y[:n] when using the model for the simulation
y_test_1_model = apply_model(model, test_1.u, test_1.y[:n])
y_test_2_model = apply_model(model, test_2.u, test_2.y[:n])

# Compute RMSE in [ticks/s] and print results: 
#   - Keep this functionally unchanged 
from nonlinear_benchmarks.error_metrics import RMSE
test_1_RMSE = RMSE(test_1.y[n:], y_test_1_model[n:])
test_2_RMSE = RMSE(test_2.y[n:], y_test_2_model[n:])

print(f'{test_1_RMSE = :.3f} [ticks/s]')
print(f'{test_2_RMSE = :.3f} [ticks/s]')
print(f'RMSE to submit = [{test_1_RMSE:.3f}; {test_2_RMSE:.3f}]')
