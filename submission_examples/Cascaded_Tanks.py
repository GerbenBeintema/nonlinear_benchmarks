
import nonlinear_benchmarks

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
train, test = nonlinear_benchmarks.Cascaded_Tanks()
n = test.state_initialization_window_length

# Train model:
#     - Feel free to change anything in this
from simple_model import train_model, apply_model
model = train_model(train)

# Apply model on test data using only u and y[:n] returning y_models:
#     - Feel to change this but make sure that you only use y[:n] when using the model for the simulation
y_test_model = apply_model(model, test.u, test.y[:n])


# Compute RMSE and print results: 
#   - Keep this functionally unchanged 
from nonlinear_benchmarks.error_metrics import RMSE
test_RMSE = RMSE(test.y[n:], y_test_model[n:])

print(f'{test_RMSE = :.3f}')
print(f'RMSE to submit = {test_RMSE:.3f}')
