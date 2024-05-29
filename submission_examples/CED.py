
import nonlinear_benchmarks

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
train_sets, (test_1, test_2) = nonlinear_benchmarks.CED()
n = test_1.state_initialization_window_length

# Train model:
# Modify this part such that you can train your model starting from the training data.
from simple_model import train_model, apply_model
model = train_model(train_sets)

# Apply model on test data:
# Only use u and y[:n] returning y_models (keep n fixed to the value provided by the benchmark dataset)
# Modify this part such that you can simulate the response of your model to the test input.
y_test_1_model = apply_model(model, test_1.u, test_1.y[:n])
y_test_2_model = apply_model(model, test_2.u, test_2.y[:n])

# Compute RMSE in [ticks/s] and print results: 
# Keep this functionally unchanged
from nonlinear_benchmarks.error_metrics import RMSE
test_1_RMSE = RMSE(test_1.y[n:], y_test_1_model[n:])
test_2_RMSE = RMSE(test_2.y[n:], y_test_2_model[n:])

print(f'{test_1_RMSE = :.3f} [ticks/s]')
print(f'{test_2_RMSE = :.3f} [ticks/s]')
print(f'RMSE to submit = [{test_1_RMSE:.3f}; {test_2_RMSE:.3f}]') # report this number during submission
