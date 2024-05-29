
import nonlinear_benchmarks

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
train, test = nonlinear_benchmarks.WienerHammerBenchMark()
n = test.state_initialization_window_length

# Train model:
# Modify this part such that you can train your model starting from the training data.
from simple_model import train_model, apply_model
model = train_model(train)

# Apply model on test data:
# Only use u and y[:n] returning y_models (keep n fixed to the value provided by the benchmark dataset)
# Modify this part such that you can simulate the response of your model to the test input.
y_test_model = apply_model(model, test.u, test.y[:n])


# Compute RMSE in mV and print results: 
# Keep this functionally unchanged
from nonlinear_benchmarks.error_metrics import RMSE
test_RMSE_mV = 1000*RMSE(test.y[n:], y_test_model[n:])

print(f'{test_RMSE_mV = :.3f} mV')
print(f'RMSE to submit = {test_RMSE_mV:.3f}') # report this number during submission
