
import nonlinear_benchmarks

# Load data: 
# Keep this part fixed, though you can split the train set further in a train and validation set. 
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
train, test = nonlinear_benchmarks.Silverbox()
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
n = test_multisine.state_initialization_window_length

# Train model:
# Modify this part such that you can train your model starting from the training data.
from simple_model import train_model, apply_model
model = train_model(train)

# Apply model on test data:
# Only use u and y[:n] returning y_models (keep n fixed to the value provided by the benchmark dataset)
# Modify this part such that you can simulate the response of your model to the test input.
y_test_multisine_model = apply_model(model, test_multisine.u, test_multisine.y[:n])
y_test_arrow_full_model = apply_model(model, test_arrow_full.u, test_arrow_full.y[:n])
y_test_arrow_no_extrapolation_model = apply_model(model, test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y[:n])


# Compute RMSE in mV and print results: 
# Keep this functionally unchanged
from nonlinear_benchmarks.error_metrics import RMSE
test_multisine_RMSE = 1000*RMSE(test_multisine.y[n:], y_test_multisine_model[n:])
test_arrow_full_RMSE = 1000*RMSE(test_arrow_full.y[n:], y_test_arrow_full_model[n:])
test_arrow_no_extrapolation_RMSE = 1000*RMSE(test_arrow_no_extrapolation.y[n:], y_test_arrow_no_extrapolation_model[n:])

print(f'{test_multisine_RMSE = :.3f} mV')
print(f'{test_arrow_full_RMSE = :.3f} mV')
print(f'{test_arrow_no_extrapolation_RMSE = :.3f} mV')

print(f'RMSE to submit = [{test_multisine_RMSE:.3f}; {test_arrow_full_RMSE:.3f}; {test_arrow_no_extrapolation_RMSE:.3f}]') # report this number during submission
