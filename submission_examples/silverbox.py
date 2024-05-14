
import nonlinear_benchmarks

#load data:
train, test = nonlinear_benchmarks.Silverbox()
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
n = test_multisine.state_initialization_window_length

#train model:
from simple_model import train_model, apply_model
model = train_model(train)

#apply model on test data using only u and y[:n] returning y_models
y_test_multisine_model = apply_model(model, test_multisine.u, test_multisine.y[:n])
y_test_arrow_full_model = apply_model(model, test_arrow_full.u, test_arrow_full.y[:n])
y_test_arrow_no_extrapolation_model = apply_model(model, test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y[:n])


#compute RMSE in mV
from nonlinear_benchmarks.error_metrics import RMSE
test_multisine_RMSE = 1000*RMSE(test_multisine.y[n:], y_test_multisine_model[n:])
test_arrow_full_RMSE = 1000*RMSE(test_arrow_full.y[n:], y_test_arrow_full_model[n:])
test_arrow_no_extrapolation_RMSE = 1000*RMSE(test_arrow_no_extrapolation.y[n:], y_test_arrow_no_extrapolation_model[n:])

print(f'{test_multisine_RMSE = :.3} mv')
print(f'{test_arrow_full_RMSE = :.3} mv')
print(f'{test_arrow_no_extrapolation_RMSE = :.3} mv')

print(f'RMSE to submit = [{test_multisine_RMSE:.3}; {test_arrow_full_RMSE:.3}; {test_arrow_no_extrapolation_RMSE:.3}]')
