
# nonlinear_benchmarks
 
The official dataloader of [nonlinearbenchmark.org](http://www.nonlinearbenchmark.org/). This toolbox can automatically download, load and split many of the datasets which can be found on the website. 

For instance, https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein.

```python
import nonlinear_benchmarks
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
print(train_val) 
# prints : Input_output_data "train WH" u.shape=(100000,) y.shape=(100000,) sampling_time=1.953e-05
print(test)
# prints: Input_output_data "test WH" u.shape=(78800,) y.shape=(78800,) sampling_time=1.953e-05 n_initialization_samples=50
sampling_time = train_val.sampling_time #in seconds
u_train, y_train = train_val  #or train_val.u, train_val.y
u_test, y_test = test         #or test.u,      test.y
print(test.n_initialization_samples) #the number of samples that can be used at the start of the test set to initialize the model state.
```

In the `WienerHammerBenchMark` and all other benchmark function multiple options are provided. 
 * `data_file_locations=True`: will return the raw data files locations
 * `train_test_split=False`: Gets the entire dataset without a split
 * `force_download=True`: Will force the (re-)downloading of benchmark files instead of 
 * `url=` : Will let you overwrite the download link manually (please contact maintainers if the default link is broken)
 * `atleast_2d=True`: e.g. convert `u.shape = (250,)` to `u.shape = (250,1)` (same for y)

# Install

```
pip install nonlinear-benchmarks
```

# Datasets

Multiple datasets have been implemented with an official train test split which are given below. 

(p.s. datasets without an official train test split can be found in `nonlinear_benchmarks.not_splitted_benchmarks`)

## EMPS

![image](figures/EMPS.jpg)

```python
train_val, test = nonlinear_benchmarks.EMPS()
print(test.n_initialization_samples) # = 20
train_val_u, train_val_y = train_val
test_u, test_y = test
```

## CED

![image](figures/CED.jpg)

```python
train_val, test = nonlinear_benchmarks.CED()
print(test[0].n_initialization_samples) # = 4
(train_val_u_1, train_val_y_1), (train_val_u_2, train_val_y_2) = train_val
(test_u_1, test_y_1), (test_u_2, test_y_2) = test
```

This dataset consists of two time series where the first has a low input amplitude (`train_val_1` and `test_1`) and the second a high input amplitude (`train_val_2` and `test_2`).  

You can use both training sets in your training, and please report the RMSE values on both test sets separately. 

## Cascaded_Tanks

![image](figures/Cascaded_Tanks.jpg)

```python
train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
print(test.n_initialization_samples) # = 4
train_val_u, train_val_y = train_val
test_u, test_y = test
```


## WienerHammerBenchMark

![image](figures/WienerHammerBenchMark.jpg)

```python
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
print(test.n_initialization_samples) # = 50
train_val_u, train_val_y = train_val
test_u, test_y = test
```


## Silverbox

![image](figures/Silverbox.jpg)

```python
train_val, test = nonlinear_benchmarks.Silverbox()
multisine_train_val = train_val
print(test[0].n_initialization_samples) # = 50 (for all test sets)
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
```

Note that the test arrow full and the test arrow no extrapolation have some overlap.

# Error Metrics

We also provide error metrics in `nonlinear_benchmarks.error_metrics`.

```python
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index

#generate example ouput data and prediction 
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100)/100

print(f"RMSE: {RMSE(y_true, y_pred)} (Root Mean Square Error)")
print(f"NRMSE: {NRMSE(y_true, y_pred)} (Normalized Root Mean Square Error)")
print(f"R-squared: {R_squared(y_true, y_pred)} (coefficient of determination R^2)")
print(f'MAE: {MAE(y_true, y_pred)} (Mean Absolute value Error)')
print(f"fit index: {fit_index(y_true, y_pred)} (https://arxiv.org/pdf/1902.00683.pdf page 31)")
```
