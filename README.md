# Repository for "Stable Optimization of Gaussian Likelihoods", ICLR 2023 Submission


## Setup
Setup a virtual environment, install requirements with
```
py -m pip install -r requirements_py3101.txt
py -m pip install -e projections
```


## Non-Contextual Experiments
- run_1d_gradient_plot : generates 1d and 3d plots of gradients (i.e. fig. 2a)
- run_1d_multiple_parametrization_graphs : plots the 1d samples from the appendix
- run_2d_gradient_plot : plots bivariate gradient magnitude scatter plot (i.e. fig. 2b)
- run_10d_multiple_parametrization_graphs : runs and plots 10d runs from the appendix and fig 3


## Contextual Experiments


### NNConstant, Pitfalls, Detlefsen

Use the parameters
```
function = { nn_constant, pitfalls, detlefsen }
model = { AdamModel, PitfallsModel, TractableModel, TRPLW2Model, TrustableW2Model }
model_type = { simple_model, complex_model } (respective 50x50 and 50x50x50)
```
and run with
```
run_simplefunction.py <function> <model> <model_type>
```

Run all models for one function / model_type and then plot + print results with
```
plot_simplefunction.py <function> <model_type>
```


### 3D Spiral
Use the parameters
```
model = { AdamModel, PitfallsModel, TractableModel, TRPLW2Model, TrustableW2Model }
```
and run with
```
run_complexfunction.py 3dspiral_sm5 <model>
```

Run all models for one function and then plot + print results with
```
plot_complexfunction.py <function>
```


### UCI Univariate
Use the parameters
```
uci_set = { carbon, concrete, energy, housing, kin8m, naval, power, yacht }
model = { AdamModel, Pitfalls05Model, Pitfalls10Model, TrustableW2Model }
```
and run with
```
run_uci_univariate.py <uci_set> <model>
```

Run all models for one set and then plot + print results with
```
run_uci_univariate.py <uci_set>
```


### UCI Multivariate
Use the parameters
```
uci_set = { carbon, energy }
model = { AdamModel, Pitfalls05Model, Pitfalls10Model, TrustableW2Model }
```
and run with
```
run_uci_multivariate.py <uci_set> <model>
```

Run all models for one set and then plot + print results with
```
run_uci_multivariate.py <uci_set>
```