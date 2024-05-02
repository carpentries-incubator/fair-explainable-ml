---
```python
import numpy as np
import pandas as pd

from IPython.display import Markdown, display

%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.datasets import MEPSDataset19

from fairlearn.postprocessing import ThresholdOptimizer

from collections import defaultdict
```
This notebook is adapted from AIF360's [Medical Expenditure Tutorial](https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb).

The tutorial uses data from the [Medical Expenditure Panel Survey](https://meps.ahrq.gov/mepsweb/). We include a short description of the data below. For more details, especially on the preprocessing, please see the AIF360 tutorial. 
## Scenario and data

The goal is to develop a healthcare utilization scoring model -- i.e., to predict which patients will have the highest utilization of healthcare resources. 

The original dataset contains information about various types of medical visits; the AIF360 preprocessing created a single output feature 'UTILIZATION' that combines utilization across all visit types. Then, this feature is binarized based on whether utilization is high, defined as >= 10 visits. Around 17% of the dataset has high utilization. 

The sensitive feature (that we will base fairness scores on) is defined as race. Other predictors include demographics, health assessment data, past diagnoses, and physical/mental limitations. 

The data is divided into years (we follow the lead of AIF360's tutorial and use 2015), and further divided into Panels. We use Panel 19 (the first half of 2015).
### Loading the data

First, the data needs to be moved into the correct location for the AIF360 library to find it. If you haven't yet, run `setup.sh` to complete that step. (Then, restart the kernel and re-load the packages at the top of this file.)

First, we load the data. Next, we create the train/validation/test splits and setup information about the privileged and unprivileged groups. (Recall, we focus on race as the sensitive feature.)
```python
(dataset_orig_panel19_train,
 dataset_orig_panel19_val,
 dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]

```
Show details about the data.
```python
def describe(train=None, val=None, test=None):
    if train is not None:
        display(Markdown("#### Training Dataset shape"))
        print(train.features.shape)
    if val is not None:
        display(Markdown("#### Validation Dataset shape"))
        print(val.features.shape)
    display(Markdown("#### Test Dataset shape"))
    print(test.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(test.favorable_label, test.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(test.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(test.privileged_protected_attributes, 
          test.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names\n See [MEPS documentation](https://meps.ahrq.gov/data_stats/download_data/pufs/h181/h181doc.pdf) for details on the various features"))
    print(test.feature_names)

describe(dataset_orig_panel19_train, dataset_orig_panel19_val, dataset_orig_panel19_test)
```
Next, we will look at whether the dataset contains bias; i.e., does the outcome 'UTILIZATION' take on a positive value more frequently for one racial group than another?

The disparate impact score will be between 0 and 1, where 1 indicates *no bias*.
```python
metric_orig_panel19_train = BinaryLabelDatasetMetric(
        dataset_orig_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

print(explainer_orig_panel19_train.disparate_impact())

```
We see that the disparate impact is about 0.48, which means the privileged group has the favorable outcome at about 2x the rate as the unprivileged group does. 

(In this case, the "favorable" outcome is label=1, i.e., high utlization)
## Train a model

We will train a logistic regression classifier.
```python
dataset = dataset_orig_panel19_train
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
```
### Validate the model
Recall that a logistic regression model can output probabilities (i.e., `model.predict(dataset).scores`) and we can determine our own threshold for predicting class 0 or 1. 

The following function, `test`, computes performance on the logistic regression model based on a variety of thresholds, as indicated by `thresh_arr`, an array of threshold values. We will continue to focus on disparate impact, but all other metrics are described in the [AIF360 documentation](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric). 
```python
def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError as e:
        print(e)
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
        
    pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    metric_arrs = defaultdict(list)
    
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        # various metrics - can look up what they are on your own
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs
```
```python
thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=lr_orig_panel19,
                   thresh_arr=thresh_arr)
lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])
```
We will plot `val_metrics`. The x-axis will be the threshold we use to output the label 1 (i.e., if the raw score is larger than the threshold, we output 1). 

The y-axis will show both balanced accuracy (in blue) and disparate impact (in red). 

Note that we plot 1 - Disparate Impact, so now a score of 0 indicates no bias.
```python
def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
```
```python
disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - disp_imp
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - DI')
```
If you like, you can plot other metrics, e.g., average odds difference.

In the next cell, we write a function to print out a variety of other metrics. Since we look at 1 - disparate impact, **all of these metrics have a value of 0 if they are perfectly fair**. Again, you can learn more details about the various metrics in the [AIF360 documentation](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric).
```python
def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    disp_imp_at_best_ind = 1 - metrics['disp_imp'][best_ind]
    print("\nCorresponding 1-DI value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))

describe_metrics(val_metrics, thresh_arr)
```
### Test the model
Now that we have used the validation data to select the best threshold, we will evaluate the test the model on the test data.
```python
lr_metrics = test(dataset=dataset_orig_panel19_test,
                       model=lr_orig_panel19,
                       thresh_arr=[thresh_arr[lr_orig_best_ind]])
describe_metrics(lr_metrics, [thresh_arr[lr_orig_best_ind]])
```
## Mitigate bias with in-processing
We will use reweighting as an in-processing step to try to increase fairness. AIF360 has a function that performs reweighting that we will use. If you're interested, you can look at details about how it works in [the documentation](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.Reweighing.html). 

If you look at the documentation, you will see that AIF360 classifies reweighting as a preprocessing, not an in-processing intervention. Technically, AIF360's implementation modifies the dataset, not the learning algorithm so it is pre-processing. But, it is functionally equivalent to modifying the learning algorithm's loss function, so we follow the convention of the fair ML field and call it in-processing.
```python
# Reweighting is a AIF360 class to reweight the data 
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)
```
We'll also define metrics for the reweighted data and print out the disparate impact of the dataset.
```python
metric_transf_panel19_train = BinaryLabelDatasetMetric(
        dataset_transf_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_transf_panel19_train = MetricTextExplainer(metric_transf_panel19_train)

print(explainer_transf_panel19_train.disparate_impact())
```
Then, we'll train a model, validate it, and evaluate of the test data.
```python
# train
dataset = dataset_transf_panel19_train
model = make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

```
```python
# validate
thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_orig_panel19_val,
                   model=lr_transf_panel19,
                   thresh_arr=thresh_arr)
lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])
```
```python
# plot validation results
disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')
```
```python
# describe validation results
describe_metrics(val_metrics, thresh_arr)
```
### Test
lr_transf_metrics = test(dataset=dataset_orig_panel19_test,
                         model=lr_transf_panel19,
                         thresh_arr=[thresh_arr[lr_transf_best_ind]])
describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])
We see that the disparate impact score on the test data is better after reweighting than it was originally.

How do the other fairness metrics compare?
## Mitigate bias with preprocessing
We will use a method, [ThresholdOptimizer](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html#fairlearn.postprocessing.ThresholdOptimizer), that is implemented in the library [Fairlearn](https://fairlearn.org/). ThresholdOptimizer finds custom thresholds for each demographic group so as to achieve parity in the desired group fairness metric.

We will focus on demographic parity, but feel free to try other metrics if you're curious on how it does.

The first step is creating the ThresholdOptimizer object. We pass in the demographic parity constraint, and indicate that we would like to optimize the balanced accuracy score (other options include accuracy, and true or false positive rate -- see [the documentation](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html#fairlearn.postprocessing.ThresholdOptimizer) for more details). 
```python
to = ThresholdOptimizer(estimator=model, constraints="demographic_parity", objective="balanced_accuracy_score", prefit=True)
```
Next, we fit the ThresholdOptimizer object to the validation data.
```python
to.fit(dataset_orig_panel19_val.features, dataset_orig_panel19_val.labels, 
       sensitive_features=dataset_orig_panel19_val.protected_attributes[:,0])
```
Then, we'll create a helper function, `mini_test` to allow us to call the `describe_metrics` function even though we are no longer evaluating our method as a variety of thresholds.

After that, we call the ThresholdOptimizer's predict function on the validation and test data, and then compute metrics and print the results.
```python
def mini_test(dataset, preds):
    metric_arrs = defaultdict(list)
    dataset_pred = dataset.copy()
    dataset_pred.labels = preds
    metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

    # various metrics - can look up what they are on your own
    metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                    + metric.true_negative_rate()) / 2)
    metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    metric_arrs['disp_imp'].append(metric.disparate_impact())
    metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
    metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs
```
```python
to_val_preds = to.predict(dataset_orig_panel19_val.features, sensitive_features=dataset_orig_panel19_val.protected_attributes[:,0])
to_test_preds = to.predict(dataset_orig_panel19_test.features, sensitive_features=dataset_orig_panel19_test.protected_attributes[:,0])
```
```python
to_val_metrics = mini_test(dataset_orig_panel19_val, to_val_preds)
to_test_metrics = mini_test(dataset_orig_panel19_test, to_test_preds)
```
```python
print("Remember, `Threshold corresponding to Best balanced accuracy` is just a placeholder here.")
describe_metrics(to_val_metrics, [0])
```
```python
print("Remember, `Threshold corresponding to Best balanced accuracy` is just a placeholder here.")

describe_metrics(to_test_metrics, [0])
```
Scroll up and see how these results compare with the original classifier and with the in-processing technique. 

A major difference is that the accuracy is lower, now. In practice, it might be better to use an algorithm that allows a custom tradeoff between the accuracy sacrifice and increased levels of fairness.

We can also see what threshold is being used for each demographic group by examining the `interpolated_thresholder_.interpretation_dict` property of the ThresholdOptimzer.
```python
threshold_rules_by_group = to.interpolated_thresholder_.interpolation_dict
threshold_rules_by_group
```
Recall that a value of 1 in the Race column corresponds to White people, while a value of 0 corresponds to non-White people.

Due to the inherent randomness of the ThresholdOptimizer, you might get slightly different results than your neighbors. When we ran the previous cell, the output was

`
{0.0: {'p0': 0.9287205987170348,
  'operation0': [>0.5],
  'p1': 0.07127940128296517,
  'operation1': [>-inf]},
 1.0: {'p0': 0.002549618320610717,
  'operation0': [>inf],
  'p1': 0.9974503816793893,
  'operation1': [>0.5]}}
`

This tells us that for non-White individuals:

* If the score is above 0.5, predict 1. 

* Otherwise, predict 1 with probability 0.071

And for White individuals:

* If the score is above 0.5, predict 1 with probability 0.997


**Discussion question:** what are the pros and cons of improving the model fairness by introducing randomization?
