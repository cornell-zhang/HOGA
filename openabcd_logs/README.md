HOGA on OpenABC-D
===============================

Note
------------
We mainly followed the [OpenABC-D implementation](https://github.com/NYU-MLDA/OpenABC), including the computation of the MAPE score using the API from the [sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html). However, unlike the standard MAPE metric implemented by [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsolutePercentageError), we later discovered that the sklearn-based MAPE does not
include the scaling factor of 100, which we initially expected. As a result, our reported results are two orders of magnitude smaller than the actual MAPE scores. In other words, the ‘%’ symbol should be disregarded for all results in Table 2 of our paper to accurately reflect their MAPE scores. For example, the 5.0% average MAPE actually indicates a MAPE score of 5.0. That being said, this does not affect our claims in the paper, as both HOGA and the baseline model were evaluated in the same way (i.e., both are 100 times smaller than their actual MAPE scores).


Reproducibility
------------
For simplicity, we only provide HOGA predictions on the test set, from which you can easily calculate the MAPE for each test design, as well as generate the plots in Figure 4 of our paper, using the following command:
```
python analyze.py
```
