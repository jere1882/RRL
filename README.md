# RRL_classification

Python code of experiments carried out during my master's thesis. SKLearn implementations were used throughout.

This thesis builts on the following paper:

```
Cabral, J. B., Ramos, F., Gurovich, S., & Granitto, P. (2020). 
Automatic Catalog of RRLyrae from ∼ 14 million VVV Light Curves: How far can we go with traditional machine-learning?
https://arxiv.org/abs/2005.00220
```

It consists of classifying  RR-Lyrae stars in massive astronomical datasets using machine learning methods.

Section 1: Random Forests
- cross validation grid search in order to optimize hyperparameters.

Section 2: Linear SVM 
- cross validation grid search in order to optimize hyperparameters.
- effect of C, effect of undersampling

Section 3: SVM-RBF
- cross validation grid search in order to optimize hyperparameters.
- effect of C and gamma
- kernel approximators: Nystroem
- effect of undersampling

Sections 4, 5 and 6:  Preprocessing in SVM
- Scaling (MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, etc)
- Transformations ( KbinsDiscretizer , QuantileTransf..)

Section 7: Feature selection and dimensionality reduction
- Univariate Feature Selection (f_classif, mutual_info_classif, chi2)
- Variable importance
- PCA
– Feature Agglomeration
- Correlation of features

Section 8: Visualization
- Univariate distribution of each feature

Section 9: Dealing with class imbalance
- class weight in SVM
– Oversampling: Naive, SMOTE, ADASYN
