*“The sun's rays do not burn until brought to a focus” (Alexander Graham Bell)*

# Introduction
This repo contains the artifacts of various AutoML and ML Experiments on [Kaggle's Feb 2021 Tabular Contest Dataset](https://www.kaggle.com/c/tabular-playground-series-feb-2021).

# Setup of the Experiments

The initial promising attempt to build the ML Model (https://www.kaggle.com/gvyshnya/ensemble-lgb-xgb-catboost-optimized) was engineered as follows

-	Weighted Ensemble of several GBDT-style models (lightgbm, xgboost, and catboost);
-	Basic feature engineering (cat variables label-encoded, and numeric variables passed to the standard scaler);
-	Hyperparameters for each of GBDT models in the ensemble searched via an appropriate AutoML tool (hyperopt, in this case);
-	10-fold CV of the ensembled model to verify its error metric.

Other experiments that did not work well enough demonstrated that

-	Additional feature engineering with naïve approach (that is, adding basic numeric interaction features - sum, differences, products etc., - polynomial features, statistically calculated features etc.) did not add the edge – each time adding such features decreased the model performance
-	OHE of categorical features worked worse for GBDT models vs. the label encoding (this is also confirmed by other contest participants, for instance, here: https://www.kaggle.com/dwin183287/tps-feb-2021-base-model-features-engineering)
-	Target encoding the cat variables by some of the numeric variables drastically increased the error metric value (more experiments will have to be taken to see if it is beneficial for this contest)
-	Full pipeline AutoML-backed models did not work equally well vs. the manually tuned individual GBDT models or ensembles with GDBT models tuned via hyperopt-based hyperparameter search

Regarding the full pipeline AutoML-backed solutions, I would like to share one observation though. 

-	AutoViML-backed solution trained good-enough model (scored 0.85252 on the public LB) in less then 10 min (see the code of that experiment in https://github.com/gvyshnya/tab-feb-2021/blob/main/AutoViML%20Baseline%20Prediction.ipynb )
-	H2O AutoML-backed solution scored 0.8611 on the public LB , with 8+ h of training spent to achieve such a result (see the code of this experiment in https://github.com/gvyshnya/tab-feb-2021/blob/main/H2O%20AutoML%20Raw%20Features%20Prediction.ipynb) 

Although such solutions did not display the performance on a par with the manually orchestrated GBDT ensemble models, it was still a good experience trying such an approach.

Then I implemnted the 7-step extreme training for a *lightgbm* model (see *extreme-fine-tuning-lgbm-using-7-step-training-GV.ipynb*), and it scored quite well (scored 0.84258 on the public LB, 0.84192 on the private LB).

# Additional Ideas

I observed several productive ideas already shared by other colleagues here. For instance, these are
-	Interesting SOM and feature transformation ideas from https://www.kaggle.com/desareca/eda-bimodal-distribution-dae-som
-	Interesting data pre-processing suggestions shared in https://www.kaggle.com/maunish/lgbm-goes-brrr

However, I did not have time to entertain such ideas.

# Files and Folders

- */data* subfolder contains the copy of the dataset of [Kaggle's Feb 2021 Tabular Contest](https://www.kaggle.com/c/tabular-playground-series-feb-2021)
- *AutoViML Baseline Prediction.ipynb* - the ML experiment to build a prediction model, using [AutoViML](https://github.com/AutoViML/AutoViz), one of the popular freeware full pipeline AutoML tools
- *Generic Express EDA with Comprehensive insights.ipynb* - the comprehensive EDA for the dataset, using [AutoViz](https://github.com/AutoViML/AutoViz), one of the popular Rapid EDA tools available in the market
- *H2O AutoML Raw Features Prediction.ipynb* - the baseline ML model built with H2O AutoML
- *LightGBM Best Models Prediction.ipynb* - the series of ML experiments to train *lightgbm* models in a manual fashion
- *LightGBM LE Perm FI model scoring.ipynb* - the notebook with a series of ML experiments to tune the baseline *lightgbm* model under the different data preprocessing/feature engineering flows (OHE vs. label encoding of cat features; applying log transform to the numeric features and target variables vs. not applying, detecting the most impactful features based on their permutative feature importance scores etc.)
- *LightGBM Raw Features Tuned.ipynb* - the notebook with the log of ML experiments with the time-effective method of the manual parameter tuning for GDBT models (only original raw features used, with label encoding applied to the category features)
- *Tab-Feb-2021 EDA and Feature Engineering Insights.ipynb*
- *ensemble-lgb-xgb-cat-other-params.ipynb*
- *ensemble-lgb-xgb-catboost-optimized.ipynb*
- *ensemble-lgb-xgb-with-hyperopt.ipynb*
- *extreme-fine-tuning-lgbm-using-7-step-training-GV.ipynb* - the best manually set lightgbm ML model, with the 7-step extreme training technique applied

# References

The materials of the experiments documented in this repo have been also discussed in the publications below

- https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/221021
- https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/217483