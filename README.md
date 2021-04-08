*“The sun's rays do not burn until brought to a focus” (Alexander Graham Bell)*

# Introduction
This repo contains the artifacts of various AutoML and ML Experiments on Kaggle's Feb 2021 Tabular Contest Dataset

# Setup of the Experiments

The initial promising attempt to build the ML Model (https://www.kaggle.com/gvyshnya/ensemble-lgb-xgb-catboost-optimized), scored 0.84247 on the public LB, was engineered as follows

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

Then I implemnted the extreme training for a lightgbm model, and it scored quite well.

# Additional Ideas
I observed several productive ideas already shared by other colleagues here. For instance, these are
-	Interesting SOM and feature transformation ideas from https://www.kaggle.com/desareca/eda-bimodal-distribution-dae-som
-	Interesting data pre-processing suggestions shared in https://www.kaggle.com/maunish/lgbm-goes-brrr

However, I did not have time to entertain such ideas.

# Files and Folders

TBD

# References

The materials of the experiments documented in this repo have been also discussed in the publications below

- https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/221021
- https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/217483