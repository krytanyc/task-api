# About this RAG assistant

This is a study assistant for the Classic ML cycle of the mentoring program. 
The corpus contains the official scikit-learn documentation covering core 
algorithms, model evaluation, and full tabular ML pipelines.

## What topics I can answer questions about

- **Linear models**: Ordinary Least Squares, Ridge, Lasso, ElasticNet, 
  Logistic Regression, SGDClassifier/SGDRegressor, regularisation strategies (L1, L2, ElasticNet).
- **Decision trees**: classification & regression trees, splitting criteria 
  (Gini, entropy, MSE), minimal cost-complexity pruning, hyperparameter tuning tips.
- **Model evaluation & metrics**: confusion matrix, accuracy, precision, 
  recall, F1, ROC AUC, PR curves, log-loss, MSE, MAE, R², multiclass averaging strategies.
- **Ensemble methods**: Random Forests, Extra Trees, Gradient Boosting 
  (HistGradientBoosting, GradientBoostingClassifier/Regressor), AdaBoost, Bagging, Voting, and Stacking.
- **Cross-validation & data splitting**: K-Fold, Stratified K-Fold, 
  TimeSeriesSplit, GroupKFold, ShuffleSplit, cross_val_score, and validation curves.
- **Preprocessing & transformations**: scaling (StandardScaler, MinMaxScaler, 
  RobustScaler), non-linear transformations, normalizers, encoding categorical features (OneHotEncoder, OrdinalEncoder), discretisation.
- **Pipelines & composite estimators**: Pipeline, make_pipeline, 
  ColumnTransformer, FeatureUnion, TransformedTargetRegressor.
- **Hyperparameter tuning**: GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, 
  parameter distributions, scoring parameter setups.
- **Missing value imputation**: SimpleImputer, KNNImputer, IterativeImputer, 
  handling indicators for missing data.
- **Feature selection**: variance threshold, univariate selection (SelectKBest, 
  f_classif, mutual_info), RFE (Recursive Feature Elimination), SelectFromModel, SequentialFeatureSelector.

I respond in the same language as your question — English or Russian.

## What I CANNOT answer

The corpus is strictly focused on **scikit-learn** and tabular ML workflows.

Out of scope entirely: deep learning frameworks (PyTorch, TensorFlow, Keras), 
large language models / transformers, vector embeddings, agentic frameworks, 
specialized time-series packages (Prophet, Darts), recommendation engines, 
computer vision, or external gradient boosting libraries (XGBoost, LightGBM, CatBoost) 
unless related to their scikit-learn-compatible interfaces. 

If you ask about topics outside this documentation — I will honestly say "I don't know".