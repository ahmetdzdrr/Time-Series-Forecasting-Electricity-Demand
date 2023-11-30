# Time-Series-Forecasting-Electricity-Demand

## **Table of Contents**

- [Introduction](#introduction)
- [What is Time Series Forecasting?](#what-is-time-series-forecasting)
- [Project Overview](#project-overview)
- [Machine Learning Methods](#machine-learning-methods)
- [Deep Learning Methods](#deep-learning-methods)
- [Installation](#installation)
- [Usage Code](#usage)
- [How to Use Optuna?](#how-to-use-optuna)


## **Introduction**

The electricity demand project focuses on predicting energy consumption, utilizing a dataset covering the years 2014, 2015, 2016, and 2017. The dataset encompasses hourly records and includes various environmental features alongside the electricity demand. The task involves forecasting the electricity demand for the initial three months of 2018.

The dataset consists of several columns providing diverse environmental factors:

- **datetime**: Time stamp for each hourly record.
- **precipitation**: Quantity of precipitation during the given hour.
- **temperature**: Hourly temperature measurement.
- **irradiance_surface**: Surface irradiance data.
- **irradiance_toa**: Irradiance data measured at the top of the atmosphere.
- **snowfall**: Amount of snowfall during the hour.
- **snow_mass**: Measurement of snow mass.
- **cloud_cover**: Percentage of cloud cover.
- **air_density**: Air density during the specific hour.
- **demand**: The electricity demand recorded for each hour.

The objective of this project is to employ machine learning techniques to build models that can accurately forecast the electricity demand for the first quarter of 2018, based on historical patterns and the provided environmental data.

## **What is Time-Series-Forecasting?**

Time series forecasting is a statistical technique used to predict future data points based on previously observed values in a chronological sequence. It's specifically designed for data that is collected and indexed in time order, such as hourly, daily, monthly, or yearly intervals.

The primary goal of time series forecasting is to uncover patterns, trends, and relationships within the data to make predictions about future values. This method accounts for the inherent sequential dependencies in the data, considering factors like seasonality, trends, cyclic patterns, and irregular fluctuations.

Various methods, including statistical models and machine learning algorithms, are employed in time series forecasting. These methods utilize historical data to develop models that capture the underlying structure of the time series. The models are then used to forecast future values, enabling businesses and analysts to make informed decisions, plan resources, and anticipate future trends based on past observations.

Overall, time series forecasting plays a vital role in numerous fields, including finance, economics, weather forecasting, inventory management, and many others, aiding in making accurate predictions and facilitating strategic decision-making processes.

## **Project Overview**

> ### Dataset Descriptive Analysis

Utilized describe_dataset function to generate descriptive statistics such as **mean**, **standard deviation**, **percentiles**, etc., providing an overview of the dataset.
Analyzed the descriptive statistics to understand the distribution and characteristics of the data.

> ### Missing Values Handling

- Employed the **check_missing_values** function to assess and visualize missing values in the dataset.
- Presented missing values as a percentage and visualized using horizontal bar plots for better comprehension.

> ### Correlation Analysis

- Conducted a correlation matrix using the **correlation_matrix** function to explore relationships among numerical features.
- Visualized the correlation matrix as a heatmap to identify correlated features within the dataset.

> ### Data Summary

- Executed the **grab_col_names** function to categorize features based on data type and unique value counts.
- Reported the number of **observations**, **variables**, **categorical columns**, and **numerical columns** to provide an overview of the dataset's composition.

> ### Data Distribution Visualization

- Utilized **boxplot** and **hist_plot** functions to visualize the distribution and characteristics of numerical features.
- Displayed box plots and histograms to identify **outliers**, **distributions**, and central tendencies in the data.

> ### Outlier Detection

- Implemented **check_outlier** and **OutlierDetection** functions to identify outliers within numerical columns.
- Applied **Isolation Forest** and **Local Outlier Factor (LOF)** algorithms to detect and mark anomalies in the dataset, if enabled.

> ### Datetime Feature Engineering

- Applied **datetime_process** to convert a specified date column to datetime format.
- Extracted various date and time-related features such as **year**, **quarter**, **month**, **week**, etc., for further analysis or modeling.

## **Machine Learning Methods**

The following machine learning methods have been implemented and applied to the dataset:

> ### Scaler Processing

- Utilized the Scaler class to scale numerical features in the dataset.
- Implemented **MinMaxScaler**, **StandardScaler**, and **RobustScaler** based on the configuration settings.

> ### Optuna Hyperparameter Optimization

- Utilized the Optuna class to perform hyperparameter optimization using Optuna library.
The optimization process is executed for the following regression models:
- HistGradientBoostingRegressor
- LGBMRegressor
- XGBRegressor
- CatBoostRegressor

Hyperparameter ranges and optimization objectives are tailored for each model.

> ### Machine Learning Model Training

The MLModels class is employed to train and evaluate machine learning models based on the best parameters obtained from the Optuna optimization.
The following models are considered:

- HistGradientBoostingRegressor
- LGBMRegressor
- XGBRegressor
- CatBoostRegressor

> ### Model Evaluation Metrics

- Evaluated the performance of the trained models using various metrics such as **Root Mean Squared Error (RMSE)**, **R-squared (R2)**, and **Mean Absolute Error (MAE)**.
- Displayed the results for each model, showcasing their effectiveness in predicting the target variable.

> ### Model Summary

- Summarized the functionality of each model and their corresponding evaluation metrics.
- Provided an overview of the machine learning methods applied to the dataset for regression tasks.

**Note:** All processes are enabled or disabled based on the configuration settings, allowing for flexibility in the analysis and model training pipeline.

## Deep Learning Methods

The implemented code comprises various deep learning methods, predominantly focusing on Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models, including their combinations.

> ### LSTM Model Building

- Created via the build_lstm_model function, allowing hyperparameter tuning.
- Configured with multiple LSTM layers, dropout regularization, and kernel regularization.
- Optimized using the Adam optimizer with mean squared error loss.

> ### GRU Model Building

- Constructed through the build_gru_model function, also adjustable via hyperparameters.
- Comprises GRU layers, dropout regularization, kernel regularization, and Adam optimizer with mean squared error loss.

> ### Combined LSTM-GRU Model Building

- Developed using the build_lstm_gru_model function, facilitating hyperparameter tuning.
- Integrates both LSTM and GRU layers, dropout regularization, kernel regularization, and the Adam optimizer for mean squared error loss.

> ### KerasTuner for Hyperparameter Optimization

- Utilized within the KerasTuner class to tune hyperparameters for LSTM, GRU, and combined LSTM-GRU models.
- Employs RandomSearch to explore hyperparameter spaces and identify optimal configurations for each model type.

> ### Deep Learning Model Execution

- Performed through the DLModels class, which enables the selection and execution of specific deep learning models based on configuration flags.
- Facilitates training, validation, and prediction using LSTM, GRU, or combined LSTM-GRU architectures.
- Trains models with the Adam optimizer, Nadam optimizer, and mean squared error loss.

These methods allow for flexible model creation, tuning, and execution, catering to various sequence prediction tasks, particularly suitable for time series forecasting or sequential data analysis. The combination of LSTM and GRU architectures provides versatility and potential improvements in capturing temporal dependencies within the data, aiding in achieving accurate predictions for target variables.

## Installation

To get started with this project, follow these steps:

1. Clone this repository to your local machine using Git:

         git clone https://github.com/ahmetdzdrr/Time-Series_Forecasting-Electricity-Demand.git

2. Install the required Python libraries by running:

          pip install -r requirements.txt

## Usage Code

> ### Usage
To run the project, follow these steps:

- To use this code, you can customize its behavior by modifying the CFG class in the main script (multi_label_classification.ipynb). 

- Each flag in the CFG class controls whether a specific functionality is enabled or disabled. Set the flags to True or False based on your requirements.

- Open the Jupyter Notebook file (multi_label_classification.ipynb) in your Jupyter Notebook environment.

- Run each cell in the notebook sequentially. The code in the notebook will process the data, perform the selected operations, and generate the desired output.

> ### How to Use Optuna?

- Optuna is a Python library for optimizing machine learning model hyperparameters. You can use it with various machine learning frameworks, including XGBoost, LightGBM (LGBM), and CatBoost. Here's a short guide on how to do that:

>> ### Step 1: Install Optuna

- You need to install Optuna in your Python environment. You can do this using pip:

       pip install optuna

>> ### Step 2: Import Optuna and the Machine Learning Library

- In your Jupyter Notebook or Python script, import Optuna and the machine learning library you want to optimize (e.g., XGBoost, LGBM, or CatBoost).

      import optuna
      import xgboost as xgb
      import lightgbm as lgb
      from catboost import CatBoostClassifier

>> ### Step 3: Define an Objective Function

- Create an objective function that Optuna will optimize. This function takes an Optuna trial object as an argument and returns a score that you want to minimize or maximize. This score is typically a metric of your model's performance.

Here's an example for optimizing the AUC score with XGBoost:

      def objective(trial):
          params = {
              "objective": "binary:logistic",
              "eval_metric": "auc",
              "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
              "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
              # Add more hyperparameters to tune
          }
      
       dtrain = xgb.DMatrix(X_train, label=y_train)
       model = xgb.train(params, dtrain)
       predictions = model.predict(dtest)
      
       auc = sklearn.metrics.roc_auc_score(y_test, predictions)
       return auc
       
>> ### Step 4: Create an Optuna Study

- Create an Optuna study to run the optimization. You can specify the optimization direction, e.g., "maximize" or "minimize" based on your chosen objective (e.g., AUC).

    ```bash
      study = optuna.create_study(direction="maximize")

>> ### Step 5: Start the Optimization

-Run the optimization process with a specified number of trials. Optuna will search for the best hyperparameters based on the objective function.
    
      study.optimize(objective, n_trials=100)
      
>> ### Step 6: Retrieve the Best Parameters

- Once the optimization is complete, you can retrieve the best set of hyperparameters from the study object:

      best_params = study.best_params

- You can then use these best hyperparameters to train your final model with XGBoost, LGBM, or CatBoost.
