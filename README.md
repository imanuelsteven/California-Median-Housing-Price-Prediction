
# California Median Housing Price Prediction 

This project aims to build and evaluate a machine learning model to predict California housing prices based on the 1990 California Census data. The project follows a standard machine learning workflow, including data understanding, exploratory data analysis (EDA), data preprocessing, modeling, hyperparameter tuning, and evaluation.

## Dataset Overview

The dataset used is the California Housing Prices dataset from the 1990 California Census. It contains various features describing housing blocks and the target variable, `median_house_value`.

| Feature            | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `longitude`        | How far west the block is located; higher = further west                               |
| `latitude`         | How far north the block is located; higher = further north                             |
| `housingMedianAge` | Median age of houses in the block; lower = newer buildings                             |
| `totalRooms`       | Total number of rooms in the block (not just per house)                                |
| `totalBedrooms`    | Total number of bedrooms in the block                                                  |
| `population`       | Total number of people living in the block                                             |
| `households`       | Total number of households in the block (household = group in a housing unit)          |
| `medianIncome`     | Median income in the block (measured in tens of thousands of USD; e.g. 3.5 = \$35,000) |
| `medianHouseValue` | Median house value in the block (in USD)                                               |
| `oceanProximity`   | Proximity of the block to the ocean (categories like ‘NEAR OCEAN’, ‘INLAND’)           |

## Project Steps

1.  **Data Understanding**: Loaded and examined the dataset to understand its structure, feature types, and initial statistics. Key observations included scaled `median_income`, capped `median_house_value`, missing values in `total_bedrooms`, and varying feature scales.

2.  **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets using **stratified sampling** based on the `median_income` feature to ensure that the distribution of income categories was representative in both sets. This helps prevent data leakage and ensures the model's generalizability.

3.  **Exploratory Data Analysis (EDA)**: Performed detailed analysis on the training set.
    *   Confirmed missing values in `total_bedrooms` and decided to handle them using the median due to the skewed distribution.
    *   Visualized geographical data (`latitude` and `longitude`) to observe the spatial distribution of house prices, confirming the strong influence of location (especially proximity to the ocean) and population density.
    *   Analyzed correlations between features and the target variable, identifying `median_income` as the strongest predictor. Noted the capping effect in `median_house_value`.
    *   Engineered new features like `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` to provide more insightful metrics at the household level. These engineered features showed stronger correlations with the target variable than their raw counterparts.
    *   Handled the categorical feature `ocean_proximity` using One-Hot Encoding.
    *   Addressed skewed distributions in numerical features by applying `log1p` transformation and scaling using `RobustScaler`.

4.  **Data Preprocessing**: Built a comprehensive data preprocessing pipeline using `ColumnTransformer` and `Pipeline` to handle missing values, add engineered features, apply transformations, and scale numerical features, and one-hot encode categorical features. This pipeline ensures that data preprocessing steps are applied consistently to both training and test sets.

5.  **Modeling**: Trained and evaluated several regression models using the preprocessed training data:
    *   Linear Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
    *   XGBoost Regressor

    Cross-validation (CV=5) was used to assess the performance of each model and mitigate overfitting. The **Random Forest Regressor** showed the best performance among the initial models based on cross-validation scores.

6.  **Hyperparameter Tuning**: Applied **Bayesian Optimization** (`BayesSearchCV`) to fine-tune the hyperparameters of the best-performing model, the Random Forest Regressor. This process efficiently searched for the optimal combination of parameters (`n_estimators`, `max_features`, `max_depth`) to maximize the model's performance (evaluated using negative mean squared error).

7.  **Model Evaluation**: Evaluated the final optimized Random Forest model on the **untouched test set**. Key metrics calculated were:
    *   **R² Score**: 0.7948 (Explains about 79.5% of the variance in housing prices)
    *   **MAPE**: 20.32% (Predictions are, on average, about 20.32% off from actual values)
    *   **RMSE**: \$51,720.04 (The average prediction error is around \$51.7K)

    The residual plot indicated that the model performs better for lower to mid-range house values and the prediction error increases for higher-priced houses.

8.  **Feature Importance Analysis**: Analyzed the feature importance from the final Random Forest model, confirming that `median_income` is the most important feature, followed by geographic features (`ocean_proximity_INLAND`, `latitude`) and engineered features (`population_per_household`, `bedrooms_per_room`).

## Conclusion and Future Improvements

The project successfully developed a Random Forest model for predicting California housing prices, achieving an R² of 0.7948 and an RMSE of \$51,720.04 on the test set.

**Potential areas for future improvement include:**

*   **Addressing the target variable capping:** Investigate methods to handle or remove the capped values (\$500,000) if predicting higher-end homes is a critical requirement.
*   **Exploring other advanced algorithms:** Evaluate other algorithms like LightGBM, CatBoost, or deep learning approaches that might capture more complex patterns.
*   **More extensive feature engineering:** Create additional features based on geographical clusters or more complex interactions between existing features.
*   **Outlier treatment:** Implement specific strategies to handle outliers beyond Robust Scaling if they are significantly impacting performance.
*   **Collecting more recent data:** Use more up-to-date housing data if available, as market dynamics change over time.
*   **Error Analysis**: Deep dive into the residuals to understand where the model struggles most (e.g., specific geographic areas, price ranges, or property types).

The final trained model was saved using `joblib` for future use.