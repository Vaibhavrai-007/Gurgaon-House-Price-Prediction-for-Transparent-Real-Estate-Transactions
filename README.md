# Gurgaon House Price Prediction & Real Estate Transparency Analysis

**(A Data-Driven Approach to Transparent Real Estate Transactions)**

## Description

This project develops a machine learning model to predict the fair market value of properties in the Gurgaon real estate market. Its goal is to leverage data to bring greater transparency to real estate transactions, aiding homebuyers in making informed decisions and identifying potential overpricing.

The project cleans data, engineers new features, analyzes market segments, and trains and evaluates various regression models to find the optimal price prediction model. A simple application (`app.ipynb`) also demonstrates how the model can be used for basic predictions.

## Key Findings

* **Best Model:** Random Forest Regressor performs the best.
* **Accuracy:** The model achieves a high predictive accuracy (R² score) of **0.979**.
* **Prediction Error:** The average prediction error (MAE) is approximately **₹487,217**.
* **Market Transparency:** Analysis (on 500 properties) indicates largely fair pricing in the market (98%), but identifies some properties as significantly overpriced (>15%) or underpriced (approx. 0.6% overpriced). The market-wide average discrepancy is around -0.1%.
* **Key Price Factors:** Property price is most influenced by `Rate per sqft` (0.555 importance) and `Area` (0.395 importance), followed by `Size_Category`, `Property Type`, and `BHK_Final`.
* **RERA Impact:** RERA-approved properties have a higher average rate/sqft (₹18,670) compared to non-RERA (₹13,870), although the average price difference is minimal (₹40.36M vs ₹40.24M).

## Dataset

* **Source File:** `data of gurugram real Estate 2024.csv`
* **Initial Records:** 19515
* **Processed Records:** 19514
* **Features:** Price, Status, Area, Rate per sqft, Property Type, Locality, Builder Name, RERA Approval, BHK\_Count, Socity, Company Name, Flat Type.

## Methodology

1.  **Data Loading & Initial Exploration:** Loaded the dataset and performed a basic overview.
2.  **Data Preprocessing & Cleaning:** Cleaned numeric columns and handled missing values, especially in critical columns.
3.  **Feature Engineering:** Created new informative features such as `BHK_Final`, `Luxury_Property`, `Size_Category`, `Sector_Region`, and `RERA_Status`.
4.  **Statistical Analysis & EDA:** Calculated descriptive statistics, examined correlations, and generated visualizations (e.g., price distribution, area vs price, price by BHK, RERA impact).
5.  **Model Training & Evaluation:** Split data into training/testing sets. Trained multiple regression models (Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso) and evaluated them based on R² score, RMSE, and MAE. Random Forest was identified as the best model. Cross-validation was also performed.
6.  **Feature Importance Analysis:** Extracted feature importances from the best model (Random Forest).
7.  **Fair Market Value Analysis:** Used the best model to predict fair prices for sample properties and analyzed discrepancies with quoted prices.
8.  **Market-Wide Analysis:** Applied the model to a sample set to analyze overall overpricing/underpricing trends in the market.
9.  **Model Deployment:** Saved the best model (`gurgaon_property_predictor.pkl`) for potential application use.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (for LabelEncoder, StandardScaler, train_test_split, RandomForestRegressor, GradientBoostingRegressor, LinearRegression, Ridge, Lasso, metrics, cross_val_score)
* SciPy (for statistical tests)
* Joblib (for saving the model)

## Files in the Project

* `Gurgaon House Price Prediction for Transparent Real Estate Transactions.ipynb`: The main Jupyter notebook containing analysis, model training, and evaluation.
* `app.ipynb`: An example Jupyter notebook demonstrating a basic price prediction application. **Note:** This notebook currently trains a *simplified* Linear Regression model if the `gurgaon_property_predictor.pkl` file is missing or invalid. Modification might be needed to use the full Random Forest model.
* `data of gurugram real Estate 2024.csv`: The raw dataset used for the project.
* `gurgaon_property_predictor.pkl`: The saved, trained best model (Random Forest).
* `project_results_summary.txt`: A summary of the project's key findings and conclusions.

## Setup & Installation

1.  If this project is in a repository, clone it:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Ensure you have Python installed.
3.  Install the necessary libraries manually:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib jupyter
    ```
4.  Ensure the dataset file (`data of gurugram real Estate 2024.csv`) is present in the project directory.

## Usage

1.  **Full Analysis & Model Training:**
    * Open Jupyter Notebook or Jupyter Lab.
    * Run the `Gurgaon House Price Prediction for Transparent Real Estate Transactions.ipynb` notebook. This will execute all steps from data loading to model saving.

2.  **Basic Price Prediction Example:**
    * Open Jupyter Notebook or Jupyter Lab.
    * Run the `app.ipynb` notebook.
    * It will prompt you for Area, BHK, and Bathrooms and provide an estimated price (Note: This might use the simplified model as mentioned above).

3.  **Using the Trained Model (Advanced):**
    * Load the `gurgaon_property_predictor.pkl` file (e.g., using `joblib.load()`) in your custom Python scripts or applications to make predictions with the Random Forest model. Modifying `app.ipynb` could be a starting point.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a Pull Request with your changes.
