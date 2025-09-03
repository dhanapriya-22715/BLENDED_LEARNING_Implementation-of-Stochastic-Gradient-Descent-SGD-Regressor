# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Necessary Libraries**: 
   - Import required libraries such as `pandas`, `numpy`, `matplotlib`, `sklearn` for the implementation.

2. **Load the Dataset**: 
   - Load the dataset (e.g., `CarPrice_Assignment.csv`) using `pandas`.

3. **Data Preprocessing**: 
   - Drop unnecessary columns (e.g., 'CarName', 'car_ID').
   - Handle categorical variables using `pd.get_dummies()`.

4. **Split the Data**: 
   - Split the dataset into features (X) and target variable (Y).
   - Split the data into training and testing sets using `train_test_split()`.

5. **Standardize the Data**: 
   - Standardize the feature data (X) and target variable (Y) using `StandardScaler()` to ensure they have mean=0 and variance=1.

6. **Create the SGD Regressor Model**: 
   - Initialize the SGD Regressor model with `max_iter=1000` and `tol=1e-3`.

7. **Train the Model**: 
   - Fit the model to the training data using the `fit()` method.

8. **Make Predictions**: 
   - Use the trained model to predict the target values for the test set.

9. **Evaluate the Model**: 
   - Calculate performance metrics like Mean Squared Error (MSE) and R-squared score using `mean_squared_error()` and `r2_score()`.

10. **Display Model Coefficients**: 
    - Display the model's coefficients and intercept.

11. **Visualize the Results**: 
    - Create a scatter plot comparing actual vs predicted prices.

12. **End**: 
    - The program finishes by displaying the evaluation metrics, model coefficients, and a visual representation of the predictions.

## Program:
```

'''Program to implement SGD Regressor for linear regression.
Developed by: DHANAPPRIYA S
RegisterNumber:  212224230056 '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/CarPrice_Assignment (1).csv')
print(df.head())
print(df.info())


columns_to_drop = ['CarName', 'car_ID']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

if existing_columns_to_drop:
    df = df.drop(existing_columns_to_drop, axis=1)

df = pd.get_dummies(df, drop_first=True)

x = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
x = scaler.fit_transform(x)
y= scaler.fit_transform(np.array(y).reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(x_train, y_train)

y_pred = sgd_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```
```python


print("DHANAPPRIYA S")
print("212224230056")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

print("\nModel Coefficients:")
print("coefficient:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)


```
```python

print("DHANAPPRIYA S")
print("212224230056")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices using SGD Reggressor")
plt.plot([min(y_test ),max(y_test)],[min(y_test),max(y_test)],color='yellow')
plt.show()
```



## Output:

### LOAD THE DATASET:
<img width="1089" height="654" alt="Screenshot 2025-09-03 011415" src="https://github.com/user-attachments/assets/7988aa2a-e1ce-4296-a832-04bd9736b73e" />

<img width="777" height="742" alt="Screenshot 2025-09-03 011427" src="https://github.com/user-attachments/assets/b3a5ead5-91c3-4062-820e-bf14deb2c0bc" />

### EVALUATION METRICS AND MODEL COEFFICIENTS

<img width="869" height="344" alt="image" src="https://github.com/user-attachments/assets/21077056-3d29-4146-a092-800ce9442887" />


### VISUALIZATION OF ACTUAL VS PREDICTED VALUES:
<img width="1050" height="850" alt="image" src="https://github.com/user-attachments/assets/4667d3b3-3b60-4390-b0bd-5962e2c3c047" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
