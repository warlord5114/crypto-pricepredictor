 🪙 Crypto Price Predictor

This project is built to **predict the future price of cryptocurrencies**, like Bitcoin, using **machine learning**. It uses a **simple Linear Regression model** to learn from past price trends and make future predictions.

## 🔍 What is this project about?

Cryptocurrencies are highly volatile, and predicting their prices is both challenging and valuable. This project takes historical price data (stored in `crypto.csv`) and tries to understand the **pattern of how prices have moved over time**.

By learning from past trends, the model attempts to forecast what the price might be in the near future.


## 🧰 What does the project do?

Here's a step-by-step breakdown:

1. **Loads Historical Data:**

   * It uses a CSV file that contains historical price data (like Open, High, Low, Close prices for Bitcoin).
   * Libraries like `pandas` help in reading and processing this data.

2. **Cleans and Prepares the Data:**

   * Missing values are filled.
   * New features are created (like lagged values or percentage changes).
   * The target variable is set as the price we want to predict.

3. **Splits the Data:**

   * The dataset is split into a **training set** (to train the model) and a **test set** (to evaluate how well the model performs).

4. **Builds a Prediction Model:**

   * A **Linear Regression model** is trained using `scikit-learn`.
   * It learns the relationship between the current and past prices.

5. **Predicts Future Prices:**

   * Once trained, the model makes predictions on the test data.
   * These predicted prices are compared to the actual prices to check accuracy.

6. **Visualizes the Results:**

   * The predicted and actual prices are plotted using `matplotlib` to easily see how close the predictions are.

7. **Saves the Model:**

   * The trained model is saved as a `.pkl` file using `joblib` so it can be reused later without retraining.

## 🧠 Why Linear Regression?

Linear Regression is a good starting point for any prediction problem. It’s simple, interpretable, and gives a baseline performance. While not perfect for complex crypto trends, it helps you:

* Understand model-building steps
* Learn how to deal with time-series data
* Visualize predictions

## 📈 What can be improved?

This project uses a basic model. You can take it further by:

* Using more advanced models like LSTM (good for time-series)
* Adding more features like trading volume, news sentiment, etc.
* Deploying it as a real-time prediction app

## ✅ In Summary

* Input: Historical crypto prices (`crypto.csv`)
* Model: Linear Regression
* Output: Predicted prices for future days
* Tools: Python, pandas, scikit-learn, matplotlib, joblib
