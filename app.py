from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('auto_mpg.csv')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()
df = df.drop(columns=['car name'])
X = df.drop(columns=['mpg'])
y = df['mpg']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return r2, rmse, mae, nse

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input from form
        cylinders = float(request.form["cylinders"])
        displacement = float(request.form["displacement"])
        horsepower = float(request.form["horsepower"])
        weight = float(request.form["weight"])
        acceleration = float(request.form["acceleration"])
        model_year = float(request.form["model_year"])
        origin = float(request.form["origin"])
        model_type = request.form["model"]
        
        # Prepare input for prediction
        input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
        
        # Select model
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "lasso":
            model = Lasso(alpha=0.1)  # Adjust alpha for Lasso regularization strength
        elif model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42, verbose=True)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        
        # Make prediction
        y_pred = model.predict(input_data)[0]
        
        # Calculate performance metrics
        r2, rmse, mae, nse = calculate_metrics(y_train, y_pred_train)
        
        # Render result without plot
        return render_template('result.html', y_pred=y_pred, r2=r2, rmse=rmse, mae=mae, nse=nse)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
