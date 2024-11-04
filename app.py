from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # STEP 1
    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)  # Random values between 0 and 1
    # Generate Y with normal additive error (mean mu, variance sigma^2)
    Y = np.random.normal(mu, np.sqrt(sigma2), N)  # Mean = mu, Std Dev = sqrt(sigma^2)

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)  # Reshape X for sklearn
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]  # Extract slope
    intercept = model.intercept_  # Extract intercept

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, color='blue', label='Data points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Regression line')
    plt.title(f"Scatter Plot with Regression Line\ny = {intercept:.2f} + {slope:.2f}x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations and create histograms of slopes and intercepts
    slopes = []  # List to store slopes
    intercepts = []  # List to store intercepts

    for _ in range(S):
        # Generate random X values with size N between 0 and 1
        X_sim = np.random.rand(N)
        # Generate Y values with normal additive error
        Y_sim = np.random.normal(mu, np.sqrt(sigma2), N)

        # Fit a linear regression model to X_sim and Y_sim
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)  # Reshape for sklearn
        sim_model.fit(X_sim_reshaped, Y_sim)

        # Append the slope and intercept of the model to the lists
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
