📊 Linear Regression with Gradient Descent & Closed Form (Spark + Breeze)

This project demonstrates how to implement Linear Regression using both Gradient Descent and the Closed Form Solution in Apache Spark with Breeze for linear algebra operations. It’s built to run inside a Databricks notebook or any Spark environment with Scala support.

📦 Technologies Used

Apache Spark (RDD-based API)

Breeze (for vector/matrix operations and pseudo-inverse)

Scala

Databricks Notebook (optional)

🚀 Features

Computes gradient updates using computeSummand

Predicts values for labeled points using dot products

Calculates RMSE (Root Mean Squared Error) over predictions

Performs iterative gradient descent with decaying learning rate

Solves linear regression analytically via the closed form:

Compares performance between both methods

🛠️ Setup Instructions

Clone or copy this file into a Databricks notebook cell or a .scala file in your Spark project.

Make sure the following dependencies are available:

Spark 3.x+

Breeze (Databricks has it pre-installed)

Run the code block. It will:

Train using gradient descent

Train using the closed-form formula

Print intermediate outputs, RMSEs, and weights

📈 Sample Output

The code runs on a small example dataset:

Summand for (1.0,[2.0,3.0,1.0]): DenseVector(...)

Label: 1.0, Prediction: 2.3

RMSE: 0.82

Final Weights (GD): DenseVector(...)

Training Errors: [0.92, 0.85, ...]

Closed Form Weights: DenseVector(...)

🧠 What You’ll Learn

How to implement gradient descent manually using RDDs

How to compute predictions and losses with custom functions

The difference in performance between iterative and analytical solutions

How to use Breeze for matrix algebra in Spark applications

📂 Project Structure

computeSummand: Calculates gradient component per data point

predict: Generates label and prediction

computeRMSE: Calculates the RMSE of predictions

gradientDescent: Runs GD with learning rate decay

closedFormSolution: Computes weights using pseudo-inverse

main: Entry point to run all tests and methods

🔄 Future Improvements

Replace RDD with Spark DataFrame for scalability

Add support for regularization (L1/L2)

Visualize RMSE over iterations

Parallelize gradient steps more efficiently

🧪 Run with

LinearRegressionGradientDescent.main(Array.empty)

This will print logs, RMSE, and model weights for both approaches.

Here are some challenges I likely faced while building and running my Spark + Breeze Linear Regression project:

💡 Conceptual & Algorithmic Challenges 

Understanding Gradient Descent: Implementing the gradient descent algorithm from scratch (instead of using built-in MLlib methods) requires a solid grasp of loss minimization and vectorized updates.

Choosing Learning Rate (α): Deciding how fast the model should converge is tricky. A rate too high can cause divergence; too low can slow convergence.

RMSE Calculation: Ensuring correct error measurement across distributed data using RDD transformations adds an extra layer of complexity.

🧮 Technical & Programming Challenges

Using Breeze with Spark: Breeze and Spark are separate ecosystems. Using DenseVector from Breeze with Spark’s RDDs required careful type conversions and compatibility checks.

RDD Debugging: Debugging logic inside .map() and .reduce() functions is harder since print statements don’t always show up immediately or in order.

Matrix Operations for Closed Form: Constructing matrices from RDDs using DenseMatrix needed transformation and reshaping of raw feature arrays.

⚙️ Environment Challenges

Databricks/Spark Configuration: Running this inside a Databricks environment required you to configure Spark context correctly (setLogLevel, handling default sc reuse).

Dependency Management: Ensuring Breeze was available and didn’t conflict with MLlib or other dependencies.

Memory Efficiency: Storing entire datasets in memory for matrix inversion (closed form) can cause memory issues for large datasets.

🔍 Debugging & Testing Challenges

Interpreting Output: Matching computed weights and RMSE values with expectations to confirm correctness is non-trivial without a baseline.

Validating Gradient Updates: Making sure the summands computed during gradient descent actually push the weights in the correct direction.

Data Shape & Type Issues: RDD operations can fail silently if the structure of vectors or inputs is off by even a small amount.










