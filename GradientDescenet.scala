// Databricks notebook source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector, pinv, DenseMatrix}
object LinearRegressionGradientDescent extends Serializable {
def computeSummand(w: DenseVector[Double], lp: LabeledPoint): DenseVector[Double]
= {
val prediction = w.dot(new DenseVector(lp.features.toArray))
val error = prediction - lp.label
error * new DenseVector(lp.features.toArray)
}
def predict(w: DenseVector[Double], lp: LabeledPoint): (Double, Double) = {
val prediction = w.dot(new DenseVector(lp.features.toArray))
(lp.label, prediction)
}
def computeRMSE(predictions: RDD[(Double, Double)]): Double = {
val squaredErrors = predictions.map { case (label, prediction) =>
val error = label - prediction
error * error
}
math.sqrt(squaredErrors.mean())
}
def gradientDescent(trainData: RDD[LabeledPoint], numIterations: Int):
(DenseVector[Double], Array[Double]) = {
val numFeatures = trainData.first().features.size
var weights = DenseVector.zeros[Double](numFeatures)
var alpha = 1.0
val n = trainData.count()
var trainingErrors = Array[Double]()
for (i <- 1 to numIterations) {
val gradient = trainData.map(lp => computeSummand(weights, lp)).reduce(_ + _)
alpha = alpha / math.sqrt(i)
weights = weights - (gradient * alpha)
val predictions = trainData.map(lp => predict(weights, lp))
val rmse = computeRMSE(predictions)
trainingErrors = trainingErrors :+ rmse
}
(weights, trainingErrors)
}
def closedFormSolution(data: RDD[LabeledPoint]): DenseVector[Double] = {
val featuresMatrix = DenseMatrix(data.map(_.features.toArray).collect: _*)
val labelsVector = DenseVector(data.map(_.label).collect)
val weights = pinv(featuresMatrix.t * featuresMatrix) * (featuresMatrix.t *
labelsVector)
weights
}
def main(args: Array[String]): Unit = {
val sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")
// Sample data
val data: RDD[LabeledPoint] = sc.parallelize(Seq(
LabeledPoint(1.0, Vectors.dense(2.0, 3.0, 1.0)),
LabeledPoint(2.0, Vectors.dense(4.0, 1.0, 3.0)),
LabeledPoint(3.0, Vectors.dense(1.0, 2.0, 5.0))
))
// Test Part 1: computeSummand function on one example
val w = DenseVector(0.5, 0.2, 0.8)
val lp = data.first()
val summand = computeSummand(w, lp)
println(s"Summand for $lp: $summand")
// Test Part 2: predict function on one example LabeledPoint
val prediction = predict(w, lp)
println(s"Label: ${prediction._1}, Prediction: ${prediction._2}")
// Test Part 3: computeRMSE function on an example RDD of (label, prediction)
tuples
val predictions = data.map(lp => predict(w, lp))
val rmse = computeRMSE(predictions)
println(s"RMSE: $rmse")
// Test Part 4: Run gradient descent for 3 iterations
val numIterations = 5
val (weights, trainingErrors) = gradientDescent(data, numIterations)
println(s"Final Weights: $weights")
println(s"Training Errors: ${trainingErrors.mkString(", ")}")
// Test Part 5: Compute weights using closed form solution
val closedFormWeights = closedFormSolution(data)
println(s"Weights using Closed Form Solution: $closedFormWeights")
}
}
// Call the main method to execute the code
LinearRegressionGradientDescent.main(Array.empty)
