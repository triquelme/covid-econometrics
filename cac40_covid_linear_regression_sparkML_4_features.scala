package fr.m2i.exercice

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.regression.LinearRegression 
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler

object cac40_covid_linear_regression_sparkML {

  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val spark = SparkSession.builder.appName("LinearRegressionExample").master("local[*]").getOrCreate()
    
  def main(args: Array[String]) {
    
    // read csv from hdfs
    val data = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema", "true").csv("hdfs://localhost:9000/project/joined_cac40_covid_data.csv")
    
    data.printSchema()
/*
root
 |-- Date: string (nullable = true)
 |-- Adj Close: double (nullable = true)
 |-- hospitalized: integer (nullable = true)
 |-- in intensive care: integer (nullable = true)
 |-- returning home: integer (nullable = true)
 |-- deceased: integer (nullable = true)
*/
    data.show(5)
/*
+----------+-----------+------------+-----------------+--------------+--------+
|      Date|  Adj Close|hospitalized|in intensive care|returning home|deceased|
+----------+-----------+------------+-----------------+--------------+--------+
|2020-03-18|3754.840088|        2972|              771|           816|     218|
|2020-03-19|     3855.5|        4073|             1002|          1180|     327|
|2020-03-20|4048.800049|        5226|             1297|          1587|     450|
|2020-03-23|3914.310059|        8673|             2080|          2567|     860|
|2020-03-24|4242.700195|       10176|             2516|          3281|    1100|
+----------+-----------+------------+-----------------+--------------+--------+
only showing top 5 rows
*/
     
    val data2 = data.na.drop()
    data2.show(5)
    
    println("data : " + data.count + " data2 : " + data2.count)
/*
data : 39 data2 : 39
no drop because no null values
*/

    val assembler = new VectorAssembler().setInputCols(Array("hospitalized","in intensive care","returning home","deceased")).setOutputCol("features") 
    
    val output = assembler.transform(data2)
    
    // final data format must contain colnames "features" and "label"
    var final_data = output.select("features", "Adj Close")
    // we want to predict "cac40 Adjusted close price" based on covid "hospitalized","in intensive care","returning home","deceased" features
    
    final_data = final_data.withColumnRenamed("Adj Close", "label")
    final_data.printSchema()
/*
root
 |-- features: vector (nullable = true)
 |-- label: double (nullable = true)
*/

    final_data.show(5)
/*
+--------------------+-----------+
|            features|      label|
+--------------------+-----------+
|[2972.0,771.0,816...|3754.840088|
|[4073.0,1002.0,11...|     3855.5|
|[5226.0,1297.0,15...|4048.800049|
|[8673.0,2080.0,25...|3914.310059|
|[10176.0,2516.0,3...|4242.700195|
+--------------------+-----------+
*/
    
    // split data into 70% train data and 30% test data
    val training_test = final_data.randomSplit(Array(0.7, 0.3), seed = 11L)
    
    val training = training_test(0)
    val test = training_test(1)
    
    // create new LinearRegression Object
    val lr = new LinearRegression()
    println("Create new LinearRegression Object")
    
    // Fit the model   
    val lrModel = lr.fit(training)
    println("Fit the model")
    
    // Print the coefficients and intercept for linear regression
    println("Coefficients: " + lrModel.coefficients + " - Intercept: " + lrModel.intercept)
/*
Coefficients: [0.11010419573834197,-0.3195811440303151,0.03216765787107099,-0.1688582763130278] 
Intercept: 3823.2574726177213
*/
    
    // Evaluate model
    val test_results = lrModel.evaluate(test)
    
    test_results.residuals.show()
/*
+-------------------+
|          residuals|
+-------------------+
|  320.8055442771465|
| 390.35311951412314|
|  85.74409237690998|
|   85.1601314342679|
| 121.33062234698173|
| 124.12680036747588|
|-30.905273791047875|
| -90.87712015193756|
| 107.03892339099548|
| -58.37513110676082|
|-109.51890519126846|
| 140.68122679273802|
|  142.3032721251966|
|-126.58872822121793|
|  55.61900302599224|
+-------------------+
*/
    println("R2 : " + test_results.r2)
    // R2 : -2.362541572847185

    println("RMSE : " + test_results.rootMeanSquaredError) 
    //RMSE : 162.328352012628
  }
}