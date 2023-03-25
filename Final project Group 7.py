# Databricks notebook source
# DBTITLE 1,Alcohol-Binary Classification
# MAGIC %md
# MAGIC Alcohol dataset contains 34000 records. Each records represent a high school student's data. The objective of this data is to predict underage drinking by these high school students, meaning utilizing the date of this dataset we will predict whether the student will drink alcohol or not (this is the ‘alc’ column: 1=Yes, 0=No). This is an important prediction task to detect underage drinking and deploy intervention techniques.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/alcohol.csv"
file_type = "csv"

#the applied options are for CSV files. For other files types, these will be ignored.
df=spark.read.csv(file_location, header =True,inferSchema=True)

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.pandas as ps
import matplotlib.pyplot as mpl
from pyspark.sql import SparkSession
from pyspark.pandas import DataFrame
from pyspark.pandas import Series
import seaborn as sns


# COMMAND ----------

#Check dimension's 
print((df.count(),len(df.columns)))

# COMMAND ----------

#This is to view the data of data 
df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

df.dtypes

# COMMAND ----------

#Import the f function from pyspark
import pyspark.sql.functions as f
my_data = df


# COMMAND ----------

# visualization of some of the variables to see if there is high correlation
jp = my_data.toPandas()
sns.set_style('whitegrid')
sns.pairplot(jp,  x_vars=["alc", "age", "famrel", "studytime", "health"],
                y_vars=["alc", "age", "famrel", "studytime", "health"],
                 hue='age', palette='Greens', kind='reg', corner=True)
# famrel = Quality of family relationships (1: very bad … 5: excellent)
# studytime = Weekly study time (1: <2 hours, 2: 2 to 5 hours, 3: 5 to 10 hours, or 4: >10 hours)
# health = Current health status (1: very bad … 5: very good)

# COMMAND ----------

#null values in each column
data_agg = my_data.agg(*[f.count(f.when(f.isnull(c),c)).alias(c) for c in my_data.columns])
data_agg.show()

# COMMAND ----------

my_data.groupBy('alc').count().show()
print()

# COMMAND ----------

#'alc' is weekend alcohol consumption (1: Yes, 0: No)
sns.countplot(data=jp, x='alc')

# COMMAND ----------

my_data.groupBy('gender').count().show()
print()

# COMMAND ----------

j=my_data
jp2=j.toPandas()
sns.boxplot(data=jp2, hue='gender', x='alc', y='age')
#'alc' is weekend alcohol consumption (1: Yes, 0: No)

# COMMAND ----------

#Preprocessing steps
from pyspark.ml.feature import StringIndexer, OneHotEncoder
#Create object of StringIndexer class and specify input and output column
SI_gender = StringIndexer(inputCol='gender',outputCol = 'gender_Index')

# COMMAND ----------

#transform the data
my_data = SI_gender.fit(my_data).transform(my_data)

# COMMAND ----------

my_data.groupBy('gender_Index').count().show()
print()

# COMMAND ----------

my_data.show()

# COMMAND ----------

#Creating the feature columns
from pyspark.ml.feature import VectorAssembler

#specify the input and output columns of the vector assembler

assembler = VectorAssembler(inputCols=['age',
                                       'Medu',
                                       'Fedu',
                                       'traveltime',
                                       'studytime',
                                       'failures',
                                       'famrel',
                                       'freetime',
                                       'goout',
                                       'health',
                                       'absences',
                                       'gender_Index'],
                           outputCol='features')
#fill the null values
my_data = my_data.fillna(0)

#transform the data
final_data = assembler.transform(my_data)

# COMMAND ----------

#view the transformed vector
final_data.select('features','alc').show()

# COMMAND ----------

model_df =final_data.select(['features','alc'])
model_df = model_df.withColumnRenamed('alc','label')
model_df.printSchema()

# COMMAND ----------

#Split into training & testing Dataframe
training_df, test_df = model_df.randomSplit([0.7,0.30], seed = 21)

# COMMAND ----------

# MAGIC %md
# MAGIC #Logistic Regression

# COMMAND ----------

#Create a logistic regression model object
from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression().fit(training_df)

# COMMAND ----------

lr_summary=log_reg.summary

# COMMAND ----------

#Overall accuracy of the classification model
lr_summary.accuracy

# COMMAND ----------

#Area under ROC
lr_summary.areaUnderROC

# COMMAND ----------

#Precision of both classes
print(lr_summary.precisionByLabel)

# COMMAND ----------

print(lr_summary.recallByLabel)

# COMMAND ----------

#Get predictions
predictions = log_reg.transform(test_df)

# COMMAND ----------

predictions.select('label','prediction').show(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Confusion Matrix 

# COMMAND ----------

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

# MAGIC %md
# MAGIC #Unsupervised Learning K-means Clustering

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

my_data2 = final_data.select('features','alc')

# COMMAND ----------

kmeans = KMeans().setK(2).setSeed(21)
model = kmeans.fit(my_data2)

# COMMAND ----------

predictions5 = model.transform(my_data2)

# COMMAND ----------

evaluator = ClusteringEvaluator()

# COMMAND ----------

silhouette = evaluator.evaluate(predictions5)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
print("=========================================================================")
for center in centers:
    print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Tree Classifier

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

#Spliting 
train_data, test_data = my_data.randomSplit([0.7, 0.3], seed=21)

# COMMAND ----------

assembler2 = VectorAssembler(inputCols=['age', 'Medu', 'Fedu', 'traveltime','studytime','failures','famrel','freetime','goout','health','absences','gender_Index'], outputCol="features")

# COMMAND ----------

train_data = assembler2.transform(train_data)
test_data = assembler2.transform(test_data)

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="alc", featuresCol="features")
model = dt.fit(train_data)
predictions = model.transform(test_data)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="alc")
auc = evaluator.evaluate(predictions)
print("Area under ROC curve = " + str(auc))
