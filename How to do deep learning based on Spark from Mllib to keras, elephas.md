# How to do deep learning based on Spark: from Mllib to keras, elephas

Published: 2019-04-17

# Spark ML model pipelines on Distributed Deep Neural Nets

This notebook describes how to build machine learning pipelines with Spark ML for distributed versions of Keras deep learning models. As data set we use the Otto Product Classification challenge from Kaggle. The reason we chose this data is that it is small and very structured. This way, we can focus more on technical components rather than prepcrocessing intricacies. Also, users with slow hardware or without a full-blown Spark cluster should be able to run this example locally, and still learn a lot about the distributed mode.

Often, the need to distribute computation is not imposed by model training, but rather by building the data pipeline, i.e. ingestion, transformation etc. In training, deep neural networks tend to do fairly well on one or more GPUs on one machine. Most of the time, using gradient descent methods, you will process one batch after another anyway. Even so, it may still be beneficial to use frameworks like Spark to integrate your models with your surrounding infrastructure. On top of that, the convenience provided by Spark ML pipelines can be very valuable (being syntactically very close to what you might know from scikit-learn).

TL;DR: We will show how to tackle a classification problem using distributed deep neural nets and Spark ML pipelines in an example that is essentially a distributed version of the one found here.

## Using this notebook

As we are going to use elephas, you will need access to a running Spark context to run this notebook. If you don’t have it already, install Spark locally by following the instructions provided here. Make sure to also export SPARK_HOME to your path and start your ipython/jupyter notebook as follows:

```
IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --driver-memory 4G elephas/examples/Spark_ML_Pipeline.ipynb
```

To test your environment, try to print the Spark context (provided as sc), i.e. execute the following cell.

```
from __future__ import print_function
print(sc)
<pyspark.context.SparkContext object at 0x1132d61d0>
```

## Otto Product Classification Data

Training and test data is available here. Go ahead and download the data. Inspecting it, you will see that the provided csv files consist of an id column, 93 integer feature columns. train.csv has an additional column for labels, which test.csv is missing. The challenge is to accurately predict test labels. For the rest of this notebook, we will assume data is stored at data_path, which you should modify below as needed.

```
data_path = "./" # <-- Make sure to adapt this to where your csv files are.
```

Loading data is relatively simple, but we have to take care of a few things. First, while you can shuffle rows of an RDD, it is generally not very efficient. But since data in train.csv is sorted by category, we’ll have to shuffle in order to make the model perform well. This is what the function shuffle_csv below is for. Next, we read in plain text in load_data_rdd, split lines by comma and convert features to float vector type. Also, note that the last column in train.csv represents the category, which has a Class_ prefix.

### Defining Data Frames

Spark has a few core data structures, among them is the data frame, which is a distributed version of the named columnar data structure many will now from either R or Pandas. We need a so called SQLContext and an optional column-to-names mapping to create a data frame from scratch.

```python
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
import numpy as np
import random

sql_context = SQLContext(sc)

def shuffle_csv(csv_file):
    lines = open(csv_file).readlines()
    random.shuffle(lines)
    open(csv_file, 'w').writelines(lines)

def load_data_frame(csv_file, shuffle=True, train=True):
    if shuffle:
        shuffle_csv(csv_file)
    data = sc.textFile(data_path + csv_file) # This is an RDD, which will later be transformed to a data frame
    data = data.filter(lambda x:x.split(',')[0] != 'id').map(lambda line: line.split(','))
    if train:
        data = data.map(
            lambda line: (Vectors.dense(np.asarray(line[1:-1]).astype(np.float32)),
                          str(line[-1])) )
    else:
        # Test data gets dummy labels. We need the same structure as in Train data
        data = data.map( lambda line: (Vectors.dense(np.asarray(line[1:]).astype(np.float32)),"Class_1") ) 
    return sqlContext.createDataFrame(data, ['features', 'category'])
```

Let’s load both train and test data and print a few rows of data using the convenient show method.

```python
train_df = load_data_frame("train.csv")
test_df = load_data_frame("test.csv", shuffle=False, train=False) # No need to shuffle test data

print("Train data frame:")
train_df.show(10)

print("Test data frame (note the dummy category):")
test_df.show(10)
Train data frame:
+--------------------+--------+
|            features|category|
+--------------------+--------+
|[0.0,0.0,0.0,0.0,...| Class_8|
|[0.0,0.0,0.0,0.0,...| Class_8|
|[0.0,0.0,0.0,0.0,...| Class_2|
|[0.0,1.0,0.0,1.0,...| Class_6|
|[0.0,0.0,0.0,0.0,...| Class_9|
|[0.0,0.0,0.0,0.0,...| Class_2|
|[0.0,0.0,0.0,0.0,...| Class_2|
|[0.0,0.0,0.0,0.0,...| Class_3|
|[0.0,0.0,4.0,0.0,...| Class_8|
|[0.0,0.0,0.0,0.0,...| Class_7|
+--------------------+--------+
only showing top 10 rows

Test data frame (note the dummy category):
+--------------------+--------+
|            features|category|
+--------------------+--------+
|[1.0,0.0,0.0,1.0,...| Class_1|
|[0.0,1.0,13.0,1.0...| Class_1|
|[0.0,0.0,1.0,1.0,...| Class_1|
|[0.0,0.0,0.0,0.0,...| Class_1|
|[2.0,0.0,5.0,1.0,...| Class_1|
|[0.0,0.0,0.0,0.0,...| Class_1|
|[0.0,0.0,0.0,0.0,...| Class_1|
|[0.0,0.0,0.0,1.0,...| Class_1|
|[0.0,0.0,0.0,0.0,...| Class_1|
|[0.0,0.0,0.0,0.0,...| Class_1|
+--------------------+--------+
only showing top 10 rows
```

## Preprocessing: Defining Transformers

Up until now, we basically just read in raw data. Luckily, Spark ML has quite a few preprocessing features available, so the only thing we will ever have to do is define transformations of data frames.

To proceed, we will first transform category strings to double values. This is done by a so called StringIndexer. Note that we carry out the actual transformation here already, but that is just for demonstration purposes. All we really need is too define string_indexer to put it into a pipeline later on.

```
from pyspark.ml.feature import StringIndexer

string_indexer = StringIndexer(inputCol="category", outputCol="index_category")
fitted_indexer = string_indexer.fit(train_df)
indexed_df = fitted_indexer.transform(train_df)
```

Next, it’s good practice to normalize the features, which is done with a StandardScaler.

```
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
fitted_scaler = scaler.fit(indexed_df)
scaled_df = fitted_scaler.transform(indexed_df)
print("The result of indexing and scaling. Each transformation adds new columns to the data frame:")
scaled_df.show(10)
The result of indexing and scaling. Each transformation adds new columns to the data frame:
+--------------------+--------+--------------+--------------------+
|            features|category|index_category|     scaled_features|
+--------------------+--------+--------------+--------------------+
|[0.0,0.0,0.0,0.0,...| Class_8|           2.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_8|           2.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_2|           0.0|[-0.2535060296260...|
|[0.0,1.0,0.0,1.0,...| Class_6|           1.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_9|           4.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_2|           0.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_2|           0.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_3|           3.0|[-0.2535060296260...|
|[0.0,0.0,4.0,0.0,...| Class_8|           2.0|[-0.2535060296260...|
|[0.0,0.0,0.0,0.0,...| Class_7|           5.0|[-0.2535060296260...|
+--------------------+--------+--------------+--------------------+
only showing top 10 rows
```

## Keras Deep Learning model

Now that we have a data frame with processed features and labels, let’s define a deep neural net that we can use to address the classification problem. Chances are you came here because you know a thing or two about deep learning. If so, the model below will look very straightforward to you. We build a keras model by choosing a set of three consecutive Dense layers with dropout and ReLU activations. There are certainly much better architectures for the problem out there, but we really just want to demonstrate the general flow here.

```
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, generic_utils

nb_classes = train_df.select("category").distinct().count()
input_dim = len(train_df.select("features").first()[0])

model = Sequential()
model.add(Dense(512, input_shape=(input_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
```

## Distributed Elephas model

To lift the above Keras model to Spark, we define an Estimator on top of it. An Estimator is Spark’s incarnation of a model that still has to be trained. It essentially only comes with only a single (required) method, namely fit. Once we call fit on a data frame, we get back a Model, which is a trained model with a transform method to predict labels.

We do this by initializing an ElephasEstimator and setting a few properties. As by now our input data frame will have many columns, we have to tell the model where to find features and labels by column name. Then we provide serialized versions of Keras model and Elephas optimizer. We can not plug in keras models into the Estimator directly, as Spark will have to serialize them anyway for communication with workers, so it’s better to provide the serialization ourselves. In fact, while pyspark knows how to serialize model, it is extremely inefficient and can break if models become too large. Spark ML is especially picky (and rightly so) about parameters and more or less prohibits you from providing non-atomic types and arrays of the latter. Most of the remaining parameters are optional and rather self explainatory. Plus, many of them you know if you have ever run a keras model before. We just include them here to show the full set of training configuration.

```
from elephas.ml_model import ElephasEstimator
from elephas import optimizers as elephas_optimizers

# Define elephas optimizer (which tells the model how to aggregate updates on the Spark master)
adadelta = elephas_optimizers.Adadelta()

# Initialize SparkML Estimator and set all relevant properties
estimator = ElephasEstimator()
estimator.setFeaturesCol("scaled_features")             # These two come directly from pyspark,
estimator.setLabelCol("index_category")                 # hence the camel case. Sorry :)
estimator.set_keras_model_config(model.to_yaml())       # Provide serialized Keras model
estimator.set_optimizer_config(adadelta.get_config())   # Provide serialized Elephas optimizer
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_num_workers(1)  # We just use one worker here. Feel free to adapt it.
estimator.set_nb_epoch(20) 
estimator.set_batch_size(128)
estimator.set_verbosity(1)
estimator.set_validation_split(0.15)
ElephasEstimator_415398ab22cb1699f794
```

## SparkML Pipelines

Now for the easy part: Defining pipelines is really as easy as listing pipeline stages. We can provide any configuration of Transformers and Estimators really, but here we simply take the three components defined earlier. Note that string_indexer and scaler and interchangable, while estimator somewhat obviously has to come last in the pipeline.

```
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[string_indexer, scaler, estimator])
```

### Fitting and evaluating the pipeline

The last step now is to fit the pipeline on training data and evaluate it. We evaluate, i.e. transform, on training data, since only in that case do we have labels to check accuracy of the model. If you like, you could transform the test_df as well.

```
from pyspark.mllib.evaluation import MulticlassMetrics

fitted_pipeline = pipeline.fit(train_df) # Fit model to data

prediction = fitted_pipeline.transform(train_df) # Evaluate on train data.
# prediction = fitted_pipeline.transform(test_df) # <-- The same code evaluates test data.
pnl = prediction.select("index_category", "prediction")
pnl.show(100)

prediction_and_label = pnl.map(lambda row: (row.index_category, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print(metrics.precision())
61878/61878 [==============================] - 0s     
+--------------+----------+
|index_category|prediction|
+--------------+----------+
|           2.0|       2.0|
|           2.0|       2.0|
|           0.0|       0.0|
|           1.0|       1.0|
|           4.0|       4.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           3.0|       3.0|
|           2.0|       2.0|
|           5.0|       0.0|
|           0.0|       0.0|
|           4.0|       4.0|
|           0.0|       0.0|
|           4.0|       1.0|
|           2.0|       2.0|
|           1.0|       1.0|
|           0.0|       0.0|
|           6.0|       0.0|
|           2.0|       2.0|
|           1.0|       1.0|
|           2.0|       2.0|
|           8.0|       8.0|
|           1.0|       1.0|
|           5.0|       0.0|
|           0.0|       0.0|
|           0.0|       3.0|
|           0.0|       0.0|
|           1.0|       1.0|
|           4.0|       4.0|
|           2.0|       2.0|
|           0.0|       3.0|
|           3.0|       3.0|
|           0.0|       0.0|
|           3.0|       0.0|
|           1.0|       5.0|
|           3.0|       3.0|
|           2.0|       2.0|
|           1.0|       1.0|
|           0.0|       0.0|
|           2.0|       2.0|
|           2.0|       2.0|
|           1.0|       1.0|
|           6.0|       6.0|
|           1.0|       1.0|
|           0.0|       3.0|
|           7.0|       0.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           1.0|       1.0|
|           1.0|       1.0|
|           6.0|       6.0|
|           0.0|       0.0|
|           0.0|       3.0|
|           2.0|       2.0|
|           0.0|       0.0|
|           2.0|       2.0|
|           0.0|       0.0|
|           4.0|       4.0|
|           0.0|       0.0|
|           6.0|       6.0|
|           2.0|       5.0|
|           0.0|       3.0|
|           3.0|       0.0|
|           0.0|       0.0|
|           3.0|       3.0|
|           4.0|       4.0|
|           0.0|       3.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           4.0|       4.0|
|           3.0|       0.0|
|           2.0|       2.0|
|           1.0|       1.0|
|           7.0|       7.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           0.0|       3.0|
|           1.0|       1.0|
|           1.0|       1.0|
|           5.0|       4.0|
|           1.0|       1.0|
|           1.0|       1.0|
|           4.0|       4.0|
|           3.0|       3.0|
|           0.0|       0.0|
|           2.0|       2.0|
|           4.0|       4.0|
|           7.0|       7.0|
|           2.0|       2.0|
|           0.0|       0.0|
|           1.0|       1.0|
|           0.0|       0.0|
|           4.0|       4.0|
|           1.0|       1.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           0.0|       0.0|
|           0.0|       3.0|
|           0.0|       3.0|
|           0.0|       0.0|
+--------------+----------+
only showing top 100 rows

0.764132648114
```

## Conclusion

It may certainly take some time to master the principles and syntax of both Keras and Spark, depending where you come from, of course. However, we also hope you come to the conclusion that once you get beyond the stage of struggeling with defining your models and preprocessing your data, the business of building and using SparkML pipelines is quite an elegant and useful one.

If you like what you see, consider helping further improve elephas or contributing to Keras or Spark. Do you have any constructive remarks on this notebook? Is there something you want me to clarify?In any case, feel free to contact me.