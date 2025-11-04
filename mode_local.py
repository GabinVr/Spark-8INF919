import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import math
import sys
import json
import os
import time
SEED=42


class SparkSessionBuilder():
    def __init__(self, app_name="GabinoTest", logging_level="ERROR"):
        self.app_name = app_name
        self.session = SparkSession.builder.appName(self.app_name).getOrCreate()
        self.session.sparkContext.setLogLevel(logging_level)

    def get_session(self):
        return self.session

    def stop_session(self):
        self.session.stop()
    

class AdultDataLoader():
    def __init__(self, spark_session, file_path="./adult/adult.data"):
        self.spark = spark_session
        self.file_path = file_path
        self.numeric_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        self.categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]


    def load_data(self):
        data = self.spark.read.option("delimiter", ", ")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .csv(self.file_path)
        return data
    
    def show_data(self, data, n=5):
        data.show(n)
        data.printSchema()
    
    def size_data(self, data):
        """
        Returns the number of bytes of the dataset.
        """
        return data.rdd.map(lambda row: len(str(row))).sum()
    
class AdultDataPreprocessor():
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        indexers = [StringIndexer().setInputCol(col).setOutputCol(f"{col}_indexed") for col in self.categorical_cols]
        assembler = VectorAssembler().setInputCols(
            [f"{col}_indexed" for col in self.categorical_cols if col != "income"] + self.numeric_cols
        ).setOutputCol("features")
        pipeline = Pipeline().setStages(indexers + [assembler])
        return pipeline
    
    def preprocess_data(self, data):
        model = self.pipeline.fit(data)
        df_prepared = model.transform(data)
        return df_prepared

class MRTreeModelTrainer():
    def __init__(self, seed=42):
        self.seed = seed

    def train_model(self, train_data):
        classifier = DecisionTreeClassifier(impurity="entropy", maxBins=70)\
            .setLabelCol("income_indexed")\
            .setFeaturesCol("features")
        paramGrid = ParamGridBuilder()\
            .addGrid(classifier.maxDepth, [5, 10, 15, 20])\
            .addGrid(classifier.minInstancesPerNode, [100, 250, 500])\
            .build()
        evaluator = MulticlassClassificationEvaluator()\
            .setMetricName("accuracy")\
            .setPredictionCol("prediction")\
            .setLabelCol("income_indexed")
        crossval = CrossValidator()\
            .setEstimatorParamMaps(paramGrid)\
            .setNumFolds(5)\
            .setEstimator(classifier)\
            .setEvaluator(evaluator)
        t0 = time.time()
        cvModel = crossval.fit(train_data)
        t1 = time.time()
        print(f"Classifier trained in {t1 - t0} seconds")
        return cvModel, evaluator, (t1 - t0)


class AbstractExperimentRunner():
    def __init__(self, DataLoaderClass, DataPreprocessorClass, ModelTrainerClass, seed=42):
        self.seed = seed
        self.DataLoaderClass = DataLoaderClass
        self.DataPreprocessorClass = DataPreprocessorClass
        self.ModelTrainerClass = ModelTrainerClass
        self.metrics = {}
    
    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_trace(self, filename):
        # Save the metrics to a JSON file to allow easy plotting later
        with open(filename, 'w') as f:
            json.dump(self.metrics, f)

class LocalExperimentRunner(AbstractExperimentRunner):
    """
    Run the experiment while increasing the size of the dataset each time.
    at a scale of 1 for 500ko or 250ko based on cores available.
    """
    def run(self):
        cores = os.cpu_count() or 1
        bytes_per_unit = 250 if cores <= 2 else 500   # 250Ko / 500Ko per step
        spark_builder = SparkSessionBuilder()
        spark = spark_builder.get_session()

        data_loader = self.DataLoaderClass(spark)
        data = data_loader.load_data()
        
        file_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
        fraction = bytes_per_unit * 1024 / file_size_bytes

        # Base dataset to duplicate
        base_subset_data = data.sample(fraction=fraction).cache()
        base_size_bytes = bytes_per_unit * 1024
        base_subset_count = base_subset_data.count()


        for i in range(1, 101):

            # preprocess + train
            preprocessor = self.DataPreprocessorClass(data_loader.numeric_cols, data_loader.categorical_cols)
            df_prepared = preprocessor.preprocess_data(base_subset_data)
            train, test = df_prepared.randomSplit([0.7, 0.3], seed=self.seed)

            model_trainer = self.ModelTrainerClass(seed=self.seed)
            cv_model, evaluator, elapsed = model_trainer.train_model(train)
            accuracy = evaluator.evaluate(cv_model.transform(test))
            print(f"Test Accuracy = {accuracy}, Time taken = {elapsed} seconds")

            self.metrics[i] = {
                "dataset_size_bytes": base_size_bytes * i,
                "accuracy": accuracy,
                "scale_factor": bytes_per_unit * 1024 / file_size_bytes,
                "training_time_seconds": elapsed,
            }

            print(f"Dataset size increased FROM {base_subset_data.count()} rows.")
            base_subset_data = base_subset_data.union(data_loader.load_data().sample(fraction=fraction)).cache()
            print(f"______________________ TO.  {base_subset_data.count()} rows.")
            base_subset_count = base_subset_data.count()
            
        spark_builder.stop_session()

class ClusterExperimentRunner(AbstractExperimentRunner):
    def run(self, number_of_nodes):
        bytes_per_unit = number_of_nodes * 1024 * 1024 * 10  # 10MB per node
        spark_builder = SparkSessionBuilder()
        spark = spark_builder.get_session()
        data_loader = self.DataLoaderClass(spark)
        data = data_loader.load_data()
        file_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
        fraction = bytes_per_unit / file_size_bytes
        base_subset_data = data.sample(fraction=fraction).cache()
        base_size_bytes = bytes_per_unit
        base_subset_count = base_subset_data.count()
        for i in range(1, 101):
            preprocessor = self.DataPreprocessorClass(data_loader.numeric_cols, data_loader.categorical_cols)
            df_prepared = preprocessor.preprocess_data(base_subset_data)
            train, test = df_prepared.randomSplit([0.7, 0.3], seed=self.seed)

            model_trainer = self.ModelTrainerClass(seed=self.seed)
            cv_model, evaluator, elapsed = model_trainer.train_model(train)
            accuracy = evaluator.evaluate(cv_model.transform(test))
            print(f"Test Accuracy = {accuracy}, Time taken = {elapsed} seconds")

            self.metrics[i] = {
                "dataset_size_bytes": base_size_bytes * i,
                "accuracy": accuracy,
                "scale_factor": bytes_per_unit / file_size_bytes,
                "training_time_seconds": elapsed,
            }

            print(f"Dataset size increased FROM {base_subset_data.count()} rows.")
            base_subset_data = base_subset_data.union(data_loader.load_data().sample(fraction=fraction)).cache()
            print(f"______________________ TO.  {base_subset_data.count()} rows.")
            base_subset_count = base_subset_data.count()
        spark_builder.stop_session()

if __name__ == "__main__":
    runner = LocalExperimentRunner(AdultDataLoader, AdultDataPreprocessor, MRTreeModelTrainer, seed=SEED)
    runner.run()
    runner.save_trace("local_experiment_trace.json")