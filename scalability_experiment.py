import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from typing import List, Dict
import math
import sys
import json
import os
import time
SEED=42


######################################################################
######################## SPARK SESSION ###############################
######################################################################

class SparkSessionBuilder():
    def __init__(self, app_name="GabinoTest", logging_level="ERROR"):
        self.app_name = app_name
        self.session = SparkSession.builder.appName(self.app_name).getOrCreate()
        self.session.sparkContext.setLogLevel(logging_level)

    def get_session(self):
        return self.session

    def stop_session(self):
        self.session.stop()

######################################################################
######################## DATALOADER CLASSES  #########################
######################################################################

class BaseDataloader():
    def __init__(self, spark_session, file_path: str):
        self.spark = spark_session
        self.file_path = file_path
        self.numeric_cols = None
        self.categorical_cols = None
        self.target_col = None
    
    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def show_data(self, data, n=5):
        data.show(n)
        data.printSchema()
    
    def size_data(self, data):
        """
        Returns the number of bytes of the dataset on disk.
        """
        path = self.file_path
        if os.path.exists(path):
            return os.path.getsize(path)
        else:
            return 1
    
    def set_target_col(self, target_col):
        self.target_col = target_col
    
    def get_target_col(self):
        return self.target_col

class IntrusionDataloader(BaseDataloader):
    def __init__(self, spark_session, file_path="./intrusion_data/NF-ToN-IoT-v2-train-shuffled-half.csv"):
        super().__init__(spark_session, file_path)
        self.numeric_cols = None
        self.categorical_cols = None
    
    def load_data(self):
        data = self.spark.read.option("delimiter", ",")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .csv(self.file_path)
        data=data.drop("Label","IPV4_SRC_ADDR","IPV4_DST_ADDR","SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES")

        self.categorical_cols = ["Attack"]
        self.numeric_cols = [field.name for field in data.schema.fields if field.name not in self.categorical_cols]
        self.set_target_col("Attack")
        return data

class AdultDataLoader(BaseDataloader):
    def __init__(self, spark_session, file_path="./adult/adult.data"):
        super().__init__(spark_session, file_path)
        self.numeric_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        self.categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]

    def load_data(self):
        data = self.spark.read.option("delimiter", ", ")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .csv(self.file_path)
        self.set_target_col("income")
        return data

##############################################################################
######################## PREPROCESSOR CLASSES  ###############################
##############################################################################
    
class BasePreprocessor():
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_data(self, data):
        model = self.pipeline.fit(data)
        df_prepared = model.transform(data)
        return df_prepared

class IntrusionPreprocessor(BasePreprocessor):
    def _build_pipeline(self):
        indexers = [StringIndexer().setInputCol(col).setOutputCol(f"{col}_indexed") for col in self.categorical_cols]
        assembler = VectorAssembler().setInputCols(
            [f"{col}_indexed" for col in self.categorical_cols if col != "Attack"] + self.numeric_cols
        ).setOutputCol("features")
        pipeline = Pipeline().setStages(indexers + [assembler])
        return pipeline

class AdultDataPreprocessor(BasePreprocessor):
    def _build_pipeline(self):
        indexers = [StringIndexer().setInputCol(col).setOutputCol(f"{col}_indexed") for col in self.categorical_cols]
        assembler = VectorAssembler().setInputCols(
            [f"{col}_indexed" for col in self.categorical_cols if col != "income"] + self.numeric_cols
        ).setOutputCol("features")
        pipeline = Pipeline().setStages(indexers + [assembler])
        return pipeline
    
##############################################################################
######################## MODEL TRAINER CLASSE  ###############################
##############################################################################

class MRTreeModelTrainer():
    def __init__(self, seed=42):
        self.seed = seed

    def train_model(self, train_data, label_col="income_indexed"):
        classifier = DecisionTreeClassifier(impurity="entropy", maxBins=70)\
            .setLabelCol(label_col)\
            .setFeaturesCol("features")
        paramGrid = ParamGridBuilder()\
            .addGrid(classifier.maxDepth, [5, 10, 15, 20])\
            .addGrid(classifier.minInstancesPerNode, [100, 250, 500])\
            .build()
        evaluator = MulticlassClassificationEvaluator()\
            .setMetricName("accuracy")\
            .setPredictionCol("prediction")\
            .setLabelCol(label_col)
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

###########################################################################
######################## EXPERIMENT RUNNER CLASSES  #######################
###########################################################################

class AbstractExperimentRunner():
    def __init__(self, DataLoaderClass, DataPreprocessorClass, ModelTrainerClass, seed=42):
        self.seed = seed
        self.DataLoaderClass = DataLoaderClass
        self.DataPreprocessorClass = DataPreprocessorClass
        self.ModelTrainerClass = ModelTrainerClass
        self.metrics = {}
    
    def run(self, K: List[int] = [1, 2, 4, 8, 16, 32, 64]):
        """
        @param K: List of integers representing different scale factors to test.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def save_trace(self, filename):
        # Save the metrics to a JSON file to allow easy plotting later
        with open(filename, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_trace(self, filename):
        with open(filename, 'r') as f:
            self.metrics = json.load(f)
    
    def save_plot(self, filename=None):
        if filename is None:
            metrics_to_plot = self.metrics
        else:
            self.load_trace(filename)
            metrics_to_plot = self.metrics
        
        scales = [v["scale_factor"] for k, v in metrics_to_plot.items()]
        train_times = [v["training_time_seconds"] for k, v in metrics_to_plot.items()]
        dataset_sizes_gb = [v["dataset_size_bytes"] / (1024 * 1024 * 1024) for k, v in metrics_to_plot.items()]
        
        base_time = train_times[0]
        execution_time_sf = [t / base_time for t in train_times]
        
        print(f"Scales: {scales}\nDataset Sizes (GB): {dataset_sizes_gb}\nTraining Times (s): {train_times}\nExecution Time SF: {execution_time_sf}")
        
        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        ax1 = plt.gca()
        
        line1 = ax1.plot(dataset_sizes_gb, scales, 'r-^', label='Dataset size', linewidth=2, markersize=8)
        line2 = ax1.plot(dataset_sizes_gb, execution_time_sf, 'b-o', label='Execution time', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Dataset size (GB)', fontsize=12)
        ax1.set_ylabel('Scale factors (SF)', fontsize=12)
        
        plt.title('The scalability of MR-Tree algorithm', fontsize=14, pad=20)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True)

        plt.tight_layout()
        i = 0
        while os.path.exists(f"scalability_plot_{i}.png"):
            i += 1
        plt.savefig(f"scalability_plot_{i}.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Graphique sauvegardé dans 'scalability_plot_{i}.png'")

class LocalExperimentRunner(AbstractExperimentRunner):
    """
    Run the experiment while increasing the size of the dataset each time.
    at a scale of 1 for 500ko or 250ko based on cores available.
    """
    def run(self, K: List[int] = [1, 2, 4, 8, 16, 32, 64], full_dataset=False):
        cores = os.cpu_count() or 1
        bytes_per_unit = 250 if cores <= 2 else cores * 250
        spark_builder = SparkSessionBuilder()
        spark = spark_builder.get_session()

        data_loader = self.DataLoaderClass(spark)
        data = data_loader.load_data()
        
        file_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
        if not full_dataset:
            base_size_bytes = bytes_per_unit * 1024
            print(f"File size in bytes: {file_size_bytes}, base size in bytes: {base_size_bytes}")
            fraction = base_size_bytes / file_size_bytes

            base_subset_data = data.sample(fraction=fraction).cache()
        else:
            base_subset_data = data.cache()
            base_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
            print(f"Using full dataset as base. Size in bytes: {base_size_bytes}")
        
        subset_dataset = base_subset_data.cache()
        for i in range(K[0]-1):
            subset_dataset = subset_dataset.union(base_subset_data).cache() 

        current_scale = K[0]
        scale = None
        for i in range(len(K)):
            if i < len(K) -1:
                scale = K[i+1]
            print(f"\n{'='*50}")
            print(f"Training with scale factor: {current_scale}")
            print(f"{'='*50}\n")
            preprocessor = self.DataPreprocessorClass(data_loader.numeric_cols, data_loader.categorical_cols)
            df_prepared = preprocessor.preprocess_data(subset_dataset)
            train, test = df_prepared.randomSplit([0.7, 0.3], seed=self.seed)

            model_trainer = self.ModelTrainerClass(seed=self.seed)
            cv_model, evaluator, elapsed = model_trainer.train_model(train, label_col=f"{data_loader.get_target_col()}_indexed")
            print(f"Model trained in {elapsed} seconds")
            start_time = time.time()
            accuracy = evaluator.evaluate(cv_model.transform(test))
            end_time = time.time()
            elapsed_eval = end_time - start_time
            print(f"Test Accuracy = {accuracy}, Time taken = {elapsed_eval} seconds")

            self.metrics[i] = {
                "dataset_size_bytes": base_size_bytes * current_scale,
                "accuracy": accuracy,
                "scale_factor": current_scale,
                "training_time_seconds": elapsed,
                "evaluation_time_seconds": elapsed_eval,
            }
            if scale is None:
                break
            
            print(f"Dataset size increased FROM {subset_dataset.count()} rows.")
            nb_replications = scale - current_scale
            for _ in range(nb_replications):
                subset_dataset = subset_dataset.union(base_subset_data).cache()
            current_scale = scale
            print(f"______________________ TO.  {subset_dataset.count()} rows.")
            self.save_trace(f"local_experiment_trace_scale_{i+1}.json")
        print(f"Final metrics: {self.metrics}")
        spark_builder.stop_session()

class ClusterExperimentRunner(AbstractExperimentRunner):
    def run(self, number_of_nodes, K: List[int] = [1, 2, 4, 8, 16, 32, 64], full_dataset=False):
        """ 
        Exécute l'expérience sur un cluster en faisant varier la taille du jeu de données
        avec les mêmes facteurs d'échelle K que pour `LocalExperimentRunner`.
        La taille de base est ici fonction du nombre de nœuds du cluster.
        Si full_dataset est activé, on utilise le jeu de données complet comme base.
        """
        # Taille de base liée au cluster : 10 Mo par nœud
        bytes_per_unit = number_of_nodes * 1024 * 1024 * 10

        spark_builder = SparkSessionBuilder()
        spark = spark_builder.get_session()

        data_loader = self.DataLoaderClass(spark)
        data = data_loader.load_data()

        if not full_dataset:
            file_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
            base_size_bytes = bytes_per_unit
            print(f"File size in bytes: {file_size_bytes}, base size in bytes: {base_size_bytes}")
            if file_size_bytes < base_size_bytes:
                print(f"The requested base size ({base_size_bytes} bytes) is larger than the dataset size ({file_size_bytes} bytes). Using full dataset instead.")
                base_size_bytes = file_size_bytes
            # Fraction pour obtenir un sous-ensemble de taille ~ base_size_bytes
            fraction = base_size_bytes / file_size_bytes
            base_subset_data = data.sample(fraction=fraction).cache()

            # On commence avec K[0] fois le sous-ensemble de base
            subset_dataset = base_subset_data.cache()
            for _ in range(K[0] - 1):
                subset_dataset = subset_dataset.union(base_subset_data).cache()
        else:
            base_size_bytes = data_loader.size_data(data) if data_loader.size_data(data) > 0 else 1
            subset_dataset = data.cache()
            print(f"Using full dataset as base. Size in bytes: {base_size_bytes}")

        current_scale = K[0]
        scale = None
        for i in range(len(K)):
            if i < len(K) - 1:
                scale = K[i + 1]

            print(f"\n{'='*50}")
            print(f"Cluster training with scale factor: {current_scale} (nodes = {number_of_nodes})")
            print(f"{'='*50}\n")

            preprocessor = self.DataPreprocessorClass(data_loader.numeric_cols, data_loader.categorical_cols)
            df_prepared = preprocessor.preprocess_data(subset_dataset)
            train, test = df_prepared.randomSplit([0.7, 0.3], seed=self.seed)

            model_trainer = self.ModelTrainerClass(seed=self.seed)
            cv_model, evaluator, elapsed = model_trainer.train_model(train, label_col=f"{data_loader.get_target_col()}_indexed")
            print(f"Model trained in {elapsed} seconds")

            start_time = time.time()
            accuracy = evaluator.evaluate(cv_model.transform(test))
            end_time = time.time()
            elapsed_eval = end_time - start_time

            print(f"Test Accuracy = {accuracy}, Training time = {elapsed} seconds, Evaluation time = {elapsed_eval} seconds")

            self.metrics[i] = {
                "dataset_size_bytes": base_size_bytes * current_scale,
                "accuracy": accuracy,
                "scale_factor": current_scale,
                "training_time_seconds": elapsed,
                "evaluation_time_seconds": elapsed_eval,
                "number_of_nodes": number_of_nodes,
            }

            if scale is None:
                break

            print(f"Dataset size increased FROM {subset_dataset.count()} rows.")

            # Prépare le prochain facteur d'échelle (comme dans LocalExperimentRunner)
            nb_replications = scale - current_scale
            for _ in range(nb_replications):
                subset_dataset = subset_dataset.union(base_subset_data).cache()

            current_scale = scale
            print(f"______________________ TO.  {subset_dataset.count()} rows.")
            # Sauvegarde la trace après chaque étape pour analyse ultérieure
            # try to retreive job id from slurm to differentiate traces
            if 'SLURM_JOB_ID' in os.environ:
                job_id = os.environ['SLURM_JOB_ID']
                self.save_trace(f"cluster_experiment_trace_job_{job_id}_scale_{i+1}_node{number_of_nodes}.json")
            else:
                self.save_trace(f"cluster_experiment_trace_scale_{i+1}_node{number_of_nodes}.json")

        print(f"Final cluster metrics: {self.metrics}")
        spark_builder.stop_session()

if __name__ == "__main__":
    # runner = LocalExperimentRunner(AdultDataLoader, AdultDataPreprocessor, MRTreeModelTrainer, seed=SEED)
    # runner.run(K=[1, 8, 16, 32, 64])
    # runner.save_trace("local_experiment_trace.json")
    # runner.save_plot()
    runner = LocalExperimentRunner(IntrusionDataloader, IntrusionPreprocessor, MRTreeModelTrainer, seed=SEED)
    runner.run(K=[1])
    runner.save_trace("local_experiment_trace_intrusion.json")
    runner.save_plot()
