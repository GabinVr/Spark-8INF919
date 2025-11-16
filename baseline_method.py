# Author Gabin VRILLAULT

from scalability_experiment import IntrusionDataloader, IntrusionPreprocessor, ClusterExperimentRunner, MRTreeModelTrainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import json
import argparse
parser = argparse.ArgumentParser(description='Run baseline experiment with Decision Tree Classifier')
parser.add_argument('-n', '--nodes', type=int,
                    help='Number of nodes to test')
args = parser.parse_args()

class DecisionTreeBaselineMethod:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self, train_df: pd.DataFrame, feature_cols: list, label_col: str):
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        self.model.fit(X_train, y_train)

    def evaluate(self, test_df: pd.DataFrame, feature_cols: list, label_col: str) -> float:
        """
        Cross-validation evaluation of the model on the test dataset.
        """
        # X_test = test_df[feature_cols]
        # y_test = test_df[label_col]
        # scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy').mean()
        # print(f"Test scores: {scores}")
        # print(f"Mean : {scores.mean()}")    
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]
        param_grid = {
            "max_depth": [3, 5, 10, 20],
            "min_samples_split": [100, 255, 500],
            "criterion": ["entropy"]
        }


        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring="accuracy"
        )
        time_start = time.time()
        grid.fit(X_test, y_test)
        time_end = time.time()
        print(f"Grid search time: {time_end - time_start:.2f} seconds")
        print(f"Best parameters: {grid.best_params_}")
        best_model = grid.best_estimator_
        scores = best_model.score(X_test, y_test)

        return scores
    
class IntrusionDataset:
    def __init__(self, file_path: str="./intrusion_data/NF-ToN-IoT-v2-train-shuffled.csv"):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
    def get_data(self) -> pd.DataFrame:
        return self.df


def experiment():
    metrics = {}
    print("Expérience Descision Tree avec Scikit-Learn VS MRTree de Spark MLlib")
    print(f"Expérience avec {args.nodes} nœuds.")
    dataset = IntrusionDataset()
    df = dataset.get_data()
    print("Données chargées.")
    le_ip_src = LabelEncoder()
    le_ip_dst = LabelEncoder()
    le_attack = LabelEncoder()
    categorical_cols = ["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    feature_columns = [col for col in df.columns if col not in categorical_cols]
    df["IPV4_SRC_ADDR"] = le_ip_src.fit_transform(df["IPV4_SRC_ADDR"])
    df["IPV4_DST_ADDR"] = le_ip_dst.fit_transform(df["IPV4_DST_ADDR"])
    df["Attack"] = le_attack.fit_transform(df["Attack"])
    label_column = 'Attack'
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    baseline_method = DecisionTreeBaselineMethod()
    print("Entraînement du modèle...")
    start_time = time.time()
    baseline_method.train(train_df, feature_columns, label_column)
    training_time = time.time() - start_time
    metrics['sklearn'] = {}
    metrics['sklearn']['training_time_seconds'] = training_time
    print(f"Modèle Sklearn entraîné en {training_time:.2f} secondes.")
    print("Évaluation du modèle...")
    start_time = time.time()
    accuracy = baseline_method.evaluate(test_df, feature_columns, label_column)
    evaluation_time = time.time() - start_time
    metrics['sklearn']['evaluation_time_seconds'] = evaluation_time
    metrics['sklearn']['accuracy'] = accuracy
    print(f"Modèle évalué en {evaluation_time:.2f} secondes.")
    print(f"Précision du modèle Sklearn sur le jeu de test : {accuracy:.4f}")
    print("Maintenant, entraînement et évaluation avec MRTree de Spark MLlib...")
    spark_runner = ClusterExperimentRunner(IntrusionDataloader, IntrusionPreprocessor, MRTreeModelTrainer, seed=42)
    spark_runner.run(args.nodes, K=[1], full_dataset=True)
    spark_metrics = spark_runner.get_metrics()
    metrics['spark'] =  spark_metrics[1]
    print("Résultats de Spark MLlib MRTree :")
    print(f"Temps d'entraînement : {metrics['spark']['training_time_seconds']:.2f} secondes.")
    print(f"Précision : {metrics['spark']['accuracy']:.4f}")
    print(f"Fin de l'expérience.")
    return metrics

if __name__ == "__main__":
    experiment_metrics = experiment()
    print("="*40)
    print("Résumé des métriques de l'expérience Non-Distribuée vs Distribuée :")
    print(json.dumps(experiment_metrics, indent=4))
    print("="*40)
    with open("baseline_experiment_metrics.json", "w") as f:
        json.dump(experiment_metrics, f, indent=4)
