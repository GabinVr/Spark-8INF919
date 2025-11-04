# Author Gabin VRILLAULT


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

class DecisionTreeBaselineMethod:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, train_df: pd.DataFrame, feature_cols: list, label_col: str):
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        self.model.fit(X_train, y_train)

    def evaluate(self, test_df: pd.DataFrame, feature_cols: list, label_col: str) -> float:
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]
        scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy').mean()
        print(f"Test scores: {scores}")
        print(f"Mean : {scores.mean()}")    
        return scores
    
class DataLoader:
    pass

if __name__ == "__main__":
    pass

    