import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class DronePredictivePipeline:
    def __init__(self):
        
        # 1. DATA GENERATION: Simulating Drone Telemetry
        np.random.seed(42)
        n_samples = 1000
        
        self.raw_data = {
            "wind_speed": np.random.uniform(5.0, 45.0, n_samples),
            "rotor_rpm": np.random.normal(3200, 200, n_samples),
            "battery_temp": np.random.normal(45, 10, n_samples),
            "vibration_level": np.random.uniform(0.1, 2.5, n_samples)
        }
        
        df = pd.DataFrame(self.raw_data)
        
        risk_condition = (df["wind_speed"] > 32.0) | (df["vibration_level"] > 1.9) | (df["battery_temp"] > 62.0)
        df["flight_status"] = np.where(risk_condition, 0, 1) # 0 = Failure Risk, 1 = Stable
        
        self.X = df.drop(columns=["flight_status"])
        self.y = df["flight_status"]

    def run_pipeline(self):

        # 2. TRAIN / TEST SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Total Dataset Size: {len(self.X)} flights")
        print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

        # 3. PIPELINE CONSTRUCTION (Feature Scaling + XGBoost)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBClassifier(eval_metric="logloss", random_state=42))
        ])

        # 4. HYPERPARAMETER TUNING GRID
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2]
        }

        # 5. CROSS-VALIDATION & GRID SEARCH
        print("\n--- Executing 5-Fold Cross-Validation & Hyperparameter Grid Search ---")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # 6. EVALUATION ON UNSEEN DATA
        print("\n--- Optimization Complete ---")
        print(f"Best Hyperparameters Found:\n{grid_search.best_params_}")
        print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n--- Final Test Set Evaluation ---")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    app = DronePredictivePipeline()
    app.run_pipeline()
