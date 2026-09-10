from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "artifacts" / "iris_model.joblib"

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2,
    random_state=42, stratify=iris.target
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=300, random_state=42)),
])
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

artifact = {
    "model": model,
    "target_names": iris.target_names.tolist(),
    "test_accuracy": accuracy,
}
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(artifact, MODEL_PATH)
print(f"정확도:{accuracy:.3f}")
print(f"저장 완료:{MODEL_PATH}")