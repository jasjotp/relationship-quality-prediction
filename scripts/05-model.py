import argparse
from pathlib import Path
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_preprocessor",
        type=str,
        required=True,
        help="Path to the preprocessed dataset pickle (X_train, X_test, y_train, y_test)",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Where to save the trained model pickle file",
    )
    args = parser.parse_args()

    input_path = Path(args.input_preprocessor)
    output_model_path = Path(args.output_model)

    print(f"Loading preprocessed data from: {input_path}")
    X_train, X_test, y_train, y_test = joblib.load(input_path)

    # train models
    
        print("Training Dummy Classifier (baseline)...")
    dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_model.fit(X_train, y_train)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)


    # save trained models

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving trained Logistic Regression model to: {output_model_path}")
    joblib.dump(model, output_model_path)

    dummy_output_path = output_model_path.with_name("dummy_model.pkl")
    print(f"Saving trained Dummy model to: {dummy_output_path}")
    joblib.dump(dummy_model, dummy_output_path)

    
    # evaluation / reporting
    
    models = {
        "Logistic Regression": model,
        "Dummy Classifier": dummy_model,
    }

    for name, m in models.items():
        print(f"\n=== {name} ===")

        y_train_pred = m.predict(X_train)
        y_test_pred = m.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"Train accuracy: {train_acc:.3f}")
        print(f"Test accuracy:  {test_acc:.3f}")
    
    # plots (Logistic Regression only)
    
    plots_dir = output_model_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    ax_cm.set_title("Logistic Regression – Confusion Matrix")
    plt.tight_layout()

    cm_pickle_path = plots_dir / "confusion_matrix.pkl"
    print(f"Saving confusion matrix figure to: {cm_pickle_path}")
    with open(cm_pickle_path, "wb") as f:
        pickle.dump(fig_cm, f)

    plt.show()

    # ROC Curve
y_test_prob = model.predict_proba(X_test)[:, 1]

fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax_roc)
ax_roc.set_title("Logistic Regression – ROC Curve")
plt.tight_layout()

roc_pickle_path = plots_dir / "roc_curve.pkl"
print(f"Saving ROC curve figure to: {roc_pickle_path}")
with open(roc_pickle_path, "wb") as f:
    pickle.dump(fig_roc, f)

plt.show()


if __name__ == "__main__":
    main()
