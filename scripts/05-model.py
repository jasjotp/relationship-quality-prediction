from pathlib import Path
import pickle

import click
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize


@click.command()
@click.argument("processed_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("input_preprocessor", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_model", type=click.Path())
def main(processed_dir, input_preprocessor, output_model):
    processed_dir = Path(processed_dir)
    preprocessor_path = Path(input_preprocessor)
    output_model_path = Path(output_model)

    # load split data
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")

    # load and apply preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # fit on train only, transform train and test
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # train models
    dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy_model.fit(X_train_t, y_train)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_t, y_train)

    # save models
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_output_path = output_model_path.with_name("dummy_model.pkl")
    joblib.dump(dummy_model, dummy_output_path)
    joblib.dump(model, output_model_path)

    # evaluation
    y_train_pred = model.predict(X_train_t)
    y_test_pred = model.predict(X_test_t)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # multiclass ROC AUC (one-vs-rest)
    y_test_prob = model.predict_proba(X_test_t)  # (n_samples, n_classes)
    test_roc_auc_ovr = roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro")

    print(f"train accuracy: {train_acc:.3f}")
    print(f"test accuracy:  {test_acc:.3f}")
    print(f"test ROC AUC (ovr, macro): {test_roc_auc_ovr:.3f}")

    # roc curve plot (one-vs-rest curve per class) -> pickle
    plots_dir = output_model_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    class_order = list(model.classes_)  # preserves training class order
    y_test_bin = label_binarize(y_test, classes=class_order)  # (n_samples, n_classes)

    fig_roc, ax_roc = plt.subplots()

    for i, cls in enumerate(class_order):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_test_prob[:, i],
            ax=ax_roc,
            name=str(cls),
        )

    ax_roc.set_title("Logistic Regression ROC Curves (OvR)")
    plt.tight_layout()

    roc_pickle_path = plots_dir / "roc_curve.pkl"
    with open(roc_pickle_path, "wb") as f:
        pickle.dump(fig_roc, f)


if __name__ == "__main__":
    main()