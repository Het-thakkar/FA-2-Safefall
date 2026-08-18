import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

FEATURE_DIR = "extracted_features"
MODEL_SAVE_PATH = "fall_detection_rf.pkl"
SCALER_SAVE_PATH = "feature_scaler.pkl"

CLASSES = {'Falling': 0, 'Walking': 1, 'Sitting': 2, 'Standing': 3, 'Normal': 4}


def load_data():
    X, y = [], []
    for f in os.listdir(FEATURE_DIR):
        if f.endswith('.npy'):
            label_str = f.split('_')[0]
            if label_str in CLASSES:
                X.append(np.load(os.path.join(FEATURE_DIR, f)))
                y.append(CLASSES[label_str])
    return np.array(X), np.array(y), CLASSES


if __name__ == "__main__":
    X, y, classes = load_data()
    if len(X) == 0:
        exit("No features found. Run extract_features.py first.")

    print("Samples per class:")
    for name, idx in classes.items():
        count = int(np.sum(y == idx))
        print(f"  {name}: {count}")
        if count < 30:
            print(f"    [!] Low sample count for '{name}' — this class will likely hurt test "
                  f"accuracy. Consider increasing MAX_TARGET_* in extract_features.py and re-running.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # --- Feature scaling (helps some splits/models, cheap to do, doesn't hurt Random Forest) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Hyperparameter tuning via cross-validation instead of guessed values ---
    # The previous fixed settings (n_estimators=300, max_depth=8) overfit: 99% train / 78% test.
    # Searching min_samples_leaf / min_samples_split / max_features directly targets that gap,
    # since these control how much a tree can "memorize" individual training points.
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6, 8, 10],
        'min_samples_leaf': [2, 4, 8],
        'min_samples_split': [4, 8, 12],
        'max_features': ['sqrt', 'log2'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    print("\nRunning cross-validated hyperparameter search (this may take a few minutes)...")
    search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_scaled, y_train)

    print(f"\nBest params found: {search.best_params_}")
    print(f"Best cross-validated training accuracy: {search.best_score_:.4f}")

    rf_model = search.best_estimator_

    # --- Also report stratified 5-fold CV score on the full dataset for a more honest estimate ---
    full_cv_scores = cross_val_score(rf_model, scaler.fit_transform(X), y, cv=cv, scoring='accuracy')
    print(f"5-fold CV accuracy on full dataset: {full_cv_scores.mean():.4f} (+/- {full_cv_scores.std():.4f})")

    # Refit scaler + model on the actual train split for the held-out test evaluation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_model.fit(X_train_scaled, y_train)

    joblib.dump(rf_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Scaler saved to {SCALER_SAVE_PATH} (app.py must use this to transform features before predicting)")

    train_acc = rf_model.score(X_train_scaled, y_train)
    test_acc = rf_model.score(X_test_scaled, y_test)
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Testing Accuracy:  {test_acc:.4f}")

    if test_acc < 0.9:
        print("\n[!] Still below 90% test accuracy. This usually means the dataset itself is the "
              "limiting factor, not the model. Next steps to try:")
        print("    1. Increase MAX_TARGET_FALL_SAMPLES / MAX_TARGET_PER_NORMAL_CLASS in extract_features.py")
        print("       and re-run it to get more training examples per class.")
        print("    2. Check the per-class sample counts printed above — any class under ~30-40")
        print("       samples will be unreliable no matter how the model is tuned.")
        print("    3. The Walking/Sitting/Standing/Normal labels are heuristic (rule-based), not")
        print("       ground truth — some mislabeling is expected and caps achievable accuracy.")
