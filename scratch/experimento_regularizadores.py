import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

# Configurar rutas para importar los motores
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mex"))
sys.path.insert(0, os.path.join(ROOT, "bra"))
sys.path.insert(0, os.path.join(ROOT, "chile"))

LIGAS = {
    "México (Liga MX)": {"path": "mex"},
    "Brasil (Brasileirão)": {"path": "bra"},
    "Chile (Primera)": {"path": "chile"}
}

print("Comparando Regularizadores: LASSO (L1) vs RIDGE (L2) vs ElasticNet (L1+L2)")
print("-" * 75)

for liga_name, info in LIGAS.items():
    print(f"\n📊 {liga_name}:")
    
    # Limpiar caché de motor
    for mod in ["motor", "recolectar", "recolectar_boxscore"]:
        if mod in sys.modules:
            del sys.modules[mod]
            
    sys.path.insert(0, os.path.join(ROOT, info["path"]))
    import motor as mo
    
    M = mo.cargar()
    df = M["df_dataset"]
    features = M["features"]
    
    train_mask = df["temporada"] <= 2024
    X_train = df.loc[train_mask, features].fillna(0.0)
    y_train = df.loc[train_mask, "resultado"]
    X_test = df.loc[~train_mask, features].fillna(0.0)
    y_test = df.loc[~train_mask, "resultado"]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    
    models = {
        "LASSO (L1 - SAGA)": LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=2000, random_state=42),
        "RIDGE (L2 - SAGA)": LogisticRegression(penalty="l2", solver="saga", C=0.05, max_iter=2000, random_state=42),
        "ElasticNet (L1_ratio=0.5)": LogisticRegression(penalty="elasticnet", solver="saga", C=0.05, l1_ratio=0.5, max_iter=2000, random_state=42),
        "ElasticNet (L1_ratio=0.2)": LogisticRegression(penalty="elasticnet", solver="saga", C=0.05, l1_ratio=0.2, max_iter=2000, random_state=42),
        "ElasticNet (L1_ratio=0.8)": LogisticRegression(penalty="elasticnet", solver="saga", C=0.05, l1_ratio=0.8, max_iter=2000, random_state=42)
    }
    
    results = []
    for name, clf in models.items():
        clf.fit(X_tr_sc, y_train)
        probs = clf.predict_proba(X_te_sc)
        preds = clf.predict(X_te_sc)
        
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        brier = np.mean([brier_score_loss(y_test_bin[:, i], probs[:, i]) for i in range(3)])
        
        results.append({
            "Modelo": name,
            "Accuracy": acc,
            "Log-Loss": loss,
            "Brier Score": brier
        })
        
    df_res = pd.DataFrame(results).sort_values("Log-Loss")
    print(df_res.to_string(index=False))
