import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

# Configurar rutas para importar los motores
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mex"))
sys.path.insert(0, os.path.join(ROOT, "bra"))
sys.path.insert(0, os.path.join(ROOT, "chile"))

LIGAS = {
    "México (Liga MX)": {"path": "mex", "best_c": 0.05},
    "Brasil (Brasileirão)": {"path": "bra", "best_c": 0.02},
    "Chile (Primera)": {"path": "chile", "best_c": 0.05}
}

report_lines = []
report_lines.append("# Reporte de Experimentos: Modelos Predictivos Avanzados")
report_lines.append("Este reporte muestra los resultados de evaluar clasificadores no lineales avanzados contra el modelo actual (Regresión Logística L1 SAGA) utilizando partición temporal (Train: <= 2024, Test: 2025-2026).\n")

for liga_name, info in LIGAS.items():
    print(f"Evaluando modelos para {liga_name}...")
    report_lines.append(f"## 📊 {liga_name}")
    
    # Importar el motor específico de la liga
    import sys
    # Limpiar caché de motor para importar el correcto
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
    
    # Definir modelos
    models = {
        "Logistic Reg (L1 SAGA)": LogisticRegression(penalty="l1", solver="saga", C=info["best_c"], max_iter=2000, random_state=42),
        "Random Forest (depth=6)": RandomForestClassifier(max_depth=6, n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.05, random_state=42, eval_metric="mlogloss", n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, max_iter=100, learning_rate=0.05, random_state=42)
    }
    
    liga_results = []
    
    for name, clf in models.items():
        clf.fit(X_tr_sc, y_train)
        probs = clf.predict_proba(X_te_sc)
        preds = clf.predict(X_te_sc)
        
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        brier = np.mean([brier_score_loss(y_test_bin[:, i], probs[:, i]) for i in range(3)])
        
        liga_results.append({
            "Modelo": name,
            "Accuracy": f"{acc:.2%}",
            "Log-Loss": f"{loss:.4f}",
            "Brier Score": f"{brier:.4f}",
            "raw_loss": loss
        })
        
    df_res = pd.DataFrame(liga_results).sort_values("raw_loss")
    
    # Formatear como tabla markdown
    report_lines.append("| Modelo | Accuracy | Log-Loss (menor es mejor) | Brier Score |")
    report_lines.append("|---|---|---|---|")
    for idx, r in df_res.iterrows():
        # Resaltar al ganador
        bold_prefix = "**" if idx == df_res["raw_loss"].idxmin() else ""
        bold_suffix = "**" if idx == df_res["raw_loss"].idxmin() else ""
        report_lines.append(f"| {bold_prefix}{r['Modelo']}{bold_suffix} | {r['Accuracy']} | {r['Log-Loss']} | {r['Brier Score']} |")
    
    best_model = df_res.iloc[0]["Modelo"]
    report_lines.append(f"\n> 🏆 **Modelo Recomendado para {liga_name}:** {best_model}\n")
    print(f"Finalizado {liga_name}. Ganador: {best_model}")
    print("-" * 60)

# Escribir reporte en artefactos
artifact_path = "/Users/pabloignaciocortesvielma/.gemini/antigravity/brain/4143e6df-7203-487d-8b29-bb840f78cc73/experimento_resultados.md"
with open(artifact_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Reporte de experimentos guardado con éxito en: {artifact_path}")
