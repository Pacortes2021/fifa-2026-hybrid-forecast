"""
tda_motor.py  — Análisis Topológico de Datos (TDA) para el Mundial 2026

Implementa la misma idea que las publicaciones virales de LinkedIn, PERO:
  1. Usa las variables reales del proyecto (Elo, valor plantilla, forma reciente).
  2. Compara de forma rigurosa (log-loss) contra el modelo ML tradicional.
  3. No exagera las conclusiones: TDA describe, ML predice.

Pipeline:
  A) Nube de puntos: 48 equipos en R^5 normalizado.
  B) Filtración Vietoris-Rips → Homología persistente (ripser).
  C) Features topológicos: entropía de persistencia H0/H1, números de Betti a ε fijo.
  D) Modelo TDA: logistic regression sobre features topológicos + partido-a-partido.
  E) Comparación: log-loss TDA vs Híbrido en los mismos partidos.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

try:
    import ripser
    from persim import plot_diagrams as _plot_diagrams
    TDA_OK = True
except ImportError:
    TDA_OK = False

DATA = Path(__file__).resolve().parent.parent / "data"

# Las 5 variables del espacio de fases (igual que el análisis viral, pero con las nuestras)
VARS_TDA = ["elo", "squad_value_log", "goles_anotados_avg", "goles_recibidos_avg", "tiros_arco_avg"]
MUNDIALISTAS_48 = None   # se carga con cargar_nube()

# ─────────────────────────────────────────────────────────────────────────────
#  A) NUBE DE PUNTOS EN R^5
# ─────────────────────────────────────────────────────────────────────────────

def cargar_nube(states=None):
    """Construye la nube de puntos de los 48 equipos mundialistas en R^5 normalizado.
    Devuelve (df_raw, X_norm, nombres, scaler).
    """
    if states is None:
        states = pd.read_csv(DATA / "team_states.csv").set_index("team")

    # Solo los 48 mundialistas (importar desde motor para evitar circular)
    from motor import MUNDIALISTAS
    equipos = [t for t in MUNDIALISTAS if t in states.index]

    df = states.loc[equipos].copy()
    df["squad_value_log"] = np.log(df["squad_value"].clip(1e6))

    X_raw = df[VARS_TDA].values.astype(float)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_raw)

    return df, X_norm, equipos, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  B) VIETORIS-RIPS → HOMOLOGÍA PERSISTENTE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_persistencia(X_norm, max_dim=2):
    """Calcula la homología persistente con Vietoris-Rips usando ripser.
    Devuelve el dict de diagramas: {'dgms': [H0, H1, H2], ...}
    """
    if not TDA_OK:
        raise ImportError("Instala ripser: pip install ripser persim")
    resultado = ripser.ripser(X_norm, maxdim=max_dim)
    return resultado


def numeros_betti(dgms, epsilon):
    """Calcula β0, β1, β2 a un radio epsilon dado.
    Un rasgo [birth, death] está 'vivo' si birth <= epsilon < death.
    """
    bettis = []
    for dim, dgm in enumerate(dgms[:3]):
        if len(dgm) == 0:
            bettis.append(0)
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        # Reemplazar inf por un valor muy grande
        deaths = np.where(np.isinf(deaths), 1e10, deaths)
        vivo = np.sum((births <= epsilon) & (epsilon < deaths))
        bettis.append(int(vivo))
    return bettis   # [β0, β1, β2]


def entropia_persistencia(dgm):
    """Entropía de persistencia de un diagrama: mide la complejidad topológica.
    Valores altos = muchos rasgos de vida similar (torneo competitivo).
    Valores bajos = pocos rasgos dominantes (torneo predecible).
    """
    if len(dgm) == 0:
        return 0.0
    births, deaths = dgm[:, 0], dgm[:, 1]
    finite = ~np.isinf(deaths)
    if finite.sum() == 0:
        return 0.0
    lifetimes = (deaths - births)[finite]
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def resumen_topologico(resultado):
    """Genera un DataFrame con los rasgos topológicos más relevantes.
    Incluye: nacimiento, muerte, persistencia y dimensión de cada rasgo.
    """
    dgms = resultado["dgms"]
    filas = []
    for dim, dgm in enumerate(dgms[:3]):
        for b, d in dgm:
            persistencia = (d - b) if not np.isinf(d) else np.nan
            filas.append({
                "Dimensión": f"H{dim}",
                "Nace en ε": round(b, 4),
                "Muere en ε": round(d, 4) if not np.isinf(d) else "∞",
                "Persistencia": round(persistencia, 4) if not np.isnan(persistencia) else "∞",
                "Tipo": {0: "Componente conexa", 1: "Ciclo (loop)", 2: "Cavidad"}[dim]
            })
    df = pd.DataFrame(filas)
    df = df[df["Dimensión"] != "H0"].reset_index(drop=True) if len(df) > 0 else df
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  C) FEATURES TOPOLÓGICOS POR PAR DE EQUIPOS (para predecir partidos)
# ─────────────────────────────────────────────────────────────────────────────

def features_tda_partido(X_norm, equipos, a, b, resultado_global):
    """Features TDA para un partido concreto a vs b:
      - Diferencia de distancias al centroide (¿quién está más cerca del "núcleo duro"?)
      - Distancia euclidiana entre a y b en R^5 normalizado
      - Posición relativa en el espacio topológico (percentil de ε de conexión)
      - Entropías del diagrama global como contexto
    """
    idx_a = equipos.index(a) if a in equipos else None
    idx_b = equipos.index(b) if b in equipos else None
    if idx_a is None or idx_b is None:
        return None

    centroide = X_norm.mean(axis=0)
    dist_a = float(np.linalg.norm(X_norm[idx_a] - centroide))
    dist_b = float(np.linalg.norm(X_norm[idx_b] - centroide))
    dist_ab = float(np.linalg.norm(X_norm[idx_a] - X_norm[idx_b]))

    dgms = resultado_global["dgms"]
    ent_h0 = entropia_persistencia(dgms[0])
    ent_h1 = entropia_persistencia(dgms[1]) if len(dgms) > 1 else 0.0

    return {
        "dist_centroide_diff": dist_a - dist_b,   # + → a más periférico
        "dist_eucl_ab": dist_ab,
        "dist_centroide_a": dist_a,
        "dist_centroide_b": dist_b,
        "entropia_h0": ent_h0,
        "entropia_h1": ent_h1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  D) MODELO TDA: logistic regression con features topológicos
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_TDA_COLS = ["dist_centroide_diff", "dist_eucl_ab",
                     "dist_centroide_a", "dist_centroide_b",
                     "entropia_h0", "entropia_h1"]


def entrenar_modelo_tda(M, resultado_global, X_norm, equipos):
    """Entrena un clasificador logístico usando SOLO features topológicos.
    Devuelve (modelo, df_partidos_con_features).
    """
    from motor import MUNDIALISTAS, tipo_competicion
    K_FACTOR = {"amistoso": 1.0, "clasificatoria": 2.0, "nations_league": 2.0,
                "continental": 2.5, "mundial": 3.0}

    df = M["df"].copy()
    m48 = set(MUNDIALISTAS)
    df = df[df.local.isin(m48) & df.visita.isin(m48)].dropna(subset=["resultado"])

    filas_X, filas_y, pesos = [], [], []
    for r in df.itertuples(index=False):
        feats = features_tda_partido(X_norm, equipos, r.local, r.visita, resultado_global)
        if feats is None:
            continue
        filas_X.append([feats[c] for c in FEATURES_TDA_COLS])
        filas_y.append(int(r.resultado))
        w = K_FACTOR.get(tipo_competicion(r.competicion), 1.0)
        pesos.append(w)

    X_tr = np.array(filas_X)
    y_tr = np.array(filas_y)
    w_tr = np.array(pesos)

    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])
    pipe.fit(X_tr, y_tr, m__sample_weight=w_tr)
    return pipe, df


# ─────────────────────────────────────────────────────────────────────────────
#  E) COMPARACIÓN RIGUROSA: TDA vs Híbrido en partidos reales del Mundial
# ─────────────────────────────────────────────────────────────────────────────

def comparar_en_mundial(M, resultados_espn, modelo_ml="base"):
    """Compara TDA vs ML en los partidos reales del Mundial ya jugados.
    Devuelve tabla de resultados partido-a-partido + métricas globales.
    """
    from motor import prob_partido

    # 1. Construir nube TDA
    df_raw, X_norm, equipos, scaler = cargar_nube(M["states"])
    resultado_global = calcular_persistencia(X_norm)

    # 2. Entrenar modelo TDA (en datos históricos, no en el Mundial)
    pipe_tda, _ = entrenar_modelo_tda(M, resultado_global, X_norm, equipos)

    st = M["states"]
    filas = []
    P_ml, P_tda, y_real = [], [], []

    for r in resultados_espn.itertuples(index=False):
        a, b = r.local, r.visita
        if a not in st.index or b not in st.index:
            continue

        ga, gb = int(r.goles_local), int(r.goles_visita)
        real = 2 if ga > gb else (1 if ga == gb else 0)

        # Predicción ML tradicional
        p_ml = prob_partido(M, a, b, "auto", modelo_ml)  # [P(a), P(empate), P(b)]

        # Predicción TDA
        feats_tda = features_tda_partido(X_norm, equipos, a, b, resultado_global)
        if feats_tda is None:
            continue
        X_pred = np.array([[feats_tda[c] for c in FEATURES_TDA_COLS]])
        clases = pipe_tda.classes_
        p_raw = pipe_tda.predict_proba(X_pred)[0]
        # Ordenar en [0, 1, 2]
        p_tda_ord = np.zeros(3)
        for i, cls in enumerate(clases):
            p_tda_ord[int(cls)] = p_raw[i]

        # Predicción "naive" del TDA (solo distancia euclidiana al centroide)
        dc_a = np.linalg.norm(X_norm[equipos.index(a)] - X_norm.mean(axis=0)) if a in equipos else 0
        dc_b = np.linalg.norm(X_norm[equipos.index(b)] - X_norm.mean(axis=0)) if b in equipos else 0
        dist_ab = np.linalg.norm(X_norm[equipos.index(a)] - X_norm[equipos.index(b)]) if (a in equipos and b in equipos) else 1

        pred_ml = int(np.argmax([p_ml[2], p_ml[1], p_ml[0]]))  # 0=visita gana, 1=empate, 2=local
        pred_tda = int(np.argmax(p_tda_ord))

        filas.append({
            "Partido": f"{a} vs {b}",
            "Resultado": f"{ga}-{gb}",
            "Real": ["Gana visita", "Empate", "Gana local"][real],
            "ML P(local)": f"{p_ml[0]:.0%}",
            "ML P(empate)": f"{p_ml[1]:.0%}",
            "ML P(visita)": f"{p_ml[2]:.0%}",
            "TDA P(local)": f"{p_tda_ord[2]:.0%}",
            "TDA P(empate)": f"{p_tda_ord[1]:.0%}",
            "TDA P(visita)": f"{p_tda_ord[0]:.0%}",
            "✅ ML": "✅" if pred_ml == real else "❌",
            "✅ TDA": "✅" if pred_tda == real else "❌",
            "_real": real,
            "_p_ml": [p_ml[2], p_ml[1], p_ml[0]],
            "_p_tda": p_tda_ord.tolist(),
        })

        P_ml.append([p_ml[2], p_ml[1], p_ml[0]])
        P_tda.append(p_tda_ord.tolist())
        y_real.append(real)

    tabla = pd.DataFrame(filas)

    if len(y_real) == 0:
        return tabla, {}, resultado_global, X_norm, equipos

    P_ml_arr = np.array(P_ml)
    P_tda_arr = np.array(P_tda)
    y_arr = np.array(y_real)
    base = np.tile([0.279, 0.275, 0.446], (len(y_arr), 1))

    metricas = {
        "n": len(y_arr),
        "logloss_ml": log_loss(y_arr, P_ml_arr, labels=[0, 1, 2]),
        "logloss_tda": log_loss(y_arr, P_tda_arr, labels=[0, 1, 2]),
        "logloss_base": log_loss(y_arr, base, labels=[0, 1, 2]),
        "acierto_ml":  sum(1 for r, p in zip(y_arr, P_ml_arr) if np.argmax(p) == r) / len(y_arr),
        "acierto_tda": sum(1 for r, p in zip(y_arr, P_tda_arr) if np.argmax(p) == r) / len(y_arr),
    }

    # Columnas limpias para mostrar
    cols_show = ["Partido", "Resultado", "Real",
                 "ML P(local)", "ML P(empate)", "ML P(visita)", "✅ ML",
                 "TDA P(local)", "TDA P(empate)", "TDA P(visita)", "✅ TDA"]
    tabla_limpia = tabla[cols_show].copy()

    return tabla_limpia, metricas, resultado_global, X_norm, equipos


# ─────────────────────────────────────────────────────────────────────────────
#  F) FIGURAS (para Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def fig_nube_3d(X_norm, equipos, estados):
    """Nube de puntos 3D (proyección de R^5): ejes = Elo, valor plantilla, forma."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from motor import MUNDIALISTAS, GRUPOS

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Colores por grupo
    colores_grupo = plt.cm.tab20(np.linspace(0, 1, 12))
    GRUPO_DE = {t: g for g, ts in GRUPOS.items() for t in ts}
    grupos_unicos = sorted(GRUPOS.keys())
    color_map = {g: colores_grupo[i] for i, g in enumerate(grupos_unicos)}

    xs = X_norm[:, 0]   # Elo
    ys = X_norm[:, 1]   # squad_value_log
    zs = X_norm[:, 2]   # goles_anotados_avg

    for i, eq in enumerate(equipos):
        g = GRUPO_DE.get(eq, "?")
        c = color_map.get(g, "grey")
        ax.scatter(xs[i], ys[i], zs[i], c=[c], s=60, alpha=0.8)

    # Destacar finalistas actuales
    destacados = {"Spain": "🇪🇸", "Argentina": "🇦🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷"}
    for eq, fl in destacados.items():
        if eq in equipos:
            i = equipos.index(eq)
            ax.scatter(xs[i], ys[i], zs[i], c="gold", s=200, zorder=5,
                       edgecolors="black", linewidth=1.5)
            ax.text(xs[i], ys[i], zs[i] + 0.03, fl, fontsize=9, ha="center")

    ax.set_xlabel("Elo (norm)", fontsize=9)
    ax.set_ylabel("Valor plantilla log (norm)", fontsize=9)
    ax.set_zlabel("Goles anotados avg (norm)", fontsize=9)
    ax.set_title("Nube de puntos — 48 equipos en $\\mathbb{R}^5$ (proyección 3D)\n"
                 "⭐ = Semifinalistas actuales", fontsize=11)
    fig.tight_layout()
    return fig


def fig_diagrama_persistencia(resultado):
    """Diagrama de persistencia estilo publicación."""
    import matplotlib.pyplot as plt
    dgms = resultado["dgms"]
    colores = {0: "#0b3d91", 1: "#e63946", 2: "#2a9d5c"}
    nombres = {0: "$H_0$ — Componentes", 1: "$H_1$ — Ciclos", 2: "$H_2$ — Cavidades"}

    fig, ax = plt.subplots(figsize=(7, 6))
    max_val = 0
    for dim in range(min(3, len(dgms))):
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        births = dgm[:, 0]
        deaths = np.where(np.isinf(dgm[:, 1]), dgm[:, 1].clip(max=10), dgm[:, 1])
        finite_mask = ~np.isinf(dgms[dim][:, 1])
        ax.scatter(births[finite_mask], deaths[finite_mask],
                   c=colores[dim], label=nombres[dim], s=50, alpha=0.85, zorder=3)
        if any(~finite_mask):
            ax.scatter(births[~finite_mask], [deaths.max() * 1.05] * sum(~finite_mask),
                       c=colores[dim], marker="^", s=60, alpha=0.6)
        if len(deaths[finite_mask]) > 0:
            max_val = max(max_val, deaths[finite_mask].max())

    lim = max(max_val * 1.15, 0.5)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="Diagonal (ruido)")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim * 1.1)
    ax.set_xlabel("Nacimiento (ε de inicio)", fontsize=10)
    ax.set_ylabel("Muerte (ε de fin)", fontsize=10)
    ax.set_title("Diagrama de Persistencia — Mundial 2026\n"
                 "Puntos lejos de la diagonal = rasgos topológicos importantes", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_betti_vs_epsilon(resultado, n_eps=80):
    """Curvas de Betti β0, β1, β2 en función del radio ε."""
    import matplotlib.pyplot as plt
    dgms = resultado["dgms"]

    # Rango de epsilon
    todos = []
    for dgm in dgms[:3]:
        if len(dgm):
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite):
                todos.append(finite[:, 1].max())
    max_eps = max(todos) * 1.1 if todos else 5.0
    epsilons = np.linspace(0, max_eps, n_eps)

    betti = {0: [], 1: [], 2: []}
    for eps in epsilons:
        b = numeros_betti(dgms, eps)
        for dim in range(3):
            betti[dim].append(b[dim])

    fig, ax = plt.subplots(figsize=(9, 4))
    colores = {0: "#0b3d91", 1: "#e63946", 2: "#2a9d5c"}
    etiquetas = {0: "β₀ — Componentes conexas", 1: "β₁ — Ciclos (loops)", 2: "β₂ — Cavidades"}
    for dim in range(3):
        ax.plot(epsilons, betti[dim], color=colores[dim],
                label=etiquetas[dim], lw=2.5)

    ax.set_xlabel("Radio ε de conexión", fontsize=10)
    ax.set_ylabel("Número de Betti", fontsize=10)
    ax.set_title("Números de Betti vs Radio ε\n"
                 "β₀=1 → todos conectados | β₁>0 → ciclos | β₂>0 → cavidades", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_comparacion_logloss(metricas):
    """Gráfico de barras: log-loss TDA vs ML vs Baseline."""
    import matplotlib.pyplot as plt

    modelos = ["Baseline\n(frecuencias)", "Modelo TDA\n(topología)", "Modelo ML\n(Híbrido)"]
    valores = [metricas["logloss_base"], metricas["logloss_tda"], metricas["logloss_ml"]]
    aciertos = [None, metricas["acierto_tda"], metricas["acierto_ml"]]
    colores = ["#94a3b8", "#e63946", "#0b3d91"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars = ax1.bar(modelos, valores, color=colores, width=0.5, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Log-Loss (menor = mejor)", fontsize=10)
    ax1.set_title(f"Log-Loss en {metricas['n']} partidos del Mundial", fontsize=11)
    for bar, val in zip(bars, valores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(valores) * 1.2)
    ax1.grid(axis="y", alpha=0.3)

    # Acierto
    ac_vals = [metricas["acierto_tda"], metricas["acierto_ml"]]
    ac_cols = ["#e63946", "#0b3d91"]
    ac_labs = ["TDA", "ML Híbrido"]
    bars2 = ax2.bar(ac_labs, [v * 100 for v in ac_vals], color=ac_cols,
                    width=0.4, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("% Acierto (1X2)", fontsize=10)
    ax2.set_title("Tasa de acierto (1X2)", fontsize=11)
    for bar, val in zip(bars2, ac_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.5,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("TDA vs ML Híbrido — Comparación directa en partidos reales", fontsize=12)
    fig.tight_layout()
    return fig


def tabla_rasgos_topologicos_descripcion(resultado, equipos, X_norm):
    """Tabla resumen: ¿Qué dice la topología sobre la estructura del torneo?"""
    dgms = resultado["dgms"]

    # Radio de conexión total (cuando β0 = 1)
    dgm0 = dgms[0]
    muertes_h0 = dgm0[:, 1]
    radio_conexion_total = float(muertes_h0[~np.isinf(muertes_h0)].max()) if any(~np.isinf(muertes_h0)) else None

    # Ciclos persistentes (H1)
    ciclos = []
    if len(dgms) > 1:
        for b, d in dgms[1]:
            if not np.isinf(d) and (d - b) > 0.05:
                ciclos.append({"nace": round(b, 3), "muere": round(d, 3),
                                "persistencia": round(d - b, 3)})

    # Equipos más aislados (mayor distancia al centroide)
    centroide = X_norm.mean(axis=0)
    distancias = {eq: float(np.linalg.norm(X_norm[i] - centroide))
                  for i, eq in enumerate(equipos)}
    top_aislados = sorted(distancias.items(), key=lambda x: -x[1])[:5]
    top_centrales = sorted(distancias.items(), key=lambda x: x[1])[:5]

    return {
        "radio_conexion_total": radio_conexion_total,
        "n_ciclos_persistentes": len(ciclos),
        "ciclos": ciclos,
        "top_aislados": top_aislados,
        "top_centrales": top_centrales,
        "entropia_h0": round(entropia_persistencia(dgms[0]), 3),
        "entropia_h1": round(entropia_persistencia(dgms[1]) if len(dgms) > 1 else 0.0, 3),
    }
