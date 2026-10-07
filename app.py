"""
Evaluación volumétrica probabilística (Monte Carlo) del OOIP.

OOIP = 7758 · A · h · φ · (1 − Sw) / Bo
  A en acres, h en pies, Bo en RB/STB, OOIP en barriles (STB).

Ejecutar:  streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

CONST = 7758.0  # bbl / (acre·ft)

# ----------------------------------------------------------------------------
# Muestreo
# ----------------------------------------------------------------------------
def triangular_inv(u, a, m, b):
    """CDF inversa de una distribución triangular (min=a, moda=m, max=b)."""
    u = np.asarray(u)
    if not (a <= m <= b) or a == b:
        raise ValueError("Se requiere min ≤ moda ≤ max y min < max.")
    fc = (m - a) / (b - a)
    return np.where(
        u <= fc,
        a + np.sqrt(u * (b - a) * (m - a)),
        b - np.sqrt((1 - u) * (b - a) * (b - m)),
    )


def uniform_inv(u, a, b):
    return a + np.asarray(u) * (b - a)


def run_simulation(params, n, seed):
    rng = np.random.default_rng(seed)
    u = {k: rng.random(n) for k in params}  # un RAND() por variable e iteración
    s = {}
    for k, p in params.items():
        if p["dist"] == "Triangular":
            s[k] = triangular_inv(u[k], p["min"], p["mode"], p["max"])
        else:
            s[k] = uniform_inv(u[k], p["min"], p["max"])
    ooip = CONST * s["A"] * s["h"] * s["phi"] * (1 - s["Sw"]) / s["Bo"]
    df = pd.DataFrame(s)
    df["OOIP"] = ooip
    return df


def summarize(x):
    """Convención petrolera: P90 = valor excedido con 90 % de prob. (percentil 10)."""
    counts, edges = np.histogram(x, bins=60)
    mode = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    return {
        "Media": x.mean(),
        "Mediana": np.median(x),
        "Moda (aprox.)": mode,
        "Desv. estándar": x.std(ddof=1),
        "P90": np.percentile(x, 10),
        "P50": np.percentile(x, 50),
        "P10": np.percentile(x, 90),
    }


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
st.set_page_config(page_title="OOIP Monte Carlo", layout="wide")
st.title("Evaluación volumétrica probabilística")
st.caption("Simulación Monte Carlo del OOIP = 7758 · A · h · φ · (1 − Sw) / Bo")

DEFAULTS = {
    "A":   dict(label="Área (acres)",        dist="Triangular", min=9950.0, mode=13760.0, max=17250.0),
    "h":   dict(label="Espesor neto (ft)",   dist="Triangular", min=8.0,    mode=10.5,    max=12.0),
    "phi": dict(label="Porosidad (fracción)", dist="Triangular", min=0.14,   mode=0.16,    max=0.17),
    "Sw":  dict(label="Sw (fracción)",       dist="Triangular", min=0.182,  mode=0.215,   max=0.31),
    "Bo":  dict(label="Bo (RB/STB)",         dist="Uniforme",   min=1.119,  mode=None,    max=1.141),
}

with st.sidebar:
    st.header("Parámetros")
    n = st.number_input("Iteraciones", 1_000, 2_000_000, 200_000, step=10_000)
    seed = st.number_input("Semilla aleatoria", 0, 10**9, 42)
    st.divider()
    params = {}
    for k, d in DEFAULTS.items():
        st.subheader(d["label"])
        c = st.columns(3 if d["dist"] == "Triangular" else 2)
        mn = c[0].number_input("Mín", value=float(d["min"]), format="%.4f", key=f"{k}_min")
        if d["dist"] == "Triangular":
            mo = c[1].number_input("Moda", value=float(d["mode"]), format="%.4f", key=f"{k}_mode")
            mx = c[2].number_input("Máx", value=float(d["max"]), format="%.4f", key=f"{k}_max")
        else:
            mo = None
            mx = c[1].number_input("Máx", value=float(d["max"]), format="%.4f", key=f"{k}_max")
        params[k] = dict(dist=d["dist"], min=mn, mode=mo, max=mx)

try:
    df = run_simulation(params, int(n), int(seed))
except ValueError as e:
    st.error(f"Parámetros inválidos: {e}")
    st.stop()

x_mm = df["OOIP"] / 1e6  # MMbbl
stats = summarize(x_mm.values)

# Métricas
m = st.columns(4)
m[0].metric("P90 (MMbbl)", f"{stats['P90']:,.1f}")
m[1].metric("P50 (MMbbl)", f"{stats['P50']:,.1f}")
m[2].metric("P10 (MMbbl)", f"{stats['P10']:,.1f}")
m[3].metric("Media (MMbbl)", f"{stats['Media']:,.1f}")

# Histograma
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(x_mm, bins=60, color="#c08a3e", edgecolor="white", linewidth=0.3)
for key, color in (("P90", "#2a8c82"), ("P50", "#222222"), ("P10", "#d9822b")):
    ax.axvline(stats[key], color=color, ls="--", lw=1.5, label=f"{key}: {stats[key]:.1f}")
ax.set_xlabel("OOIP (MMbbl)")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Distribución de OOIP — {int(n):,} iteraciones")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

left, right = st.columns(2)
with left:
    st.subheader("Estadísticos (MMbbl)")
    st.dataframe(
        pd.Series(stats, name="OOIP (MMbbl)").to_frame().style.format("{:,.2f}"),
        use_container_width=True,
    )
with right:
    st.subheader("Sensibilidad (correlación de rango con OOIP)")
    corr = (
        df.drop(columns="OOIP")
        .apply(lambda c: c.rank().corr(df["OOIP"].rank()))
        .sort_values(key=np.abs, ascending=True)
    )
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh(corr.index, corr.values, color="#5b7185")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

with st.expander("Ver iteraciones y descargar"):
    st.dataframe(df.head(200))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Descargar CSV completo", buf.getvalue(), "ooip_iteraciones.csv", "text/csv")

st.caption(
    "Convención: P90 = valor que se excede con 90 % de probabilidad (caso bajo); "
    "P10 = caso alto. Cada variable usa su propio número aleatorio uniforme(0,1) por iteración."
)
