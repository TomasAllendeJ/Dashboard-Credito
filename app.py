# -*- coding: utf-8 -*-
"""
Dashboard interactivo - Certamen I Machine Learning
Grupo 1: Tomas Allende, Felipe Castillo, Juan Costa
Base: Give Me Some Credit (Kaggle)

Muestra, en dinero, cuanto cuesta cada error (FP y FN) y cual es el umbral de decision
que minimiza el costo total. Mueve los costos y el umbral y observa como cambia todo.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# Configuracion de la pagina
# ----------------------------------------------------------------------
st.set_page_config(page_title="Dashboard - Riesgo de Credito", page_icon="💳", layout="wide")

# ----------------------------------------------------------------------
# Cargar datos (probabilidades del test exportadas por el notebook)
# ----------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("datos_dashboard.csv")
    return df["y_real"].to_numpy(), df["proba"].to_numpy()

try:
    y_real, p = cargar_datos()
except FileNotFoundError:
    st.error("No se encontro 'datos_dashboard.csv'. Ejecuta el notebook para generarlo "
             "y ponlo en la misma carpeta que este archivo app.py.")
    st.stop()

P = int((y_real == 1).sum())   # morosos reales en el test
N = int((y_real == 0).sum())   # buenos pagadores reales en el test

# ----------------------------------------------------------------------
# Titulo
# ----------------------------------------------------------------------
st.title("💳 Dashboard de Riesgo de Credito")
st.caption("Certamen I - Machine Learning | Grupo 1: Allende, Castillo, Costa | "
           "Base: Give Me Some Credit")
st.markdown("Decidimos a quien aprobar un credito. El modelo estima la probabilidad de que un "
            "cliente **caiga en mora**. Aqui vemos **cuanto cuesta en dinero cada error** y cual "
            "es el **umbral optimo** que minimiza ese costo.")

# ----------------------------------------------------------------------
# Controles (barra lateral)
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Controles")
st.sidebar.markdown("**Costos del negocio**")
C_FP = st.sidebar.slider("Costo de un Falso Positivo (US$)", 100, 5000, 1500, 100,
                         help="Rechazar a un cliente que SI habria pagado: margen que se deja de ganar.")
C_FN = st.sidebar.slider("Costo de un Falso Negativo (US$)", 1000, 60000, 15000, 1000,
                         help="Aprobar a alguien que NO paga: perdida del monto prestado.")

p_star = C_FP / (C_FP + C_FN)   # umbral teorico optimo

st.sidebar.markdown("---")
usar_optimo = st.sidebar.checkbox("Usar umbral optimo p*", value=True)
umbral_manual = st.sidebar.slider("Umbral manual", 0.01, 0.99, 0.50, 0.01,
                                  disabled=usar_optimo,
                                  help="Si desmarcas 'usar optimo', puedes mover el umbral a mano.")

umbral = p_star if usar_optimo else umbral_manual

# ----------------------------------------------------------------------
# Calculos
# ----------------------------------------------------------------------
def matriz(umbral):
    pred = (p >= umbral).astype(int)
    tp = int(((pred == 1) & (y_real == 1)).sum())
    fp = int(((pred == 1) & (y_real == 0)).sum())
    fn = int(((pred == 0) & (y_real == 1)).sum())
    tn = int(((pred == 0) & (y_real == 0)).sum())
    return tp, fp, fn, tn

def costo(umbral, cfp, cfn):
    tp, fp, fn, tn = matriz(umbral)
    return cfp * fp + cfn * fn

tp, fp, fn, tn = matriz(umbral)
costo_actual = C_FP * fp + C_FN * fn
costo_05 = costo(0.5, C_FP, C_FN)
ahorro = costo_05 - costo_actual
ahorro_pct = (100 * ahorro / costo_05) if costo_05 > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# ----------------------------------------------------------------------
# Indicadores (KPIs)
# ----------------------------------------------------------------------
st.markdown("### Indicadores")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Umbral optimo p*", f"{p_star:.3f}")
c2.metric("Umbral usado", f"{umbral:.3f}")
c3.metric("Costo total", f"US$ {costo_actual:,.0f}")
c4.metric("Ahorro vs umbral 0.5", f"US$ {ahorro:,.0f}", f"{ahorro_pct:.1f}%")
c5.metric("Recall (morosos detectados)", f"{recall:.1%}")

st.markdown("---")

# ----------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------
col_izq, col_der = st.columns(2)

# --- Curva de costo vs umbral ---
with col_izq:
    st.markdown("#### Costo total vs. umbral")
    umbrales = np.linspace(0.01, 0.99, 99)
    costos = np.array([costo(t, C_FP, C_FN) for t in umbrales])
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(umbrales, costos, color="#2a5d9f")
    ax1.axvline(0.5, ls=":", color="gray", label="Umbral 0.5")
    ax1.axvline(p_star, ls="--", color="red", label=f"p* = {p_star:.3f}")
    ax1.axvline(umbral, ls="-", color="green", lw=1.5, label=f"Umbral usado = {umbral:.3f}")
    ax1.set_xlabel("Umbral")
    ax1.set_ylabel("Costo total (US$)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    st.caption("El punto mas bajo de la curva es el umbral que menos cuesta.")

# --- Matriz de confusion con costos ---
with col_der:
    st.markdown("#### Matriz de confusion (con costo de cada error)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    mc = np.array([[tn, fp], [fn, tp]])
    etiquetas = np.array([[f"TN\n{tn:,}\nUS$ 0",            f"FP\n{fp:,}\nUS$ {C_FP*fp:,}"],
                          [f"FN\n{fn:,}\nUS$ {C_FN*fn:,}",  f"TP\n{tp:,}\nUS$ 0"]])
    ax2.imshow(mc, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, etiquetas[i, j], ha="center", va="center", fontsize=10)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Aprueba (0)", "Rechaza (1)"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Paga (0)", "No paga (1)"])
    ax2.set_xlabel("Prediccion del modelo"); ax2.set_ylabel("Realidad")
    st.pyplot(fig2)
    st.caption("FP = rechazar a un buen cliente | FN = prestar a quien no paga (el error caro).")

# ----------------------------------------------------------------------
# Desglose del costo
# ----------------------------------------------------------------------
st.markdown("#### Desglose del costo frente a estrategias de referencia")
ref = pd.DataFrame({
    "Estrategia": ["Aprobar a todos", "Umbral 0.5", "Umbral usado"],
    "Costo en FP (US$)": [0, C_FP*((p >= 0.5) & (y_real == 0)).sum(), C_FP*fp],
    "Costo en FN (US$)": [C_FN*P, C_FN*((p < 0.5) & (y_real == 1)).sum(), C_FN*fn],
})
ref["Costo total (US$)"] = ref["Costo en FP (US$)"] + ref["Costo en FN (US$)"]
fig3, ax3 = plt.subplots(figsize=(10, 2.8))
ax3.barh(ref["Estrategia"], ref["Costo en FP (US$)"], color="#4e79a7", label="Gasto en FP")
ax3.barh(ref["Estrategia"], ref["Costo en FN (US$)"], left=ref["Costo en FP (US$)"],
         color="#e15759", label="Margen perdido FN")
for i, v in enumerate(ref["Costo total (US$)"]):
    ax3.text(v, i, f" US$ {v:,.0f}", va="center", fontsize=9)
ax3.set_xlabel("Costo total en test (US$)")
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3, axis="x")
st.pyplot(fig3)

st.markdown("---")
st.markdown(f"**Precision:** {precision:.1%}  |  **Clientes en el test:** {len(y_real):,} "
            f"({P:,} morosos, {N:,} buenos pagadores)")
st.caption("Mueve los costos en la barra lateral: veras como el umbral optimo p* y el ahorro "
           "cambian en vivo.")
