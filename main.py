"""
Dashboard interactivo de datos sintéticos de COVID-19
Construido con Streamlit + Plotly.

Todos los datos son 100% simulados dentro de la propia aplicación
(no se consume ninguna fuente externa). Incluye:
  1. Una muestra pequeña e interactiva (10 registros x 8 columnas, tipos mixtos)
     que el usuario puede regenerar y editar.
  2. Un dataset simulado grande (>= 3000 registros) usado para el análisis
     cuantitativo y los gráficos.
  3. Un esquema de métricas (transformaciones: agregaciones, tasas,
     medias móviles, acumulados, correlaciones) calculado a partir del dataset grande.
  4. Gráficos Plotly interactivos controlados por widgets de Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID-19 (datos sintéticos)",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

START_DATE = date(2020, 3, 1)
END_DATE = date.today()

# Metadatos fijos por país: continente, nivel de ingreso y población base.
PAISES_META = {
    "Colombia":        ("América", "Medio", 52_000_000),
    "México":          ("América", "Medio", 128_000_000),
    "Argentina":       ("América", "Medio", 46_000_000),
    "Brasil":          ("América", "Medio", 215_000_000),
    "Chile":           ("América", "Alto",  19_500_000),
    "Perú":            ("América", "Medio", 34_000_000),
    "España":          ("Europa",  "Alto",  47_500_000),
    "Francia":         ("Europa",  "Alto",  68_000_000),
    "Italia":          ("Europa",  "Alto",  59_000_000),
    "Alemania":        ("Europa",  "Alto",  84_000_000),
    "Reino Unido":     ("Europa",  "Alto",  67_500_000),
    "Estados Unidos":  ("América", "Alto",  335_000_000),
    "India":           ("Asia",    "Bajo",  1_428_000_000),
    "China":           ("Asia",    "Medio", 1_412_000_000),
    "Sudáfrica":       ("África",  "Medio", 60_000_000),
}
PAISES = list(PAISES_META.keys())


# ----------------------------------------------------------------------------
# 1. Generadores de datos sintéticos
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_muestra_interactiva(seed: int) -> pd.DataFrame:
    """
    Genera una muestra pequeña: 10 registros x 8 columnas, con tipos de
    dato variados (fecha, texto, categoría, enteros, flotante y booleano).
    Pensada para que el usuario la vea/edite directamente en la plataforma.
    """
    rng = np.random.default_rng(seed)
    n = 10
    paises = rng.choice(PAISES, size=n, replace=True)
    fechas = [START_DATE + timedelta(days=int(d)) for d in rng.integers(0, (END_DATE - START_DATE).days, size=n)]

    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(fechas),                                   # datetime64
            "pais": paises,                                                    # object (str)
            "region": pd.Categorical([PAISES_META[p][0] for p in paises]),      # category
            "casos_confirmados": rng.integers(50, 50_000, size=n),              # int64
            "casos_fallecidos": rng.integers(0, 1_500, size=n),                 # int64
            "tasa_positividad": np.round(rng.uniform(0.01, 0.35, size=n), 3),   # float64
            "poblacion_millones": np.round(
                [PAISES_META[p][2] / 1_000_000 for p in paises], 1
            ),                                                                  # float64
            "cuarentena_activa": rng.choice([True, False], size=n),             # bool
        }
    )
    return df


@st.cache_data(show_spinner=True)
def generar_dataset_covid(n_rows: int, seed: int) -> pd.DataFrame:
    """
    Genera el dataset sintético principal (>= 3000 registros por defecto)
    usado para el análisis cuantitativo y los gráficos interactivos.
    """
    rng = np.random.default_rng(seed)
    total_dias = (END_DATE - START_DATE).days

    dias_offset = rng.integers(0, total_dias, size=n_rows)
    fechas = pd.to_datetime(START_DATE) + pd.to_timedelta(dias_offset, unit="D")
    paises = rng.choice(PAISES, size=n_rows, replace=True)

    continentes = np.array([PAISES_META[p][0] for p in paises])
    niveles_ingreso = np.array([PAISES_META[p][1] for p in paises])
    poblaciones = np.array([PAISES_META[p][2] for p in paises], dtype=float)

    # Factor de "onda epidémica" para que los casos varíen de forma no trivial
    # a lo largo del tiempo (para que las series temporales tengan forma).
    fase = (dias_offset % 365) / 365.0
    onda = 0.5 + 0.5 * np.sin(2 * np.pi * fase * 3)

    incidencia_base = rng.uniform(0.00005, 0.0025, size=n_rows)
    casos_nuevos = np.maximum(
        0, rng.poisson(poblaciones * incidencia_base * (0.3 + onda))
    ).astype(int)

    tasa_letalidad = rng.uniform(0.003, 0.04, size=n_rows)
    fallecidos_nuevos = np.maximum(
        0, rng.poisson(casos_nuevos * tasa_letalidad + 0.01)
    ).astype(int)

    tasa_recuperacion = rng.uniform(0.75, 0.97, size=n_rows)
    recuperados_nuevos = np.maximum(
        0, (casos_nuevos * tasa_recuperacion).astype(int)
    )

    tasa_positividad = np.round(rng.beta(2, 8, size=n_rows), 3)
    vacunados_pct = np.round(rng.uniform(0, 95, size=n_rows), 1)
    pruebas_realizadas = np.maximum(
        casos_nuevos, (casos_nuevos / np.maximum(tasa_positividad, 0.01)).astype(int)
    )

    df = pd.DataFrame(
        {
            "fecha": fechas,
            "pais": paises,
            "continente": pd.Categorical(continentes),
            "nivel_ingreso": pd.Categorical(
                niveles_ingreso, categories=["Bajo", "Medio", "Alto"], ordered=True
            ),
            "poblacion": poblaciones.astype(int),
            "casos_nuevos": casos_nuevos,
            "fallecidos_nuevos": fallecidos_nuevos,
            "recuperados_nuevos": recuperados_nuevos,
            "tasa_positividad": tasa_positividad,
            "vacunados_pct": vacunados_pct,
            "pruebas_realizadas": pruebas_realizadas,
        }
    )
    return df.sort_values("fecha").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Esquema de métricas / transformaciones cuantitativas
# ----------------------------------------------------------------------------
def construir_metricas(df: pd.DataFrame) -> dict:
    """
    A partir del dataset filtrado, construye las tablas transformadas que
    alimentan los gráficos: series temporales (con acumulados y medias
    móviles), agregados por país / continente, y matriz de correlación.
    """
    # --- Serie temporal diaria ---
    diario = (
        df.groupby("fecha", as_index=False)
        .agg(
            casos_nuevos=("casos_nuevos", "sum"),
            fallecidos_nuevos=("fallecidos_nuevos", "sum"),
            recuperados_nuevos=("recuperados_nuevos", "sum"),
        )
        .sort_values("fecha")
    )
    diario["casos_acumulados"] = diario["casos_nuevos"].cumsum()
    diario["fallecidos_acumulados"] = diario["fallecidos_nuevos"].cumsum()
    diario["media_movil_7d"] = diario["casos_nuevos"].rolling(7, min_periods=1).mean()

    # --- Agregado por país ---
    por_pais = (
        df.groupby("pais", as_index=False)
        .agg(
            casos_totales=("casos_nuevos", "sum"),
            fallecidos_totales=("fallecidos_nuevos", "sum"),
            recuperados_totales=("recuperados_nuevos", "sum"),
            poblacion=("poblacion", "max"),
            vacunados_pct_prom=("vacunados_pct", "mean"),
            continente=("continente", "first"),
        )
    )
    por_pais["tasa_mortalidad_%"] = np.round(
        100 * por_pais["fallecidos_totales"] / por_pais["casos_totales"].replace(0, np.nan), 2
    )
    por_pais["casos_por_millon"] = np.round(
        por_pais["casos_totales"] / (por_pais["poblacion"] / 1_000_000), 1
    )
    por_pais = por_pais.sort_values("casos_totales", ascending=False)

    # --- Agregado por continente ---
    por_continente = (
        df.groupby("continente", as_index=False)
        .agg(
            casos_totales=("casos_nuevos", "sum"),
            fallecidos_totales=("fallecidos_nuevos", "sum"),
        )
    )

    # --- Matriz de correlación de variables numéricas ---
    num_cols = [
        "casos_nuevos",
        "fallecidos_nuevos",
        "recuperados_nuevos",
        "tasa_positividad",
        "vacunados_pct",
        "pruebas_realizadas",
    ]
    correlacion = df[num_cols].corr()

    # --- KPIs globales ---
    kpis = {
        "casos_totales": int(df["casos_nuevos"].sum()),
        "fallecidos_totales": int(df["fallecidos_nuevos"].sum()),
        "recuperados_totales": int(df["recuperados_nuevos"].sum()),
        "tasa_mortalidad_global": round(
            100 * df["fallecidos_nuevos"].sum() / max(df["casos_nuevos"].sum(), 1), 2
        ),
        "paises_registrados": df["pais"].nunique(),
        "vacunacion_promedio": round(df["vacunados_pct"].mean(), 1),
    }

    return {
        "diario": diario,
        "por_pais": por_pais,
        "por_continente": por_continente,
        "correlacion": correlacion,
        "kpis": kpis,
    }


# ----------------------------------------------------------------------------
# 3. Interfaz de usuario
# ----------------------------------------------------------------------------
def main():
    st.title("🦠 Dashboard de datos sintéticos de COVID-19")
    st.caption(
        "Todos los datos son simulados dentro de esta misma aplicación con "
        "fines demostrativos: no provienen de ninguna fuente oficial."
    )

    # --- Estado de sesión para semillas (permite regenerar datos) ---
    if "seed_muestra" not in st.session_state:
        st.session_state.seed_muestra = 42
    if "seed_dataset" not in st.session_state:
        st.session_state.seed_dataset = 7

    # ---------------- Sidebar: controles de simulación y filtros ----------------
    st.sidebar.header("⚙️ Simulación de datos")
    n_rows = st.sidebar.slider(
        "Cantidad de registros a simular",
        min_value=3000,
        max_value=20000,
        value=3000,
        step=500,
        help="Tamaño del dataset principal usado en métricas y gráficos.",
    )

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("🔄 Regenerar dataset"):
        st.session_state.seed_dataset = np.random.randint(0, 1_000_000)
    if col_b.button("🔄 Regenerar muestra"):
        st.session_state.seed_muestra = np.random.randint(0, 1_000_000)

    df_completo = generar_dataset_covid(n_rows, st.session_state.seed_dataset)

    st.sidebar.header("🔎 Filtros")
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(df_completo["fecha"].min().date(), df_completo["fecha"].max().date()),
        min_value=df_completo["fecha"].min().date(),
        max_value=df_completo["fecha"].max().date(),
    )
    paises_sel = st.sidebar.multiselect(
        "Países", options=PAISES, default=PAISES
    )
    continentes_sel = st.sidebar.multiselect(
        "Continentes",
        options=sorted(df_completo["continente"].unique()),
        default=sorted(df_completo["continente"].unique()),
    )

    # --- Aplicar filtros ---
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        f_ini, f_fin = rango_fechas
    else:
        f_ini, f_fin = df_completo["fecha"].min().date(), df_completo["fecha"].max().date()

    mask = (
        (df_completo["fecha"].dt.date >= f_ini)
        & (df_completo["fecha"].dt.date <= f_fin)
        & (df_completo["pais"].isin(paises_sel))
        & (df_completo["continente"].isin(continentes_sel))
    )
    df = df_completo.loc[mask].copy()

    if df.empty:
        st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros en la barra lateral.")
        st.stop()

    metricas = construir_metricas(df)
    kpis = metricas["kpis"]

    # ---------------- KPIs ----------------
    st.subheader("📌 Indicadores clave")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Casos totales", f"{kpis['casos_totales']:,}")
    k2.metric("Fallecidos totales", f"{kpis['fallecidos_totales']:,}")
    k3.metric("Recuperados totales", f"{kpis['recuperados_totales']:,}")
    k4.metric("Tasa de mortalidad", f"{kpis['tasa_mortalidad_global']}%")
    k5.metric("Vacunación promedio", f"{kpis['vacunacion_promedio']}%")

    st.divider()

    tab_resumen, tab_distribucion, tab_correlacion, tab_datos = st.tabs(
        ["📈 Resumen temporal", "🌍 Distribución", "🔬 Correlaciones", "🗂️ Datos"]
    )

    # ---------------- Tab: Resumen temporal ----------------
    with tab_resumen:
        st.markdown("#### Evolución de casos en el tiempo")
        mostrar_media_movil = st.checkbox("Mostrar media móvil (7 días)", value=True)

        diario = metricas["diario"]
        fig_linea = go.Figure()
        fig_linea.add_trace(
            go.Scatter(
                x=diario["fecha"], y=diario["casos_nuevos"],
                mode="lines", name="Casos nuevos", line=dict(color="#636EFA"),
            )
        )
        if mostrar_media_movil:
            fig_linea.add_trace(
                go.Scatter(
                    x=diario["fecha"], y=diario["media_movil_7d"],
                    mode="lines", name="Media móvil 7d",
                    line=dict(color="#EF553B", width=3),
                )
            )
        fig_linea.update_layout(
            xaxis_title="Fecha", yaxis_title="Casos", hovermode="x unified"
        )
        st.plotly_chart(fig_linea, use_container_width=True)

        metrica_acum = st.radio(
            "Métrica acumulada a graficar:",
            ["casos_acumulados", "fallecidos_acumulados"],
            horizontal=True,
        )
        fig_area = px.area(
            diario, x="fecha", y=metrica_acum,
            title="Acumulado en el tiempo",
        )
        st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("#### Top países por casos totales")
        top_n = st.slider("Número de países a mostrar", 3, len(PAISES), 10)
        por_pais = metricas["por_pais"].head(top_n)
        fig_bar = px.bar(
            por_pais.sort_values("casos_totales"),
            x="casos_totales", y="pais", orientation="h",
            color="continente", text="casos_totales",
            title="Casos totales por país",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- Tab: Distribución ----------------
    with tab_distribucion:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Distribución de casos por continente")
            fig_pie = px.pie(
                metricas["por_continente"], names="continente", values="casos_totales",
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Población vs. casos totales")
            escala_log = st.checkbox("Escala logarítmica (eje X)", value=True)
            fig_scatter = px.scatter(
                metricas["por_pais"],
                x="poblacion", y="casos_totales",
                size="fallecidos_totales", color="continente",
                hover_name="pais", log_x=escala_log,
                size_max=45,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Vacunación (%) por continente")
        fig_box = px.box(
            df, x="continente", y="vacunados_pct", color="continente", points="outliers",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ---------------- Tab: Correlaciones ----------------
    with tab_correlacion:
        st.markdown("#### Matriz de correlación (variables numéricas)")
        fig_heat = px.imshow(
            metricas["correlacion"], text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", zmin=-1, zmax=1,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Distribución de la tasa de positividad")
        color_por = st.selectbox("Colorear por:", ["nivel_ingreso", "continente"])
        n_bins = st.slider("Número de bins", 10, 80, 30)
        fig_hist = px.histogram(
            df, x="tasa_positividad", color=color_por, nbins=n_bins, barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------- Tab: Datos ----------------
    with tab_datos:
        st.markdown("#### 🧪 Muestra interactiva (10 registros x 8 columnas)")
        st.caption(
            "Puedes editar los valores directamente en la tabla o pulsar "
            "'Regenerar muestra' en la barra lateral para simular una nueva."
        )
        df_muestra = generar_muestra_interactiva(st.session_state.seed_muestra)
        st.data_editor(df_muestra, use_container_width=True, num_rows="dynamic", key="editor_muestra")
        st.caption(f"Tipos de dato: {df_muestra.dtypes.astype(str).to_dict()}")

        st.markdown("#### 📦 Dataset principal simulado")
        st.write(f"Registros luego de filtros: **{len(df):,}** de **{len(df_completo):,}** generados en total.")
        st.dataframe(df.head(200), use_container_width=True)

        st.download_button(
            "⬇️ Descargar dataset filtrado (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="covid_sintetico.csv",
            mime="text/csv",
        )

        with st.expander("Ver tabla de métricas por país"):
            st.dataframe(metricas["por_pais"], use_container_width=True)


if __name__ == "__main__":
    main()
