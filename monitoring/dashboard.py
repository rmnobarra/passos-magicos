"""
Dashboard Streamlit para monitoramento do modelo em produção.
Exibe métricas de uso (via API /metrics), logs de predições e análise de drift.
"""
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Garante que src/ e monitoring/ são encontrados mesmo dentro do container
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Passos Mágicos — Monitor ML",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Passos Mágicos — Monitoramento do Modelo de Defasagem")
st.caption(f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

API_URL = os.getenv("API_URL", "http://localhost:8000")
LOG_PATH = Path("logs/predictions.jsonl")
REFERENCE_DATA = Path("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")


# ── Funções auxiliares ────────────────────────────────────────────────────────

def fetch_metrics() -> dict:
    """Busca métricas do endpoint /api/v1/metrics da API."""
    try:
        with urllib.request.urlopen(f"{API_URL}/api/v1/metrics", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def load_predictions() -> pd.DataFrame:
    """Carrega histórico de predições do arquivo JSONL."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(records) if records else pd.DataFrame()


def run_drift_detection() -> dict:
    """
    Executa análise de drift usando os dados de referência disponíveis.
    Roda diretamente in-process para evitar dependência de CLI externo.
    """
    from src.preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from monitoring.drift_detector import DriftDetector

    logging.disable(logging.CRITICAL)  # silencia logs durante a análise
    try:
        dp = DataPreprocessor()
        fe = FeatureEngineer()
        ref_df = fe.transform(dp.fit_transform(dp.load_data(str(REFERENCE_DATA))))
        feature_cols = [
            c for c in ref_df.select_dtypes(include="number").columns
            if c != "DEFASAGEM"
        ]
        reference_data = ref_df[feature_cols].dropna()
        current_data = reference_data.sample(frac=0.3, random_state=99)
        detector = DriftDetector.from_dataframe(reference_data)
        return detector.detect_drift(current_data)
    finally:
        logging.disable(logging.NOTSET)


def latest_drift_report() -> Path | None:
    """Retorna o relatório de drift mais recente, ou None se não houver."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    reports = list(reports_dir.glob("drift_report_*.html"))
    return max(reports, key=lambda p: p.stat().st_mtime) if reports else None


# ── KPIs ─────────────────────────────────────────────────────────────────────

metricas = fetch_metrics()
col1, col2, col3, col4 = st.columns(4)

if metricas:
    total = metricas.get("total_predicoes", 0)
    alto = metricas.get("predicoes_alto_risco", 0)
    col1.metric("Total de Predições", total)
    col2.metric("Em Risco de Defasagem", alto)
    col3.metric("Taxa de Risco", f"{alto/total:.1%}" if total > 0 else "—")
    col4.metric("Última Predição", metricas.get("ultima_predicao", "—"))
else:
    col1.metric("Total de Predições", 0)
    col2.metric("Em Risco", 0)
    col3.metric("Taxa de Risco", "—")
    col4.metric("Última Predição", "—")
    st.warning("API indisponível — KPIs não carregados.")

st.divider()

# ── Análise de Drift ──────────────────────────────────────────────────────────

st.subheader("📊 Análise de Drift do Modelo")

latest = latest_drift_report()
col_status, col_btn = st.columns([4, 1])

with col_status:
    if latest:
        mtime = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        st.success(f"✅ Último relatório gerado em {mtime} — `{latest.name}`")
    else:
        st.info("Nenhum relatório gerado ainda. Clique em **Gerar Relatório** para analisar.")

with col_btn:
    gerar = st.button("🔍 Gerar Relatório", disabled=not REFERENCE_DATA.exists())

if not REFERENCE_DATA.exists():
    st.caption("⚠️ Dados de referência não encontrados em `data/raw/`.")

if gerar:
    with st.spinner("Analisando distribuição das features... aguarde."):
        try:
            result = run_drift_detection()
            latest = latest_drift_report()
            if result["drift_detected"]:
                st.error(
                    f"⚠️ **Drift detectado!** Score: {result['drift_score']:.1%} — "
                    f"Features: {', '.join(result['drifted_features']) or 'N/A'}"
                )
            else:
                st.success(
                    f"✅ **Sem drift significativo.** Score: {result['drift_score']:.1%}"
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Erro ao gerar relatório: {exc}")

if latest:
    html_content = latest.read_text(encoding="utf-8")
    with st.expander("📄 Ver Relatório de Drift", expanded=True):
        components.html(html_content, height=800, scrolling=True)
    st.download_button(
        "📥 Baixar Relatório HTML",
        html_content,
        file_name=latest.name,
        mime="text/html",
    )

st.divider()

# ── Predições recentes ────────────────────────────────────────────────────────

df_preds = load_predictions()
if not df_preds.empty:
    st.subheader("📋 Predições Recentes")
    st.dataframe(df_preds.tail(50).iloc[::-1], use_container_width=True)
else:
    st.info("Nenhuma predição registrada ainda. Faça chamadas à API para ver os dados aqui.")
