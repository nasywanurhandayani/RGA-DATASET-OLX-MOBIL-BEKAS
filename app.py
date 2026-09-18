# -*- coding: utf-8 -*-
"""
app.py — Conversational BI: Dashboard + AI Chatbot RAG untuk data mobil OLX

Cara jalanin:
    streamlit run app.py

Struktur folder yang dibutuhkan (satu folder yang sama dengan app.py ini):
    dataset_olx_rag_ready.csv
    chroma_db_olx/           <- folder vector DB hasil dari Colab
"""

import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="OLX Mobil — Conversational BI",
    page_icon="🚗",
    layout="wide",
)

CSV_PATH = "dataset_olx_rag_ready.csv"
CHROMA_DIR = "./chroma_db_olx"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# LOAD DATA (di-cache biar gak reload tiap interaksi)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


@st.cache_resource
def load_vectorstore():
    """Load vector DB + embedding model yang udah dibuat di Colab."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore


@st.cache_resource
def load_gemini():
    """Setup Gemini API. API key diambil dari environment variable atau Streamlit secrets."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai


# ============================================================
# FUNGSI RAG
# ============================================================
def tanya_olx_bot(query_user, vectorstore, genai_module, k=3):
    """Retrieval-Augmented Generation: cari data relevan lalu tanya ke Gemini."""
    results = vectorstore.similarity_search(query_user, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Kamu adalah AI Sales Assistant cerdas untuk platform jual-beli mobil OLX.
Jawablah pertanyaan user berdasarkan HANYA data relevan yang diberikan di bawah ini.

DATA RELEVAN DARI OLX:
{context_text}

PERTANYAAN USER:
{query_user}

PETUNJUK JAWABAN:
1. Berikan rekomendasi unit mobil yang ada di data konteks secara ramah dan profesional.
2. Sebutkan spesifikasi penting seperti Harga, Tahun, Transmisi, KM, dan info Pajak/Plat jika ada.
3. Jika data tidak ditemukan di konteks, katakan dengan jujur bahwa unit tersebut belum tersedia di database.
"""

    model = genai_module.GenerativeModel("gemini-3.6-flash")
    response = model.generate_content(prompt)
    return response.text, results


# ============================================================
# LOAD SEMUA RESOURCE
# ============================================================
df = load_data()

st.title("🚗 OLX Mobil — Dashboard & AI Chatbot")
st.caption(f"Dataset: {len(df):,} listing mobil dari OLX")

# Layout 2 kolom: kiri dashboard, kanan chatbot
col_dashboard, col_chat = st.columns([1.3, 1])

# ============================================================
# KOLOM KIRI: DASHBOARD INTERAKTIF
# ============================================================
with col_dashboard:
    st.header("📊 Dashboard")

    # --- FILTER ---
    with st.expander("🔎 Filter", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            merek_options = ["Semua"] + sorted(df["merek"].dropna().unique().tolist())
            merek_pilih = st.selectbox("Merek", merek_options)
        with f2:
            harga_min, harga_max = int(df["harga"].min()), int(df["harga"].max())
            rentang_harga = st.slider(
                "Rentang Harga (Rp)",
                min_value=harga_min,
                max_value=harga_max,
                value=(harga_min, harga_max),
                step=1_000_000,
                format="%d",
            )

    # Terapkan filter
    df_filtered = df[(df["harga"] >= rentang_harga[0]) & (df["harga"] <= rentang_harga[1])]
    if merek_pilih != "Semua":
        df_filtered = df_filtered[df_filtered["merek"] == merek_pilih]

    # --- METRIC RINGKAS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah Unit", f"{len(df_filtered):,}")
    m2.metric("Harga Rata-rata", f"Rp {df_filtered['harga'].mean():,.0f}" if len(df_filtered) else "-")
    m3.metric("Tahun Termuda", f"{int(df_filtered['tahun'].max())}" if len(df_filtered) else "-")

    # --- GRAFIK DISTRIBUSI HARGA ---
    st.subheader("Distribusi Harga")
    fig_harga = px.histogram(
        df_filtered, x="harga", nbins=30,
        labels={"harga": "Harga (Rp)"},
        color_discrete_sequence=["#C81D4F"],
    )
    fig_harga.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_harga, use_container_width=True)

    # --- GRAFIK JUMLAH UNIT PER MEREK ---
    st.subheader("Jumlah Unit per Merek (Top 10)")
    top_merek = df_filtered["merek"].value_counts().head(10).reset_index()
    top_merek.columns = ["merek", "jumlah"]
    fig_merek = px.bar(
        top_merek, x="jumlah", y="merek", orientation="h",
        color_discrete_sequence=["#FF5A6E"],
    )
    fig_merek.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_merek, use_container_width=True)

    # --- STATUS PAJAK (hasil ekstraksi Regex) ---
    if "status_pajak" in df_filtered.columns:
        st.subheader("Status Pajak")
        pajak_count = df_filtered["status_pajak"].value_counts().reset_index()
        pajak_count.columns = ["status", "jumlah"]
        fig_pajak = px.pie(
            pajak_count, names="status", values="jumlah", hole=0.45,
            color_discrete_sequence=px.colors.sequential.RdPu,
        )
        fig_pajak.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pajak, use_container_width=True)

    # --- TABEL DATA ---
    with st.expander("📄 Lihat Data Mentah"):
        st.dataframe(
            df_filtered[["judul", "merek", "model", "tahun", "transmisi", "jarak_tempuh", "harga"]],
            use_container_width=True,
            height=300,
        )

# ============================================================
# KOLOM KANAN: AI CHATBOT RAG
# ============================================================
with col_chat:
    st.header("💬 Tanya AI")
    st.caption("Chatbot ini nyari data mobil paling relevan lalu jawab pake Gemini.")

    genai_module = load_gemini()
    if genai_module is None:
        st.warning(
            "GEMINI_API_KEY belum diset. Set environment variable `GEMINI_API_KEY` "
            "atau tambahin di `.streamlit/secrets.toml` sebelum chatbot bisa dipakai."
        )
    else:
        vectorstore = load_vectorstore()

        # Inisialisasi riwayat chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Tampilkan riwayat chat
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input pertanyaan baru
        pertanyaan = st.chat_input("Contoh: Cariin Honda Brio matic pajak hidup di bawah 170 juta")
        if pertanyaan:
            st.session_state.chat_history.append({"role": "user", "content": pertanyaan})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(pertanyaan)

                with st.chat_message("assistant"):
                    with st.spinner("Nyari data & nyusun jawaban..."):
                        jawaban, sumber = tanya_olx_bot(pertanyaan, vectorstore, genai_module)
                        st.markdown(jawaban)
                        with st.expander("Lihat data sumber"):
                            for i, doc in enumerate(sumber):
                                st.text(f"--- Sumber {i+1} ---")
                                st.text(doc.page_content[:400] + "...")

            st.session_state.chat_history.append({"role": "assistant", "content": jawaban})

        if st.button("🗑️ Reset Percakapan"):
            st.session_state.chat_history = []
            st.rerun()
