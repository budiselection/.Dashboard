import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy.stats import chi2_contingency, kruskal, gaussian_kde

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Opsis · Skrining Risiko Glaukoma",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* Root variables */
:root {
    --deep-teal:    #0d4f6e;
    --mid-teal:     #1a7a9e;
    --light-teal:   #e8f4f8;
    --accent-coral: #e05c5c;
    --accent-amber: #e8a040;
    --accent-jade:  #2da673;
    --neutral-900:  #1a1f2e;
    --neutral-700:  #3d4460;
    --neutral-400:  #8890a4;
    --neutral-100:  #f5f6fa;
    --white:        #ffffff;
    --card-shadow:  0 2px 12px rgba(13,79,110,0.10);
    --card-radius:  14px;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* App header banner */
.app-header {
    background: linear-gradient(135deg, var(--deep-teal) 0%, var(--mid-teal) 100%);
    border-radius: var(--card-radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: "👁";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.08;
    pointer-events: none;
}
.app-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: var(--white);
    margin: 0;
    line-height: 1.15;
}
.app-header-sub {
    color: rgba(255,255,255,0.78);
    font-size: 0.95rem;
    margin-top: 0.35rem;
    font-weight: 400;
}
.header-badge {
    background: rgba(255,255,255,0.15);
    color: white;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
}

/* Metric cards - FIXED: Explicit dark text for light background */
.metric-card {
    background: var(--white);
    border-radius: var(--card-radius);
    padding: 1.2rem 1.4rem;
    box-shadow: var(--card-shadow);
    border-left: 4px solid var(--mid-teal);
    margin-bottom: 0.5rem;
    color: #1a1f2e !important;
}
.metric-card-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1f2e !important;
    line-height: 1;
}
.metric-card-label {
    font-size: 0.82rem;
    color: #3d4460 !important;
    font-weight: 500;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card-delta {
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 0.15rem;
}

/* Risk result card - FIXED: Explicit dark text for pastel backgrounds */
.risk-card {
    border-radius: var(--card-radius);
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    box-shadow: var(--card-shadow);
    color: #1a1f2e !important;
}
.risk-card-high   { background: linear-gradient(135deg, #fff0f0, #ffe0e0); border-left: 5px solid var(--accent-coral); }
.risk-card-medium { background: linear-gradient(135deg, #fff8ec, #ffecd0); border-left: 5px solid var(--accent-amber); }
.risk-card-low    { background: linear-gradient(135deg, #edfaf3, #d5f0e4); border-left: 5px solid var(--accent-jade); }
.risk-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #1a1f2e !important;
}
.risk-advice {
    font-size: 0.95rem;
    color: #3d4460 !important;
    line-height: 1.55;
}

/* Input section label - FIXED: Brighter teal for Dark Mode visibility */
.input-section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #219ebc !important;
    margin-bottom: 0.6rem;
    padding-top: 0.3rem;
}

/* Guide cards - FIXED: Explicit dark text */
.guide-card {
    background: var(--white);
    border-radius: var(--card-radius); 
    padding: 1.25rem 1.5rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 1rem;
    border-top: 3px solid var(--mid-teal);
    color: #1a1f2e !important;
}
.guide-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0d4f6e !important;
    margin-bottom: 0.35rem;
}
.guide-card-body {
    font-size: 0.91rem;
    color: #3d4460 !important;
    line-height: 1.6;
}
.guide-tip {
    background: var(--light-teal);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.88rem;
    color: var(--deep-teal);
    margin-top: 0.5rem;
    font-style: italic;
}
.guide-warning {
    background: #fff8ec;
    border-left: 3px solid var(--accent-amber);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.88rem;
    color: #7a5200;
    margin-top: 0.5rem;
}

/* Alert banners */
.info-banner {
    background: var(--light-teal);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.92rem;
    color: var(--deep-teal);
    border-left: 4px solid var(--mid-teal);
    margin: 0.8rem 0;
}
.disclaimer-banner {
    background: #fff8ec;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.88rem;
    color: #7a5200;
    border-left: 4px solid var(--accent-amber);
    margin: 0.8rem 0;
}

/* Tab styling override */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--neutral-100);
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--neutral-700);
    background: transparent;
}
/* Efek saat kursor di-hover ke tab */
.stTabs [data-baseweb="tab"]:hover {
    color: #e05c5c !important;
    background: transparent !important;
}
/* Tab yang sedang aktif 
.stTabs [aria-selected="true"] {
    background: transparent !important; 
    color: #e05c5c !important;          
    font-weight: 700;
    box-shadow: none !important;        
    border-bottom: 2px solid #e05c5c !important; 
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d4f6e 0%, #1a7a9e 100%);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.95) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.15) !important;
    border-color: rgba(255,255,255,0.3) !important;
    color: white !important;
}

/* Section divider */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--mid-teal), transparent);
    border: none;
    margin: 1.5rem 0;
    border-radius: 2px;
}

/* Probability bar */
.prob-bar-wrap { margin: 0.4rem 0; }
.prob-bar-label {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
}
.prob-bar-bg {
    background: #eef0f5;
    border-radius: 99px;
    height: 12px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.4s ease;
}

/* Active factor tag */
.factor-tag {
    display: inline-block;
    background: #fff0f0;
    color: var(--accent-coral);
    border: 1px solid #f5c0c0;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.factor-tag-none {
    background: #edfaf3;
    color: var(--accent-jade);
    border-color: #b0e8cc;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--neutral-400);
    font-size: 0.82rem;
    border-top: 1px solid #e8eaf2;
    margin-top: 3rem;
}
/* FIX: Paksa warna teks gelap di dalam kotak 'Mengapa Penting?' agar terbaca di Dark Mode */
div[data-testid="stExpander"] .box-penting,
div[data-testid="stExpander"] .box-penting * {
    color: #3d4460 !important;
}
div[data-testid="stExpander"] .judul-penting {
    color: #1a1f2e !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ──────────────────────────────────────────────────────────────────
RISK_PALETTE   = {"Risiko Rendah": "#2da673", "Risiko Sedang": "#e8a040", "Risiko Tinggi": "#e05c5c"}
RISK_ORDER     = ["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"]
RISK_COLORS    = [RISK_PALETTE[r] for r in RISK_ORDER]
KORIKO_ORDER   = ["Tidak", "Jangka Pendek", "Jangka Panjang"]
KELOMPOK_ORDER = ["< 30", "30–44", "45–59", "60+"]
REFRAKSI_ORDER = [
    "Tidak Ada", "Miopia Ringan", "Miopia Sedang/Berat",
    "Hipermetropia Ringan", "Hipermetropia Sedang/Berat",
]
FAKTOR_BINER = ["Riwayat_Keluarga", "Diabetes", "Hipertensi", "Migrain", "Gangguan_Sirkulasi"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f5f6fa",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#d0d4e0",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})

# ── Load data & model ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource
def load_model(file):
    return joblib.load(file)

@st.cache_resource
def load_encoder(file):
    return joblib.load(file)

# ── Encode & predict ───────────────────────────────────────────────────────────
def encode_input(pasien: dict) -> pd.DataFrame:
    usia_scaled = (pasien["Usia"] - 20) / (85 - 20)
    gender_enc  = 1 if pasien["Jenis_Kelamin"] == "Laki-laki" else 0
    biner = {col: 1 if pasien[col] == "Ya" else 0
             for col in ["Riwayat_Keluarga", "Diabetes", "Hipertensi", "Migrain", "Gangguan_Sirkulasi"]}
    
    map_refraksi = {
        "Tidak Ada": 0, "Miopia Ringan": 1, "Hipermetropia Ringan": 1,
        "Miopia Sedang/Berat": 2, "Hipermetropia Sedang/Berat": 2,
    }
    map_koriko   = {"Tidak": 0, "Jangka Pendek": 1, "Jangka Panjang": 2}
    
    refraksi_enc = map_refraksi.get(pasien["Kelainan_Refraksi"], 0)
    koriko_enc   = map_koriko.get(pasien["Penggunaan_Kortikosteroid"], 0)
    jumlah_komorbid = sum(biner.values())
    
    risiko_komposit = round(
        (koriko_enc / 2) * 0.30 + biner["Hipertensi"] * 0.20 +
        biner["Riwayat_Keluarga"] * 0.18 + biner["Diabetes"] * 0.15 +
        biner["Migrain"] * 0.10 + biner["Gangguan_Sirkulasi"] * 0.07, 4,
    )
    
    return pd.DataFrame([{
        "Usia": usia_scaled, "Jenis_Kelamin": gender_enc,
        "Riwayat_Keluarga": biner["Riwayat_Keluarga"], "Diabetes": biner["Diabetes"],
        "Hipertensi": biner["Hipertensi"], "Migrain": biner["Migrain"],
        "Gangguan_Sirkulasi": biner["Gangguan_Sirkulasi"],
        "Kelainan_Refraksi": refraksi_enc, "Penggunaan_Kortikosteroid": koriko_enc,
        "Jumlah_Komorbid": jumlah_komorbid, "Risiko_Komposit": risiko_komposit,
    }])

def prediksi(X_input, artifact, le):
    model       = artifact["model"]
    label_names = artifact["label_names"]
    pred_enc    = int(model.predict(X_input)[0])
    pred_proba  = model.predict_proba(X_input)[0]
    label       = le.inverse_transform([pred_enc])[0]
    return label, pred_enc, {label_names[i]: round(float(p) * 100, 2) for i, p in enumerate(pred_proba)}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.5rem; color: white; line-height:1.2;'>👁 Opsis</div>
        <div style='font-size: 0.82rem; color: rgba(255,255,255,0.7); margin-top:0.2rem;'>Dashboard Risiko Glaukoma</div>
    </div>
    """, unsafe_allow_html=True)

    up_data    = "dataset_glaukoma_enriched.csv"
    up_model   = "model_glaukoma.joblib"
    up_encoder = "label_encoder.joblib"

    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:rgba(255,255,255,0.6); margin-bottom:0.6rem;'>Filter Data Analitik</div>", unsafe_allow_html=True)

    if up_data:
        df_raw     = load_data(up_data)
        gender_opt = ["Semua"] + sorted(df_raw["Jenis_Kelamin"].unique().tolist())
        risk_opt   = ["Semua"] + RISK_ORDER
        koriko_opt = ["Semua"] + KORIKO_ORDER
        
        sel_gender = st.selectbox("Jenis Kelamin", gender_opt)
        sel_risk   = st.selectbox("Label Risiko", risk_opt)
        sel_koriko = st.selectbox("Kortikosteroid", koriko_opt)
        
        usia_min   = int(df_raw["Usia"].min())
        usia_max   = int(df_raw["Usia"].max())
        sel_usia   = st.slider("Rentang Usia", usia_min, usia_max, (usia_min, usia_max))

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:rgba(255,255,255,0.55); line-height:1.5;'>
        Proyek Capstone · Coding Camp 2026<br>
        DBS Foundation × Dicoding<br><br>
        <b style='color:rgba(255,255,255,0.8);'>Tim Opsis · Data Scientist</b>
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
def apply_filters(df):
    out = df.copy()
    if sel_gender != "Semua": out = out[out["Jenis_Kelamin"] == sel_gender]
    if sel_risk   != "Semua": out = out[out["Label_Risiko"] == sel_risk]
    if sel_koriko != "Semua": out = out[out["Penggunaan_Kortikosteroid"] == sel_koriko]
    out = out[(out["Usia"] >= sel_usia[0]) & (out["Usia"] <= sel_usia[1])]
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-header-title">Opsis · Skrining Risiko Glaukoma</div>
        <div class="app-header-sub">Sistem skrining risiko berbasis data klinis & kuesioner — didukung model <b>Random Forest</b> dengan akurasi 97,67%</div>
        <div>
            <span class="header-badge">⚠ Bukan Pengganti Diagnosis Medis</span>
            <span class="header-badge" style="margin-left:0.4rem;">📊 Dataset Sintetis · 1.500 Sampel</span>
            <span class="header-badge" style="margin-left:0.4rem;">🏥 Capstone Coding Camp 2026</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not up_data:
    st.info("⬅️ Upload file di sidebar untuk memulai.")
    st.stop()

df    = apply_filters(df_raw)
total = len(df)
dist  = df["Label_Risiko"].value_counts().reindex(RISK_ORDER, fill_value=0)

if total == 0:
    st.warning("⚠️ Tidak ada data sesuai filter.")
    st.stop()

# ── Tab navigasi ───────────────────────────────────────────────────────────────
(tab_guide, tab_pred, tab_dist, tab_faktor, tab_usia,
 tab_kombinasi, tab_koriko, tab_gender, tab_refraksi) = st.tabs([
    "📖 Panduan", "🩺 Skrining Risiko", "📊 Distribusi Data", "🔴 BQ-1 Faktor",
    "📅 BQ-2 Usia", "⚡ BQ-3 Kombinasi", "💊 BQ-4 Kortikosteroid",
    "👥 BQ-5 Gender", "👁 BQ-6 Refraksi",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB PANDUAN (GUIDE BOOK)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    col_g1, col_g2 = st.columns([1.1, 0.9], gap="large")

    with col_g1:
        st.markdown("""
        <div style='font-family: DM Serif Display, serif; font-size:1.6rem; color:#17a2b8; margin-bottom:0.3rem;'>
            Panduan Penggunaan Dashboard
        </div>
        <div style='color:#8890a4; font-size:0.92rem; margin-bottom:1.2rem;'>
            Baca panduan ini sebelum mengisi formulir skrining, khususnya jika Anda bukan tenaga medis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-banner">
            ⚠️ <b>Penting:</b> Hasil dari dashboard ini <b>bukan diagnosis medis</b> dan tidak dapat menggantikan pemeriksaan langsung oleh dokter spesialis mata (oftalmolog). Gunakan hasil ini sebagai <b>sinyal awal</b> untuk mendorong Anda berkonsultasi ke dokter.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Apa itu Glaukoma?")
        st.markdown("""
        <div class="guide-card">
            <div class="guide-card-title">👁 Glaukoma: "Pencuri Penglihatan Diam-diam"</div>
            <div class="guide-card-body">
                Glaukoma adalah kelompok penyakit mata yang merusak saraf optik secara progresif. 
                Bahayanya: <b>tidak ada gejala di stadium awal</b> — penglihatan menurun perlahan tanpa terasa, hingga akhirnya bisa menyebabkan kebutaan permanen yang <b>tidak bisa disembuhkan</b>.<br><br>
                Deteksi dini adalah satu-satunya cara mencegah kebutaan. Sistem skrining ini membantu Anda mengetahui apakah Anda berisiko, agar segera memeriksakan diri ke dokter mata.
            </div>
            <div class="guide-tip">💡 WHO memproyeksikan 111,8 juta penderita glaukoma di dunia pada 2040. Mayoritas kasus terlambat terdeteksi karena tidak ada gejala awal. (Tham et al., 2014)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Cara Menggunakan Tab Skrining Risiko")
        steps = [
            ("1️⃣", "Buka tab **🩺 Skrining Risiko**", "Klik tab di atas untuk membuka formulir skrining."),
            ("2️⃣", "Isi semua kolom", "Isi data Anda sesuai panduan variabel di bawah. Tidak perlu hasil lab — cukup jawab berdasarkan riwayat yang Anda ketahui."),
            ("3️⃣", "Klik tombol Prediksi", "Sistem akan memproses data Anda menggunakan model Machine Learning dan menampilkan hasil beserta probabilitas."),
            ("4️⃣", "Baca rekomendasi", "Hasil akan disertai saran tindak lanjut — Risiko Rendah, Sedang, atau Tinggi."),
            ("5️⃣", "Konsultasikan ke dokter", "Jika hasil menunjukkan Risiko Sedang atau Tinggi, segera periksakan diri ke dokter mata."),
        ]
        for icon, title, body in steps:
            st.markdown(f"""
            <div style='display:flex; gap:0.9rem; margin-bottom:0.8rem; align-items:flex-start;'>
                <div style='font-size:1.5rem; line-height:1;'>{icon}</div>
                <div>
                    <div style='font-weight:700; color:#17a2b8; font-size:0.95rem;'>{title}</div>
                    <div style='font-size:0.88rem; margin-top:0.15rem;'>{body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("### Panduan Pengisian Setiap Variabel")
        st.markdown("<div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Klik tiap variabel untuk melihat definisi dan cara menjawabnya.</div>", unsafe_allow_html=True)

        var_guides = [
            ("👤 Usia", "#1a7a9e",
             "Masukkan usia Anda saat ini dalam satuan tahun (20–85 tahun).",
             "Risiko glaukoma meningkat signifikan mulai usia 40 tahun, dan melonjak setelah usia 60 tahun. Usia adalah prediktor terkuat dalam model ini.",
             "Worley & Grimmer-Somers (2011); Tham et al. (2014)"),
            ("🧬 Riwayat Keluarga", "#e05c5c",
             "Pilih 'Ya' jika orang tua, saudara kandung, atau anak kandung Anda pernah didiagnosis glaukoma oleh dokter.",
             "Memiliki kerabat tingkat pertama dengan glaukoma meningkatkan risiko Anda 3–10 kali lipat lebih tinggi dari populasi umum.",
             "OHTS Study (Kass et al., 2002) dikutip dalam Worley & Grimmer-Somers (2011); Grzybowski et al. (2020)"),
            ("🩺 Diabetes Melitus", "#e8a040",
             "Pilih 'Ya' jika Anda pernah atau sedang didiagnosis Diabetes Mellitus (DM Tipe 1 atau Tipe 2) oleh dokter.",
             "DM dapat meningkatkan tekanan intraokular (TIO) melalui perubahan permeabilitas pembuluh darah mata dan penumpukan matriks ekstraseluler akibat hiperglikemia.",
             "Zhou et al. (2014) dalam Costa et al. (2015); Grzybowski et al. (2020)"),
            ("❤️ Hipertensi", "#e05c5c",
             "Pilih 'Ya' jika pernah didiagnosis tekanan darah tinggi, atau sedang mengonsumsi obat antihipertensi secara rutin.",
             "Disfungsi vaskular akibat hipertensi dapat memengaruhi aliran darah ke saraf optik (tekanan perfusi okular) dan meningkatkan risiko kerusakan saraf.",
             "Grzybowski et al. (2020); Roberti et al. (2020)"),
            ("🧠 Migrain", "#8890a4",
             "Pilih 'Ya' jika Anda sering mengalami sakit kepala sebelah yang berulang dan sudah pernah didiagnosis migrain oleh dokter.",
             "Migrain dikaitkan dengan vasospasme (penyempitan pembuluh darah sementara) yang dapat mengurangi perfusi darah ke saraf optik, khususnya pada glaukoma tekanan normal.",
             "Worley & Grimmer-Somers (2011); Grzybowski et al. (2020)"),
            ("🩸 Gangguan Sirkulasi", "#1a7a9e",
             "Pilih 'Ya' jika Anda memiliki riwayat penyakit sirkulasi yang didiagnosis dokter: penyakit jantung koroner, peripheral vascular disease, sindrom Raynaud, atau gangguan aliran darah serupa.",
             "Kondisi ini mengurangi perfusi okular (aliran darah ke mata) dan meningkatkan risiko iskemia (kekurangan oksigen) pada saraf optik.",
             "Grzybowski et al. (2020); Worley & Grimmer-Somers (2011)"),
            ("👓 Kelainan Refraksi", "#2da673",
             "Pilih kategori berdasarkan kondisi penglihatan Anda dari hasil pemeriksaan dokter mata terakhir: Tidak Ada / Miopia (rabun jauh) Ringan atau Sedang/Berat / Hipermetropia (rabun dekat) Ringan atau Sedang/Berat.",
             "Miopia berat (> −3,0 D) meningkatkan risiko glaukoma karena perubahan struktural pada kepala saraf optik (optic disc). Jika belum pernah periksa, pilih 'Tidak Ada'.",
             "Grzybowski et al. (2020); Worley & Grimmer-Somers (2011)"),
            ("💊 Penggunaan Kortikosteroid", "#e05c5c",
             "Pilih sesuai riwayat penggunaan obat kortikosteroid (termasuk tablet, tetes mata, inhaler asma, atau salep yang mengandung steroid): Tidak / Jangka Pendek (< 4 minggu) / Jangka Panjang (≥ 4 minggu atau rutin).",
             "Kortikosteroid jangka panjang dapat menyebabkan steroid-induced glaucoma dengan meningkatkan resistensi aliran cairan aqueous humor dalam mata. Ini adalah salah satu faktor risiko yang paling dapat dicegah.",
             "Razeghinejad & Katz (2012); Varshney et al. (2023); Roberti et al. (2020)"),
        ]

        for title, color, how_to, why, ref in var_guides:
            with st.expander(title):
                st.markdown(f"""
                <div style='margin-bottom:0.5rem;'>
                    <div style='font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:{color} !important; margin-bottom:0.25rem;'>Cara Menjawab</div>
                    <div style='font-size:0.9rem; color:#1a1f2e !important; line-height:1.5;'>{how_to}</div>
                </div>
                
                <div class="box-penting" style='background:#f5f6fa; border-radius:8px; padding:0.75rem; margin-top:0.5rem;'>
                    <div class="judul-penting" style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.25rem;'>Mengapa Penting?</div>
                    <div style='font-size:0.88rem; line-height:1.6;'>{why}</div>
                </div>
                
                <div style='font-size:0.78rem; color:#8890a4 !important; margin-top:0.5rem; font-style:italic;'>📚 Referensi: {ref}</div>
                """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-banner" style='margin-top:1rem;'>
            📌 <b>Catatan:</b> Jika Anda tidak memiliki hasil pemeriksaan dokter untuk suatu variabel, pilih opsi yang paling mendekati kondisi Anda saat ini, atau pilih "Tidak" / "Tidak Ada" sebagai default.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB SKRINING RISIKO (PREDIKSI)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>
            Formulir Skrining Pasien
        </div>
        <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1.2rem;'>
            Isi semua kolom di bawah, lalu klik tombol prediksi. Baca tab <b>📖 Panduan</b> jika bingung cara mengisi.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-section-label">Data Demografis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            usia = st.number_input("Usia (tahun)", min_value=20, max_value=85, value=45, step=1,
                                   help="Usia saat ini dalam tahun. Rentang valid: 20–85 tahun.")
        with c2:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        st.markdown('<div class="input-section-label" style="margin-top:0.8rem;">Riwayat Penyakit</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            riwayat = st.selectbox("Riwayat Keluarga Glaukoma", ["Tidak", "Ya"],
                                   help="Orang tua, saudara kandung, atau anak kandung pernah didiagnosis glaukoma.")
            diabetes = st.selectbox("Diabetes Melitus", ["Tidak", "Ya"],
                                    help="Pernah atau sedang didiagnosis DM Tipe 1 atau Tipe 2.")
        with c4:
            hipertensi = st.selectbox("Hipertensi / Darah Tinggi", ["Tidak", "Ya"],
                                      help="Pernah didiagnosis hipertensi atau sedang mengonsumsi obat antihipertensi.")
            migrain = st.selectbox("Migrain", ["Tidak", "Ya"],
                                   help="Pernah didiagnosis migrain oleh dokter.")

        sirkulasi = st.selectbox("Gangguan Sirkulasi Darah", ["Tidak", "Ya"],
                                 help="Penyakit jantung koroner, peripheral vascular disease, Raynaud syndrome, atau gangguan aliran darah lainnya.")

        st.markdown('<div class="input-section-label" style="margin-top:0.8rem;">Kondisi Mata & Obat-obatan</div>', unsafe_allow_html=True)
        refraksi = st.selectbox("Kelainan Refraksi (hasil periksa dokter mata)",
                                REFRAKSI_ORDER,
                                help="Pilih kondisi refraksi dari hasil pemeriksaan dokter mata. Jika belum pernah periksa, pilih 'Tidak Ada'.")
        kortiko = st.selectbox("Riwayat Penggunaan Kortikosteroid",
                               KORIKO_ORDER,
                               help="Tidak = tidak pernah pakai. Jangka Pendek = < 4 minggu. Jangka Panjang = ≥ 4 minggu atau rutin.")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🩺 Prediksi Risiko Saya", type="primary", use_container_width=True)

    with col_result:
        st.markdown("""
        <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>
            Hasil Skrining
        </div>
        <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1.2rem;'>
            Hasil prediksi akan muncul di sini setelah Anda klik tombol prediksi.
        </div>
        """, unsafe_allow_html=True)

        if not up_model or not up_encoder:
            st.markdown("""
            <div class="info-banner">
                ℹ️ File model (<code>model_glaukoma.joblib</code>) dan encoder (<code>label_encoder.joblib</code>) harus tersedia di direktori yang sama untuk menggunakan fitur prediksi.
            </div>
            """, unsafe_allow_html=True)
        else:
            artifact = load_model(up_model)
            le = load_encoder(up_encoder)

            if predict_btn:
                pasien = {
                    "Usia": usia, "Jenis_Kelamin": jenis_kelamin,
                    "Riwayat_Keluarga": riwayat, "Diabetes": diabetes,
                    "Hipertensi": hipertensi, "Migrain": migrain,
                    "Gangguan_Sirkulasi": sirkulasi,
                    "Kelainan_Refraksi": refraksi,
                    "Penggunaan_Kortikosteroid": kortiko,
                }
                
                X_input = encode_input(pasien)
                label, pred_enc, proba = prediksi(X_input, artifact, le)
                conf = max(proba.values())

                # Risk card
                card_cls = "risk-card-high" if pred_enc == 2 else "risk-card-medium" if pred_enc == 1 else "risk-card-low"
                icon = "🔴" if pred_enc == 2 else "🟡" if pred_enc == 1 else "🟢"
                
                advice_map = {
                    2: "Segera periksakan diri ke dokter spesialis mata (oftalmolog) untuk pemeriksaan tekanan intraokular dan kondisi saraf optik secara lengkap.",
                    1: "Lakukan pemeriksaan rutin ke dokter mata setiap 6–12 bulan. Pantau dan kelola faktor risiko yang Anda miliki.",
                    0: "Tetap jaga gaya hidup sehat. Lakukan pemeriksaan mata secara berkala setiap 1–2 tahun meski tidak ada gejala.",
                }

                st.markdown(f"""
                <div class="risk-card {card_cls}">
                    <div class="risk-label">{icon} {label}</div>
                    <div style='font-size:0.85rem; color:#4a5568; margin-bottom:0.5rem;'>Tingkat Kepercayaan Model: <b>{conf:.1f}%</b></div>
                    <div class="risk-advice">💬 {advice_map[pred_enc]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown("<div style='margin:0.8rem 0 0.4rem 0; font-weight:600; font-size:0.9rem;'>Distribusi Probabilitas</div>", unsafe_allow_html=True)
                bar_colors = {"Risiko Rendah": "#2da673", "Risiko Sedang": "#e8a040", "Risiko Tinggi": "#e05c5c"}
                for lbl, pct in proba.items():
                    st.markdown(f"""
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-label">
                            <span>{lbl}</span><span style='color:{bar_colors[lbl]}; font-weight:700;'>{pct:.1f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style='width:{pct}%; background:{bar_colors[lbl]};'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Active factors
                faktor_aktif = [k for k, v in {
                    "Riwayat Keluarga": riwayat, "Diabetes": diabetes,
                    "Hipertensi": hipertensi, "Migrain": migrain,
                    "Gangguan Sirkulasi": sirkulasi,
                }.items() if v == "Ya"]

                st.markdown("<div style='margin-top:1rem; font-weight:600; font-size:0.9rem;'>Faktor Risiko Aktif</div>", unsafe_allow_html=True)
                
                if faktor_aktif:
                    tags = "".join([f'<span class="factor-tag">{f}</span>' for f in faktor_aktif])
                    if kortiko != "Tidak":
                        tags += f'<span class="factor-tag">Kortikosteroid: {kortiko}</span>'
                    if refraksi != "Tidak Ada":
                        tags += f'<span class="factor-tag">Refraksi: {refraksi}</span>'
                    st.markdown(f"<div style='margin-top:0.4rem;'>{tags}</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<span class="factor-tag factor-tag-none">✓ Tidak ada faktor risiko biner aktif</span>', unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                <div class="disclaimer-banner" style='margin-top:1rem;'>
                    ⚠️ Hasil ini <b>bukan diagnosis medis</b>. Sistem ini menggunakan dataset sintetis dan dirancang sebagai alat edukasi skrining awal. Konsultasikan hasil ini ke dokter spesialis mata.
                </div>
                """, unsafe_allow_html=True)
            else:
                # Placeholder state
                st.markdown("""
                <div style='background:#f5f6fa; border-radius:14px; padding:2.5rem; text-align:center; color:#8890a4;'>
                    <div style='font-size:3rem; margin-bottom:0.8rem;'>🩺</div>
                    <div style='font-size:1rem; font-weight:600;'>Isi formulir dan klik Prediksi</div>
                    <div style='font-size:0.88rem; margin-top:0.4rem;'>Hasil skrining akan ditampilkan di sini</div>
                </div>
                """, unsafe_allow_html=True)

            # Model info expander
            artifact_info = load_model(up_model)
            with st.expander("ℹ️ Informasi Model Machine Learning"):
                m = artifact_info.get("metrics", {})
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("Accuracy", f"{m.get('accuracy', '-')}")
                ci2.metric("F1 Macro", f"{m.get('f1_macro', '-')}")
                ci3.metric("CV Mean", f"{m.get('cv_mean', '-')}")
                ci4.metric("CV Std", f"±{m.get('cv_std', '-')}")
                st.caption(f"**Best Params:** {artifact_info.get('best_params', {})}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB DISTRIBUSI DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:1rem;'>
        Distribusi Dataset Glaukoma
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    metrics = [
        (mc1, "Total Pasien", f"{total:,}", "#1a7a9e", ""),
        (mc2, "🟢 Risiko Rendah", f"{dist['Risiko Rendah']:,}", "#2da673", f"{dist['Risiko Rendah']/total*100:.1f}%"),
        (mc3, "🟡 Risiko Sedang", f"{dist['Risiko Sedang']:,}", "#e8a040", f"{dist['Risiko Sedang']/total*100:.1f}%"),
        (mc4, "🔴 Risiko Tinggi", f"{dist['Risiko Tinggi']:,}", "#e05c5c", f"{dist['Risiko Tinggi']/total*100:.1f}%"),
    ]

    for col, label, value, color, delta in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
                <div class="metric-card-value">{value}</div>
                <div class="metric-card-label">{label}</div>
                {f'<div class="metric-card-delta" style="color:{color};">{delta}</div>' if delta else ''}
            </div>
            """, unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax1 = axes[0]
    bars = ax1.bar(RISK_ORDER, dist.values, color=RISK_COLORS, alpha=0.88, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, count in zip(bars, dist.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{count}\n({count/total*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, dist.max() * 1.3 + 1)
    ax1.set_ylabel("Jumlah Pasien"); ax1.set_title("Jumlah Pasien per Label Risiko")

    ax2 = axes[1]; ax2.set_facecolor("white")
    wedges, _, autotexts = ax2.pie(
        dist.values, colors=RISK_COLORS, autopct="%1.1f%%", startangle=90,
        pctdistance=0.75, wedgeprops=dict(edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_color("white"); at.set_fontweight("bold")

    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r) for r in RISK_ORDER]
    ax2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=9)
    ax2.set_title("Proporsi Label Risiko")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="info-banner">📊 Dataset seimbang (balanced) — 500 sampel per kelas. Distribusi ini adalah hasil desain generator data sintetis berbasis literatur medis.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-1: FAKTOR RISIKO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_faktor:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-1 · Faktor Risiko Paling Diskriminatif</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Faktor mana yang paling membedakan kelompok Risiko Tinggi dari Risiko Rendah?</div>
    """, unsafe_allow_html=True)

    rows = []
    for f in FAKTOR_BINER:
        prop = (df.groupby("Label_Risiko")[f]
                .apply(lambda x: (x == "Ya").mean() * 100)
                .reindex([r for r in RISK_ORDER if r in df["Label_Risiko"].unique()]))
        rows.append({
            "Faktor": f.replace("_", " "),
            "Risiko Rendah": prop.get("Risiko Rendah", 0),
            "Risiko Sedang": prop.get("Risiko Sedang", 0),
            "Risiko Tinggi": prop.get("Risiko Tinggi", 0),
            "Gap (T−R)": prop.get("Risiko Tinggi", 0) - prop.get("Risiko Rendah", 0),
        })
    
    df_bq1 = pd.DataFrame(rows).sort_values("Gap (T−R)", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    ax1 = axes[0]
    x = np.arange(len(df_bq1)); width = 0.26
    for i, (risk, color) in enumerate(zip(
        [r for r in RISK_ORDER if r in df["Label_Risiko"].unique()],
        [RISK_PALETTE[r] for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    )):
        if risk in df_bq1.columns:
            bars = ax1.bar(x + (i - 1) * width, df_bq1[risk], width=width, color=color,
                           alpha=0.88, label=risk, edgecolor="white", linewidth=1)
            for bar, val in zip(bars, df_bq1[risk]):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#444")
    
    ax1.set_xticks(x); ax1.set_xticklabels(df_bq1["Faktor"], fontsize=9)
    ax1.set_ylabel("Proporsi 'Ya' (%)"); ax1.set_ylim(0, 100)
    ax1.set_title("Proporsi Pasien 'Ya' per Faktor Risiko"); ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = axes[1]
    colors_gap = ["#e05c5c" if g > 50 else "#e8a040" if g > 30 else "#f1c40f" for g in df_bq1["Gap (T−R)"]]
    bars2 = ax2.barh(df_bq1["Faktor"], df_bq1["Gap (T−R)"], color=colors_gap, alpha=0.88, edgecolor="white")
    for bar, gap in zip(bars2, df_bq1["Gap (T−R)"]):
        ax2.text(gap + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"+{gap:.1f}%", va="center", fontsize=9, fontweight="bold", color="#333")
    ax2.set_xlim(0, 80); ax2.set_xlabel("Gap (Risiko Tinggi − Risiko Rendah)")
    ax2.set_title("Seberapa Diskriminatif?")
    ax2.axvline(50, color="#e05c5c", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("📋 Tabel Lengkap BQ-1"):
        st.dataframe(df_bq1.round(1), use_container_width=True, hide_index=True)
    st.markdown('<div class="info-banner">💡 <b>Temuan:</b> Riwayat Keluarga memiliki gap +64%, diikuti Diabetes (+63%) dan Hipertensi (+63%). Seluruh faktor biner signifikan secara statistik (Chi-Square p < 0.001).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-2: USIA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_usia:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-2 · Distribusi Usia per Kelompok Risiko</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Apakah usia berkorelasi signifikan dengan tingkat risiko glaukoma?</div>
    """, unsafe_allow_html=True)

    H, p_kw = 0, 1
    groups = [df[df["Label_Risiko"] == r]["Usia"].dropna().values for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        H, p_kw = kruskal(*groups)

    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    for risk, color in zip(RISK_ORDER, RISK_COLORS):
        grp = df[df["Label_Risiko"] == risk]["Usia"].dropna()
        if len(grp) > 1:
            kde = gaussian_kde(grp, bw_method=0.3)
            x_range = np.linspace(df["Usia"].min() - 2, df["Usia"].max() + 2, 200)
            ax1.fill_between(x_range, kde(x_range), alpha=0.25, color=color)
            ax1.plot(x_range, kde(x_range), color=color, linewidth=2.5, label=f"{risk} (μ={grp.mean():.0f})")

    if p_kw < 0.001:
        ax1.text(0.97, 0.97, f"Kruskal-Wallis\nH={H:.1f}, p<0.001",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))
    
    ax1.set_xlabel("Usia (tahun)"); ax1.set_ylabel("Densitas")
    ax1.set_title("Distribusi KDE Usia per Kelompok Risiko"); ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    tbl_usia = pd.crosstab(df["Kelompok_Usia"], df["Label_Risiko"], normalize="index")
    tbl_usia = tbl_usia.reindex(columns=[r for r in RISK_ORDER if r in tbl_usia.columns]).mul(100).round(1)
    tbl_usia = tbl_usia.reindex([k for k in KELOMPOK_ORDER if k in tbl_usia.index])
    
    if not tbl_usia.empty:
        tbl_usia.plot(kind="bar", stacked=True, ax=ax2,
                      color=[RISK_PALETTE[r] for r in tbl_usia.columns], alpha=0.88,
                      edgecolor="white", linewidth=0.8, legend=False)
        cumulative = np.zeros(len(tbl_usia))
        for risk in tbl_usia.columns:
            vals = tbl_usia[risk].values
            for i, (val, cum) in enumerate(zip(vals, cumulative)):
                if val > 8:
                    ax2.text(i, cum + val / 2, f"{val:.0f}%",
                             ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            cumulative += vals

    ax2.set_xticklabels(tbl_usia.index, rotation=0)
    ax2.set_ylabel("Proporsi (%)"); ax2.set_xlabel("Kelompok Usia")
    ax2.set_title("Proporsi Label Risiko per Kelompok Usia")
    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r) for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    ax2.legend(handles=handles, title="Label Risiko", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("📋 Statistik Usia per Kelompok Risiko"):
        stat = df.groupby("Label_Risiko")["Usia"].agg(N="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
        stat = stat.reindex([r for r in RISK_ORDER if r in df["Label_Risiko"].unique()]).round(1)
        st.dataframe(stat, use_container_width=True)
    
    st.markdown('<div class="info-banner">💡 Risiko Rendah (μ≈34 th) vs Risiko Tinggi (μ≈69 th). Kelompok usia 60+ menyumbang 72,7% dari total populasi Risiko Tinggi. Uji Kruskal-Wallis: p < 0.001.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-3: KOMBINASI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_kombinasi:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-3 · Efek Akumulatif Kombinasi Faktor Risiko</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Apakah memiliki lebih banyak faktor risiko sekaligus meningkatkan risiko secara signifikan?</div>
    """, unsafe_allow_html=True)

    tbl_kombid = pd.crosstab(df["Kombinasi_Risiko"], df["Label_Risiko"], normalize="index")
    tbl_kombid = tbl_kombid.reindex(columns=[r for r in RISK_ORDER if r in tbl_kombid.columns]).mul(100).round(1)
    tbl_kombid = tbl_kombid.sort_values(tbl_kombid.columns[-1])
    count_kombid = df["Kombinasi_Risiko"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    tbl_kombid.plot(kind="barh", stacked=True, ax=ax,
                    color=[RISK_PALETTE[r] for r in tbl_kombid.columns], alpha=0.88,
                    edgecolor="white", linewidth=0.8)
    cumulative = np.zeros(len(tbl_kombid))
    for risk in tbl_kombid.columns:
        vals = tbl_kombid[risk].values
        for i, (val, cum) in enumerate(zip(vals, cumulative)):
            if val > 8:
                ax.text(cum + val / 2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        cumulative += vals

    last_col = tbl_kombid.columns[-1]
    for i, (idx, row) in enumerate(tbl_kombid.iterrows()):
        n = count_kombid.get(idx, 0); pct = row[last_col]
        ax.text(103, i, f"{pct:.0f}% (n={n})", va="center", fontsize=9,
                color="#e05c5c" if pct > 50 else "#888", fontweight="bold" if pct > 50 else "normal")

    ax.set_xlim(0, 122); ax.set_xlabel("Proporsi (%)")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Proporsi Label Risiko per Kombinasi Faktor\n(angka merah = % Risiko Tinggi)")
    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r) for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    ax.legend(handles=handles, title="Label Risiko", bbox_to_anchor=(1.18, 1), loc="upper left", fontsize=9)

    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="info-banner">💡 Riwayat + DM + HT = <b>83% Risiko Tinggi</b>. Tanpa faktor apapun = <b>0% Risiko Tinggi</b>. Efek bersifat akumulatif, bukan sekadar aditif.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-4: KORTIKOSTEROID
# ═══════════════════════════════════════════════════════════════════════════════
with tab_koriko:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-4 · Pengaruh Durasi Kortikosteroid terhadap Risiko</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Apakah semakin lama menggunakan kortikosteroid, semakin tinggi risikonya?</div>
    """, unsafe_allow_html=True)

    tbl_koriko_pct = pd.crosstab(df["Penggunaan_Kortikosteroid"], df["Label_Risiko"], normalize="index")
    tbl_koriko_pct = tbl_koriko_pct.reindex(columns=[r for r in RISK_ORDER if r in tbl_koriko_pct.columns]).mul(100).round(1)
    tbl_koriko_pct = tbl_koriko_pct.reindex([k for k in KORIKO_ORDER if k in tbl_koriko_pct.index])

    tbl_koriko_n = pd.crosstab(df["Penggunaan_Kortikosteroid"], df["Label_Risiko"])
    tbl_koriko_n = tbl_koriko_n.reindex([k for k in KORIKO_ORDER if k in tbl_koriko_n.index])
    tbl_koriko_n = tbl_koriko_n.reindex(columns=[r for r in RISK_ORDER if r in tbl_koriko_n.columns])

    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)
    
    ax1 = fig.add_subplot(gs[0])
    tbl_koriko_pct.plot(kind="bar", stacked=True, ax=ax1,
                        color=[RISK_PALETTE[r] for r in tbl_koriko_pct.columns], alpha=0.88,
                        edgecolor="white", linewidth=0.8, legend=False)
    cumulative = np.zeros(len(tbl_koriko_pct))
    for risk in tbl_koriko_pct.columns:
        vals = tbl_koriko_pct[risk].values
        for i, (val, cum) in enumerate(zip(vals, cumulative)):
            if val > 8:
                ax1.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center",
                         fontsize=10, color="white", fontweight="bold")
        cumulative += vals
    ax1.set_xticklabels(tbl_koriko_pct.index, rotation=0)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_title("Komposisi Risiko per Durasi")

    ax2 = fig.add_subplot(gs[1])
    x_pts = list(range(len(tbl_koriko_pct)))
    if len(tbl_koriko_pct.columns) >= 2:
        last_col = tbl_koriko_pct.columns[-1]
        first_col = tbl_koriko_pct.columns[0]
        ax2.plot(x_pts, tbl_koriko_pct[last_col].values, "o-", color="#e05c5c",
                 linewidth=2.5, markersize=10, label=last_col)
        ax2.plot(x_pts, tbl_koriko_pct[first_col].values, "s--", color="#2da673",
                 linewidth=2, markersize=8, label=first_col, alpha=0.8)
        for xp, y in zip(x_pts, tbl_koriko_pct[last_col].values):
            ax2.text(xp + 0.05, y + 3, f"{y:.0f}%", fontsize=10, color="#e05c5c", fontweight="bold")
    ax2.set_xticks(x_pts); ax2.set_xticklabels(tbl_koriko_pct.index, fontsize=9)
    ax2.set_ylabel("Proporsi (%)"); ax2.set_ylim(-5, 95)
    ax2.set_title("Eskalasi Risiko vs Durasi"); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    sns.heatmap(tbl_koriko_n, ax=ax3, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8})
    ax3.set_title("Heatmap Jumlah Pasien")
    ax3.set_xlabel("Label Risiko"); ax3.set_ylabel("Durasi")

    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="info-banner">💡 Kortikosteroid Jangka Panjang → 76% Risiko Tinggi. Tidak pakai → 0% Risiko Tinggi. Pola eskalasi linier dan sangat konsisten.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-5: GENDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gender:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-5 · Pengaruh Jenis Kelamin terhadap Distribusi Risiko</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Apakah jenis kelamin menjadi faktor pembeda risiko glaukoma?</div>
    """, unsafe_allow_html=True)

    tbl_gender_pct = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"], normalize="index")
    tbl_gender_pct = tbl_gender_pct.reindex(columns=[r for r in RISK_ORDER if r in tbl_gender_pct.columns]).mul(100).round(1)
    
    tbl_gender_n = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"])
    ct = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"])
    chi2_stat, p_chi, dof = 0, 1, 0
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2_stat, p_chi, dof, _ = chi2_contingency(ct)

    GENDER_COLORS = {"Laki-laki": "#3498db", "Perempuan": "#e91e8c"}
    genders = [g for g in ["Laki-laki", "Perempuan"] if g in tbl_gender_pct.index]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax1 = axes[0]
    x = np.arange(len(tbl_gender_pct.columns)); w = 0.35
    for i, gender in enumerate(genders):
        vals = tbl_gender_pct.loc[gender]
        bars = ax1.bar(x + (i - 0.5) * w, vals, width=w, color=GENDER_COLORS[gender],
                       alpha=0.85, label=gender, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(tbl_gender_pct.columns, fontsize=9)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_ylim(0, 50)
    ax1.set_title("Proporsi per Jenis Kelamin"); ax1.legend(fontsize=9)
    ax1.text(0.5, 0.95, f"χ²={chi2_stat:.2f}\np={p_chi:.4f}",
             transform=ax1.transAxes, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))

    for ax, gender in zip(axes[1:], genders):
        ax.set_facecolor("white")
        vals = tbl_gender_pct.loc[gender].values
        wedges, _, autotexts = ax.pie(
            vals, colors=[RISK_PALETTE[r] for r in tbl_gender_pct.columns],
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2.5),
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
        n_gender = tbl_gender_n.loc[gender].sum() if gender in tbl_gender_n.index else 0
        ax.text(0, 0, f"{gender}\nn={n_gender}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=GENDER_COLORS[gender])
        ax.set_title(f"Distribusi Risiko\n{gender}")

    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r) for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    fig.legend(handles=handles, title="Label Risiko", loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    
    plt.tight_layout(); st.pyplot(fig); plt.close()

    kesimpulan = "signifikan" if p_chi < 0.05 else "tidak signifikan"
    st.markdown(f'<div class="info-banner">💡 Perbedaan gender {kesimpulan} (χ²={chi2_stat:.3f}, p={p_chi:.4f}). Jenis kelamin bukan faktor pembeda utama risiko glaukoma dalam dataset ini.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB BQ-6: REFRAKSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_refraksi:
    st.markdown("""
    <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#17a2b8; margin-bottom:0.2rem;'>BQ-6 · Kelainan Refraksi pada Pasien Risiko Glaukoma</div>
    <div style='color:#8890a4; font-size:0.88rem; margin-bottom:1rem;'>Jenis kelainan refraksi apa yang paling banyak ditemukan pada kelompok Risiko Tinggi?</div>
    """, unsafe_allow_html=True)

    tbl_refraksi = pd.crosstab(df["Kelainan_Refraksi"], df["Label_Risiko"], normalize="index")
    tbl_refraksi = tbl_refraksi.reindex(columns=[r for r in RISK_ORDER if r in tbl_refraksi.columns]).mul(100).round(1)
    tbl_refraksi = tbl_refraksi.reindex([r for r in REFRAKSI_ORDER if r in tbl_refraksi.index])

    risiko_tinggi_df = df[df["Label_Risiko"] == "Risiko Tinggi"]
    komposisi_tinggi = (risiko_tinggi_df["Kelainan_Refraksi"].value_counts(normalize=True)
                        .mul(100).reindex(REFRAKSI_ORDER).fillna(0).round(1))

    ct_r = pd.crosstab(df["Kelainan_Refraksi"], df["Label_Risiko"])
    p_r = 1.0
    if ct_r.shape[0] >= 2 and ct_r.shape[1] >= 2:
        _, p_r, _, _ = chi2_contingency(ct_r)

    REFRAKSI_COLORS = ["#95a5a6", "#3498db", "#1a5276", "#f39c12", "#e67e22"]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1 = axes[0]
    x = np.arange(len(tbl_refraksi)); width = 0.26
    for i, (risk, color) in enumerate(zip(tbl_refraksi.columns, [RISK_PALETTE[r] for r in tbl_refraksi.columns])):
        bars = ax1.bar(x + (i - 1) * width, tbl_refraksi[risk], width=width, color=color,
                       alpha=0.88, label=risk, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 6:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                         f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace(" ", "\n") for r in tbl_refraksi.index], fontsize=8)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_ylim(0, 68)
    ax1.set_title(f"Proporsi Risiko per Jenis Refraksi\n(Chi-Square p={p_r:.4f})")
    ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = axes[1]; ax2.set_facecolor("white")
    mask = komposisi_tinggi > 0
    vals_nz = komposisi_tinggi[mask]
    labels_pie = vals_nz.index.tolist()
    colors_pie = [REFRAKSI_COLORS[REFRAKSI_ORDER.index(l)] for l in labels_pie if l in REFRAKSI_ORDER]
    
    if len(vals_nz) > 0:
        wedges, _, autotexts = ax2.pie(
            vals_nz, colors=colors_pie, autopct="%1.1f%%", startangle=90, pctdistance=0.72,
            wedgeprops=dict(edgecolor="white", linewidth=2.5),
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
        ax2.legend(wedges, labels_pie, title="Jenis Refraksi", fontsize=9, bbox_to_anchor=(1.35, 0.5), loc="center right")

    ax2.set_title(f"Komposisi Refraksi pada Risiko Tinggi\n(n={len(risiko_tinggi_df)})")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="info-banner">💡 Miopia Sedang/Berat memiliki proporsi Risiko Tinggi tertinggi di antara semua kategori refraksi. Kelainan refraksi berasosiasi signifikan dengan label risiko (p < 0.001).</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <b>Opsis</b> · Dashboard Analisis Risiko Glaukoma &nbsp;| &nbsp;
    Dataset sintetis berbasis literatur medis — <b>bukan untuk keperluan klinis</b> &nbsp;| &nbsp;
    Capstone Coding Camp 2026 · DBS Foundation × Dicoding <br>
    <span style='margin-top:0.3rem; display:block;'>
        Referensi utama: Tham et al. (2014) · Roberti et al. (2020) · Grzybowski et al. (2020) ·
        Worley & Grimmer-Somers (2011) · Razeghinejad & Katz (2012)
    </span>
</div>
""", unsafe_allow_html=True)