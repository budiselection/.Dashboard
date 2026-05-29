import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy.stats import chi2_contingency, kruskal, gaussian_kde

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title          = "Dashboard Risiko Glaukoma",
    page_icon           = "",
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ── Konstanta ──────────────────────────────────────────────────────────────────
RISK_PALETTE   = {"Risiko Rendah": "#2ecc71", "Risiko Sedang": "#f39c12", "Risiko Tinggi": "#e74c3c"}
RISK_ORDER     = ["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"]
RISK_COLORS    = [RISK_PALETTE[r] for r in RISK_ORDER]
KORIKO_ORDER   = ["Tidak", "Jangka Pendek", "Jangka Panjang"]
KELOMPOK_ORDER = ["< 30", "30–44", "45–59", "60+"]
REFRAKSI_ORDER = [
    "Tidak Ada", "Miopia Ringan", "Miopia Sedang/Berat",
    "Hipermetropia Ringan", "Hipermetropia Sedang/Berat",
]
FAKTOR_BINER   = [
    "Riwayat_Keluarga", "Diabetes", "Hipertensi",
    "Migrain", "Gangguan_Sirkulasi",
]

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#f8f9fa",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.35,
    "grid.linestyle"    : "--",
    "axes.titleweight"  : "bold",
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
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


# ── Fungsi encoding & prediksi ────────────────────────────────────────────────
def encode_input(pasien: dict) -> pd.DataFrame:
    usia_scaled     = (pasien["Usia"] - 20) / (85 - 20)
    gender_enc      = 1 if pasien["Jenis_Kelamin"] == "Laki-laki" else 0
    biner = {
        col: 1 if pasien[col] == "Ya" else 0
        for col in ["Riwayat_Keluarga", "Diabetes", "Hipertensi",
                    "Migrain", "Gangguan_Sirkulasi"]
    }
    map_refraksi = {
        "Tidak Ada": 0, "Miopia Ringan": 1, "Hipermetropia Ringan": 1,
        "Miopia Sedang/Berat": 2, "Hipermetropia Sedang/Berat": 2,
    }
    map_koriko = {"Tidak": 0, "Jangka Pendek": 1, "Jangka Panjang": 2}

    refraksi_enc    = map_refraksi.get(pasien["Kelainan_Refraksi"], 0)
    koriko_enc      = map_koriko.get(pasien["Penggunaan_Kortikosteroid"], 0)
    jumlah_komorbid = sum(biner.values())
    risiko_komposit = round(
        (koriko_enc / 2) * 0.30 +
        biner["Hipertensi"]         * 0.20 +
        biner["Riwayat_Keluarga"]   * 0.18 +
        biner["Diabetes"]           * 0.15 +
        biner["Migrain"]            * 0.10 +
        biner["Gangguan_Sirkulasi"] * 0.07,
        4,
    )
    return pd.DataFrame([{
        "Usia"                      : usia_scaled,
        "Jenis_Kelamin"             : gender_enc,
        "Riwayat_Keluarga"          : biner["Riwayat_Keluarga"],
        "Diabetes"                  : biner["Diabetes"],
        "Hipertensi"                : biner["Hipertensi"],
        "Migrain"                   : biner["Migrain"],
        "Gangguan_Sirkulasi"        : biner["Gangguan_Sirkulasi"],
        "Kelainan_Refraksi"         : refraksi_enc,
        "Penggunaan_Kortikosteroid" : koriko_enc,
        "Jumlah_Komorbid"           : jumlah_komorbid,
        "Risiko_Komposit"           : risiko_komposit,
    }])


def prediksi(X_input, artifact, le):
    model       = artifact["model"]
    label_names = artifact["label_names"]
    pred_enc    = int(model.predict(X_input)[0])
    pred_proba  = model.predict_proba(X_input)[0]
    label       = le.inverse_transform([pred_enc])[0]
    return label, pred_enc, {label_names[i]: round(float(p) * 100, 2) for i, p in enumerate(pred_proba)}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title(" Glaukoma Dashboard")
    st.markdown("Analisis Faktor Risiko & Prediksi Glaukoma")

    up_data    = "dataset_glaukoma_enriched.csv"
    up_model   = "model_glaukoma.joblib"
    up_encoder = "label_encoder.joblib"

    st.divider()
    if up_data:
        df_raw = load_data(up_data)
        st.subheader("🔧 Filter Data")
        gender_opt = ["Semua"] + sorted(df_raw["Jenis_Kelamin"].unique().tolist())
        risk_opt   = ["Semua"] + RISK_ORDER
        koriko_opt = ["Semua"] + KORIKO_ORDER
        sel_gender = st.selectbox("Jenis Kelamin", gender_opt)
        sel_risk   = st.selectbox("Label Risiko",  risk_opt)
        sel_koriko = st.selectbox("Kortikosteroid", koriko_opt)
        usia_min   = int(df_raw["Usia"].min())
        usia_max   = int(df_raw["Usia"].max())
        sel_usia   = st.slider("Rentang Usia", usia_min, usia_max, (usia_min, usia_max))
    st.divider()
    st.caption("Proyek Analisis Data · Langkah 6")


# ── Terapkan filter ────────────────────────────────────────────────────────────
def apply_filters(df):
    out = df.copy()
    if sel_gender != "Semua":
        out = out[out["Jenis_Kelamin"] == sel_gender]
    if sel_risk != "Semua":
        out = out[out["Label_Risiko"] == sel_risk]
    if sel_koriko != "Semua":
        out = out[out["Penggunaan_Kortikosteroid"] == sel_koriko]
    out = out[(out["Usia"] >= sel_usia[0]) & (out["Usia"] <= sel_usia[1])]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.title(" Dashboard Analisis Risiko Glaukoma")
st.markdown(
    "Dashboard ini menampilkan **analisis data** dan **prediksi risiko glaukoma** "
    "menggunakan model Random Forest yang sudah ditraining."
)

if not up_data:
    st.info("⬅️ Upload file di sidebar untuk memulai.")
    st.stop()

df     = apply_filters(df_raw)
total  = len(df)
dist   = df["Label_Risiko"].value_counts().reindex(RISK_ORDER, fill_value=0)

if total == 0:
    st.warning("⚠️ Tidak ada data sesuai filter.")
    st.stop()


# ── Tab navigasi ───────────────────────────────────────────────────────────────
tab_pred, tab_dist, tab_faktor, tab_usia, tab_kombinasi, tab_koriko, tab_gender, tab_refraksi = st.tabs([
    "🤖 Prediksi Risiko",
    "📊 Distribusi Data",
    "🔴 BQ-1 Faktor",
    "📅 BQ-2 Usia",
    "⚡ BQ-3 Kombinasi",
    "💊 BQ-4 Kortikosteroid",
    "👥 BQ-5 Gender",
    "👁 BQ-6 Refraksi",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### 🤖 Prediksi Risiko Glaukoma Pasien Baru")

    if not up_model or not up_encoder:
        st.warning("⬅️ Upload `model_glaukoma.joblib` dan `label_encoder.joblib` di sidebar untuk menggunakan fitur prediksi.")
    else:
        artifact = load_model(up_model)
        le       = load_encoder(up_encoder)

        st.markdown("#### Input Data Pasien")
        col1, col2, col3 = st.columns(3)

        with col1:
            usia         = st.number_input("Usia (tahun)", min_value=20, max_value=85, value=45, step=1)
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            riwayat      = st.selectbox("Riwayat Keluarga", ["Tidak", "Ya"])
            diabetes     = st.selectbox("Diabetes", ["Tidak", "Ya"])

        with col2:
            hipertensi   = st.selectbox("Hipertensi", ["Tidak", "Ya"])
            migrain      = st.selectbox("Migrain", ["Tidak", "Ya"])
            sirkulasi    = st.selectbox("Gangguan Sirkulasi", ["Tidak", "Ya"])

        with col3:
            refraksi     = st.selectbox("Kelainan Refraksi", REFRAKSI_ORDER)
            kortiko      = st.selectbox("Penggunaan Kortikosteroid", KORIKO_ORDER)

        st.divider()

        if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
            pasien = {
                "Usia"                      : usia,
                "Jenis_Kelamin"             : jenis_kelamin,
                "Riwayat_Keluarga"          : riwayat,
                "Diabetes"                  : diabetes,
                "Hipertensi"                : hipertensi,
                "Migrain"                   : migrain,
                "Gangguan_Sirkulasi"        : sirkulasi,
                "Kelainan_Refraksi"         : refraksi,
                "Penggunaan_Kortikosteroid" : kortiko,
            }

            X_input             = encode_input(pasien)
            label, pred_enc, proba = prediksi(X_input, artifact, le)
            warna               = RISK_PALETTE[label]
            conf                = max(proba.values())

            # Hasil prediksi
            st.markdown("#### Hasil Prediksi")
            r1, r2, r3 = st.columns([1, 1, 2])

            with r1:
                st.metric("Label Risiko", label)
            with r2:
                st.metric("Confidence", f"{conf:.1f}%")
            with r3:
                # Bar probabilitas
                fig, ax = plt.subplots(figsize=(5, 2.2))
                fig.patch.set_facecolor("white")
                bars = ax.barh(
                    list(proba.keys()), list(proba.values()),
                    color=[RISK_PALETTE[l] for l in proba.keys()],
                    alpha=0.85, edgecolor="white",
                )
                for bar, val in zip(bars, proba.values()):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
                ax.set_xlim(0, 115)
                ax.set_xlabel("Probabilitas (%)")
                ax.set_title("Distribusi Probabilitas", fontsize=10)
                # Highlight bar prediksi
                max_idx = list(proba.values()).index(max(proba.values()))
                bars[max_idx].set_edgecolor("#333")
                bars[max_idx].set_linewidth(2)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Rekomendasi
            st.divider()
            if pred_enc == 2:
                st.error(
                    "🔴 **Risiko Tinggi** — Disarankan segera melakukan pemeriksaan mata menyeluruh "
                    "dan konsultasi dengan dokter spesialis mata."
                )
            elif pred_enc == 1:
                st.warning(
                    "🟡 **Risiko Sedang** — Disarankan melakukan pemeriksaan rutin setiap 6–12 bulan "
                    "dan memantau faktor risiko yang ada."
                )
            else:
                st.success(
                    "🟢 **Risiko Rendah** — Tetap jaga pola hidup sehat dan lakukan pemeriksaan mata "
                    "secara berkala setiap 1–2 tahun."
                )

            # Ringkasan faktor aktif
            faktor_aktif = [k for k, v in {
                "Riwayat Keluarga": riwayat, "Diabetes": diabetes,
                "Hipertensi": hipertensi, "Migrain": migrain,
                "Gangguan Sirkulasi": sirkulasi,
            }.items() if v == "Ya"]

            if faktor_aktif:
                st.info(f"**Faktor risiko aktif:** {', '.join(faktor_aktif)} "
                        f"| Kortikosteroid: {kortiko} | Refraksi: {refraksi}")
            else:
                st.info("Tidak ada faktor risiko biner yang aktif.")

        # Info model
        if up_model:
            artifact_info = load_model(up_model)
            with st.expander("ℹ️ Informasi Model"):
                m = artifact_info.get("metrics", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{m.get('accuracy', '-')}")
                c2.metric("F1 Macro",  f"{m.get('f1_macro', '-')}")
                c3.metric("CV Mean",   f"{m.get('cv_mean', '-')}")
                c4.metric("CV Std",    f"±{m.get('cv_std', '-')}")
                st.write("**Best Params:**", artifact_info.get("best_params", {}))


# ══════════════════════════════════════════════════════════════════════════════
# TAB DISTRIBUSI DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.markdown("### 📊 Distribusi Label Risiko")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pasien",    f"{total:,}")
    c2.metric("🟢 Risiko Rendah", f"{dist['Risiko Rendah']:,}", f"{dist['Risiko Rendah']/total*100:.1f}%")
    c3.metric("🟡 Risiko Sedang", f"{dist['Risiko Sedang']:,}", f"{dist['Risiko Sedang']/total*100:.1f}%")
    c4.metric("🔴 Risiko Tinggi", f"{dist['Risiko Tinggi']:,}", f"{dist['Risiko Tinggi']/total*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax1 = axes[0]
    bars = ax1.bar(RISK_ORDER, dist.values, color=RISK_COLORS, alpha=0.85, edgecolor="white")
    for bar, count in zip(bars, dist.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{count}\n({count/total*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, dist.max() * 1.25 + 1)
    ax1.set_ylabel("Jumlah Pasien")
    ax1.set_title("Jumlah Pasien per Label Risiko")

    ax2 = axes[1]
    ax2.set_facecolor("white")
    wedges, _, autotexts = ax2.pie(
        dist.values, colors=RISK_COLORS,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_color("white"); at.set_fontweight("bold")
    ax2.legend([mpatches.Patch(color=c) for c in RISK_COLORS], RISK_ORDER,
               loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    ax2.set_title("Proporsi Label Risiko")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Tabel distribusi"):
        st.dataframe(pd.DataFrame({
            "Label Risiko" : RISK_ORDER,
            "Jumlah"       : dist.values,
            "Proporsi (%)" : (dist.values / total * 100).round(1),
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-2: USIA
# ══════════════════════════════════════════════════════════════════════════════
with tab_usia:
    st.markdown("### BQ-2 · Distribusi Usia per Label Risiko")
    st.caption("Apakah usia tua identik dengan risiko glaukoma lebih tinggi?")

    grp_usia = [df[df["Label_Risiko"] == r]["Usia"].values for r in RISK_ORDER]
    valid    = [g for g in grp_usia if len(g) > 1]

    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    x_range = np.linspace(df["Usia"].min() - 3, df["Usia"].max() + 3, 300)
    for risk, color in RISK_PALETTE.items():
        subset = df[df["Label_Risiko"] == risk]["Usia"]
        if len(subset) > 1:
            kde = gaussian_kde(subset)
            y_kde = kde(x_range)
            ax1.plot(x_range, y_kde, color=color, linewidth=2.5, label=risk)
            ax1.fill_between(x_range, y_kde, alpha=0.12, color=color)
            mean_val = float(subset.mean())
            ax1.axvline(mean_val, color=color, linestyle=":", linewidth=1.5)
            ax1.text(mean_val + 0.5, float(kde(np.array([mean_val]))[0]) * 0.6,
                     f"μ={mean_val:.0f}", color=color, fontsize=9, fontweight="bold")
    if len(valid) >= 2:
        H, p_kw = kruskal(*valid)
        ax1.text(0.97, 0.97, f"Kruskal-Wallis\nH={H:.1f}, p={p_kw:.4f}",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))
    ax1.set_xlabel("Usia (tahun)"); ax1.set_ylabel("Densitas")
    ax1.set_title("Distribusi KDE Usia per Kelompok Risiko")
    ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    tbl_usia = pd.crosstab(df["Kelompok_Usia"], df["Label_Risiko"], normalize="index")
    tbl_usia = tbl_usia.reindex(columns=[r for r in RISK_ORDER if r in tbl_usia.columns])
    tbl_usia = tbl_usia.mul(100).round(1)
    tbl_usia = tbl_usia.reindex([k for k in KELOMPOK_ORDER if k in tbl_usia.index])
    if not tbl_usia.empty:
        tbl_usia.plot(kind="bar", stacked=True, ax=ax2,
                      color=[RISK_PALETTE[r] for r in tbl_usia.columns],
                      alpha=0.85, edgecolor="white", linewidth=0.5, legend=False)
        cumulative = np.zeros(len(tbl_usia))
        for risk in tbl_usia.columns:
            vals = tbl_usia[risk].values
            for i, (val, cum) in enumerate(zip(vals, cumulative)):
                if val > 8:
                    ax2.text(i, cum + val / 2, f"{val:.0f}%",
                             ha="center", va="center", fontsize=9,
                             color="white", fontweight="bold")
            cumulative += vals
    ax2.set_xticklabels(tbl_usia.index, rotation=0)
    ax2.set_ylabel("Proporsi (%)"); ax2.set_xlabel("Kelompok Usia")
    ax2.set_title("Proporsi Label Risiko per Kelompok Usia")
    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r)
               for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    ax2.legend(handles=handles, title="Label Risiko",
               bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Statistik usia"):
        stat = df.groupby("Label_Risiko")["Usia"].agg(
            N="count", Mean="mean", Median="median", Std="std", Min="min", Max="max"
        ).reindex([r for r in RISK_ORDER if r in df["Label_Risiko"].unique()]).round(1)
        st.dataframe(stat, use_container_width=True)

    st.info("💡 Risiko Rendah μ≈34 th vs Risiko Tinggi μ≈69 th. Kelompok 60+ paling berisiko.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-1: FAKTOR RISIKO
# ══════════════════════════════════════════════════════════════════════════════
with tab_faktor:
    st.markdown("### BQ-1 · Faktor Risiko Paling Diskriminatif")
    st.caption("Faktor mana yang paling membedakan kelompok Risiko Tinggi dari Risiko Rendah?")

    rows = []
    for f in FAKTOR_BINER:
        prop = (df.groupby("Label_Risiko")[f]
                .apply(lambda x: (x == "Ya").mean() * 100)
                .reindex([r for r in RISK_ORDER if r in df["Label_Risiko"].unique()]))
        rows.append({
            "Faktor"       : f.replace("_", " "),
            "Risiko Rendah": prop.get("Risiko Rendah", 0),
            "Risiko Sedang": prop.get("Risiko Sedang", 0),
            "Risiko Tinggi": prop.get("Risiko Tinggi", 0),
            "Gap (T−R)"    : prop.get("Risiko Tinggi", 0) - prop.get("Risiko Rendah", 0),
        })
    df_bq1 = pd.DataFrame(rows).sort_values("Gap (T−R)", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax1 = axes[0]
    x = np.arange(len(df_bq1)); width = 0.26
    for i, (risk, color) in enumerate(zip(
        [r for r in RISK_ORDER if r in df["Label_Risiko"].unique()],
        [RISK_PALETTE[r] for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    )):
        if risk in df_bq1.columns:
            bars = ax1.bar(x + (i - 1) * width, df_bq1[risk], width=width,
                           color=color, alpha=0.85, label=risk, edgecolor="white")
            for bar, val in zip(bars, df_bq1[risk]):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#444")
    ax1.set_xticks(x); ax1.set_xticklabels(df_bq1["Faktor"], fontsize=9)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_ylim(0, 100)
    ax1.set_title("Proporsi Pasien 'Ya' per Faktor")
    ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = axes[1]
    colors_gap = ["#e74c3c" if g > 50 else "#f39c12" if g > 30 else "#f1c40f"
                  for g in df_bq1["Gap (T−R)"]]
    bars2 = ax2.barh(df_bq1["Faktor"], df_bq1["Gap (T−R)"],
                     color=colors_gap, alpha=0.85, edgecolor="white")
    for bar, gap in zip(bars2, df_bq1["Gap (T−R)"]):
        ax2.text(gap + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"+{gap:.1f}%", va="center", fontsize=9, fontweight="bold", color="#333")
    ax2.set_xlim(0, 80); ax2.set_xlabel("Gap (T − R)")
    ax2.set_title("Seberapa Diskriminatif?")
    ax2.axvline(50, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Tabel lengkap"):
        st.dataframe(df_bq1.round(1), use_container_width=True, hide_index=True)

    st.info("💡 Hipertensi, Diabetes, dan Riwayat Keluarga memiliki gap terbesar (>60%).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-3: KOMBINASI
# ══════════════════════════════════════════════════════════════════════════════
with tab_kombinasi:
    st.markdown("### BQ-3 · Efek Akumulatif Kombinasi Faktor Risiko")
    st.caption("Apakah memiliki lebih banyak faktor sekaligus meningkatkan risiko secara signifikan?")

    tbl_kombid = pd.crosstab(df["Kombinasi_Risiko"], df["Label_Risiko"], normalize="index")
    tbl_kombid = tbl_kombid.reindex(columns=[r for r in RISK_ORDER if r in tbl_kombid.columns])
    tbl_kombid = tbl_kombid.mul(100).round(1)
    tbl_kombid = tbl_kombid.sort_values(tbl_kombid.columns[-1])
    count_kombid = df["Kombinasi_Risiko"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    tbl_kombid.plot(kind="barh", stacked=True, ax=ax,
                    color=[RISK_PALETTE[r] for r in tbl_kombid.columns],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
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
        ax.text(102, i, f"{pct:.0f}%  (n={n})", va="center", fontsize=9,
                color="#e74c3c" if pct > 50 else "#888",
                fontweight="bold" if pct > 50 else "normal")
    ax.set_xlim(0, 120); ax.set_xlabel("Proporsi (%)")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Proporsi Label Risiko per Kombinasi Faktor\n(angka merah = % label risiko tertinggi)")
    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r)
               for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    ax.legend(handles=handles, title="Label Risiko",
              bbox_to_anchor=(1.18, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("💡 Riwayat + DM + HT = 83% Risiko Tinggi. Tanpa faktor apapun = 0% Risiko Tinggi.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-4: KORTIKOSTEROID
# ══════════════════════════════════════════════════════════════════════════════
with tab_koriko:
    st.markdown("### BQ-4 · Pengaruh Durasi Kortikosteroid terhadap Risiko")
    st.caption("Apakah semakin lama menggunakan kortikosteroid, semakin tinggi risikonya?")

    tbl_koriko_pct = pd.crosstab(df["Penggunaan_Kortikosteroid"], df["Label_Risiko"], normalize="index")
    tbl_koriko_pct = tbl_koriko_pct.reindex(columns=[r for r in RISK_ORDER if r in tbl_koriko_pct.columns])
    tbl_koriko_pct = tbl_koriko_pct.mul(100).round(1)
    tbl_koriko_pct = tbl_koriko_pct.reindex([k for k in KORIKO_ORDER if k in tbl_koriko_pct.index])
    tbl_koriko_n   = pd.crosstab(df["Penggunaan_Kortikosteroid"], df["Label_Risiko"])
    tbl_koriko_n   = tbl_koriko_n.reindex([k for k in KORIKO_ORDER if k in tbl_koriko_n.index])
    tbl_koriko_n   = tbl_koriko_n.reindex(columns=[r for r in RISK_ORDER if r in tbl_koriko_n.columns])

    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    tbl_koriko_pct.plot(kind="bar", stacked=True, ax=ax1,
                        color=[RISK_PALETTE[r] for r in tbl_koriko_pct.columns],
                        alpha=0.85, edgecolor="white", linewidth=0.5, legend=False)
    cumulative = np.zeros(len(tbl_koriko_pct))
    for risk in tbl_koriko_pct.columns:
        vals = tbl_koriko_pct[risk].values
        for i, (val, cum) in enumerate(zip(vals, cumulative)):
            if val > 8:
                ax1.text(i, cum + val / 2, f"{val:.0f}%",
                         ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        cumulative += vals
    ax1.set_xticklabels(tbl_koriko_pct.index, rotation=0)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_title("Komposisi Risiko per Durasi")

    ax2 = fig.add_subplot(gs[1])
    x_pts = list(range(len(tbl_koriko_pct)))
    if len(tbl_koriko_pct.columns) >= 2:
        last_col  = tbl_koriko_pct.columns[-1]
        first_col = tbl_koriko_pct.columns[0]
        ax2.plot(x_pts, tbl_koriko_pct[last_col].values, "o-",
                 color="#e74c3c", linewidth=2.5, markersize=10, label=last_col)
        ax2.plot(x_pts, tbl_koriko_pct[first_col].values, "s--",
                 color="#2ecc71", linewidth=2, markersize=8, label=first_col, alpha=0.8)
        for xp, y in zip(x_pts, tbl_koriko_pct[last_col].values):
            ax2.text(xp + 0.05, y + 3, f"{y:.0f}%", fontsize=10, color="#e74c3c", fontweight="bold")
    ax2.set_xticks(x_pts); ax2.set_xticklabels(tbl_koriko_pct.index, fontsize=9)
    ax2.set_ylabel("Proporsi (%)"); ax2.set_ylim(-5, 95)
    ax2.set_title("Eskalasi Risiko"); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[2])
    sns.heatmap(tbl_koriko_n, ax=ax3, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8})
    ax3.set_title("Heatmap Jumlah Pasien")
    ax3.set_xlabel("Label Risiko"); ax3.set_ylabel("Durasi")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("💡 Tidak pakai = 0% Risiko Tinggi → Jangka Panjang = 76%. Pola eskalasi sangat konsisten.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-5: GENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_gender:
    st.markdown("### BQ-5 · Pengaruh Jenis Kelamin terhadap Distribusi Risiko")

    tbl_gender_pct = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"], normalize="index")
    tbl_gender_pct = tbl_gender_pct.reindex(columns=[r for r in RISK_ORDER if r in tbl_gender_pct.columns])
    tbl_gender_pct = tbl_gender_pct.mul(100).round(1)
    tbl_gender_n   = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"])
    ct             = pd.crosstab(df["Jenis_Kelamin"], df["Label_Risiko"])
    chi2_stat, p_chi, dof = 0, 1, 0
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2_stat, p_chi, dof, _ = chi2_contingency(ct)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    GENDER_COLORS = {"Laki-laki": "#3498db", "Perempuan": "#e91e8c"}
    genders = [g for g in ["Laki-laki", "Perempuan"] if g in tbl_gender_pct.index]

    ax1 = axes[0]
    x = np.arange(len(tbl_gender_pct.columns)); w = 0.35
    for i, gender in enumerate(genders):
        vals = tbl_gender_pct.loc[gender]
        bars = ax1.bar(x + (i - 0.5) * w, vals, width=w,
                       color=GENDER_COLORS[gender], alpha=0.82, label=gender, edgecolor="white")
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
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
        n_gender = tbl_gender_n.loc[gender].sum() if gender in tbl_gender_n.index else 0
        ax.text(0, 0, f"{gender}\nn={n_gender}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=GENDER_COLORS[gender])
        ax.set_title(f"Distribusi Risiko\n{gender}")

    handles = [mpatches.Patch(color=RISK_PALETTE[r], label=r)
               for r in RISK_ORDER if r in df["Label_Risiko"].unique()]
    fig.legend(handles=handles, title="Label Risiko", loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    kesimpulan = "signifikan" if p_chi < 0.05 else "tidak signifikan"
    st.info(f"💡 Perbedaan gender {kesimpulan} (p={p_chi:.4f}). Jenis kelamin bukan faktor pembeda utama.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB BQ-6: REFRAKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_refraksi:
    st.markdown("### BQ-6 · Kelainan Refraksi pada Pasien Risiko Glaukoma")

    tbl_refraksi = pd.crosstab(df["Kelainan_Refraksi"], df["Label_Risiko"], normalize="index")
    tbl_refraksi = tbl_refraksi.reindex(columns=[r for r in RISK_ORDER if r in tbl_refraksi.columns])
    tbl_refraksi = tbl_refraksi.mul(100).round(1)
    tbl_refraksi = tbl_refraksi.reindex([r for r in REFRAKSI_ORDER if r in tbl_refraksi.index])

    risiko_tinggi_df = df[df["Label_Risiko"] == "Risiko Tinggi"]
    komposisi_tinggi = (risiko_tinggi_df["Kelainan_Refraksi"]
                        .value_counts(normalize=True).mul(100)
                        .reindex(REFRAKSI_ORDER).fillna(0).round(1))
    ct_r = pd.crosstab(df["Kelainan_Refraksi"], df["Label_Risiko"])
    p_r  = 1.0
    if ct_r.shape[0] >= 2 and ct_r.shape[1] >= 2:
        _, p_r, _, _ = chi2_contingency(ct_r)

    REFRAKSI_COLORS = ["#95a5a6", "#3498db", "#1a5276", "#f39c12", "#e67e22"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax1 = axes[0]
    x = np.arange(len(tbl_refraksi)); width = 0.26
    for i, (risk, color) in enumerate(zip(
        tbl_refraksi.columns, [RISK_PALETTE[r] for r in tbl_refraksi.columns]
    )):
        bars = ax1.bar(x + (i - 1) * width, tbl_refraksi[risk], width=width,
                       color=color, alpha=0.85, label=risk, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 6:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                         f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace(" ", "\n") for r in tbl_refraksi.index], fontsize=8)
    ax1.set_ylabel("Proporsi (%)"); ax1.set_ylim(0, 65)
    ax1.set_title(f"Proporsi Risiko per Jenis Refraksi\n(p={p_r:.4f})")
    ax1.legend(title="Label Risiko", fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("white")
    mask = komposisi_tinggi > 0
    vals_nz   = komposisi_tinggi[mask]
    labels_pie = vals_nz.index.tolist()
    colors_pie = [REFRAKSI_COLORS[REFRAKSI_ORDER.index(l)]
                  for l in labels_pie if l in REFRAKSI_ORDER]
    if len(vals_nz) > 0:
        wedges, _, autotexts = ax2.pie(
            vals_nz, colors=colors_pie,
            autopct="%1.1f%%", startangle=90, pctdistance=0.72,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")
        ax2.legend(wedges, labels_pie, title="Jenis Refraksi", fontsize=9,
                   bbox_to_anchor=(1.35, 0.5), loc="center right")
    ax2.set_title(f"Komposisi Refraksi\npada Risiko Tinggi (n={len(risiko_tinggi_df)})")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("💡 Miopia Sedang/Berat memiliki proporsi Risiko Tinggi tertinggi.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(" Dashboard Analisis Risiko Glaukoma · Dataset sintetik — tidak untuk keperluan klinis")
