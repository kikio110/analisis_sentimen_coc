import json
from pathlib import Path
import joblib, pandas as pd, plotly.express as px, streamlit as st
from preprocessing import preprocess, lexicon_sentiment

BASE = Path(__file__).parent
COLORS = {"positive": "#2ca02c", "negative": "#d62728"}
st.set_page_config(page_title="Analisis Sentimen Clash of Clans", page_icon="⚔️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_dash():
    return json.load(open(BASE / "dashboard_data.json", encoding="utf-8"))


bundle, D = load_model(), load_dash()


def predict(texts):
    final = [preprocess(t)[1] for t in texts]
    X = bundle["tfidf"].transform(final)
    pred = bundle["model"].predict(X)
    proba = bundle["model"].predict_proba(X).max(axis=1)
    return pred, proba


st.title("⚔️ Analisis Sentimen Ulasan Clash of Clans")
st.caption("Ulasan Google Play Store (bahasa Indonesia) · pelabelan berbasis lexicon · model Logistic Regression + TF-IDF")
tab1, tab2, tab3 = st.tabs(["📊 Dashboard EDA", "🔮 Prediksi Sentimen", "🤖 Performa Model"])

# ---------------- Dashboard ----------------
with tab1:
    pol = D["polarity"]
    total = sum(pol.values())
    c = st.columns(5)
    c[0].metric("Ulasan mentah", f"{D['n_raw']:,}")
    c[1].metric("Setelah dibersihkan", f"{D['n_clean']:,}")
    c[2].metric("Positif", f"{pol['positive']:,}", f"{pol['positive']/total:.1%}", delta_color="off")
    c[3].metric("Negatif", f"{pol['negative']:,}", f"{pol['negative']/total:.1%}", delta_color="off")
    c[4].metric("Rata-rata panjang", f"{D['len_mean']:.1f} kata")

    a, b = st.columns(2)
    with a:
        fig = px.pie(names=list(pol), values=list(pol.values()), hole=0.45, color=list(pol),
                     color_discrete_map=COLORS, title="Distribusi sentimen")
        st.plotly_chart(fig, width="stretch")
    with b:
        sd = pd.DataFrame({"skor": [int(k) for k in D["score_dist"]], "jumlah": list(D["score_dist"].values())})
        sd["sentimen"] = sd["skor"].apply(lambda s: "positive" if s >= 0 else "negative")
        fig = px.bar(sd, x="skor", y="jumlah", color="sentimen", color_discrete_map=COLORS,
                     title="Distribusi skor polaritas (dipotong -15..15)")
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        e, n = D["len_hist"]["edges"], D["len_hist"]["counts"]
        lh = pd.DataFrame({"panjang (kata)": [f"{int(e[i])}-{int(e[i+1])}" for i in range(len(n))], "jumlah": n})
        st.plotly_chart(px.bar(lh, x="panjang (kata)", y="jumlah", title="Distribusi panjang ulasan (setelah preprocessing)"),
                        width="stretch")
        st.caption(f"Median {D['len_median']:.0f} kata · panjang >100 kata digabung ke bin terakhir.")
    with b:
        tw = pd.DataFrame(D["top_words_all"], columns=["kata", "frekuensi"]).sort_values("frekuensi")
        st.plotly_chart(px.bar(tw, x="frekuensi", y="kata", orientation="h", title="20 kata paling sering muncul"),
                        width="stretch")

    st.subheader("Kata dominan per sentimen")
    a, b = st.columns(2)
    for col, key, name in [(a, "top_words_pos", "positive"), (b, "top_words_neg", "negative")]:
        with col:
            t = pd.DataFrame(D[key], columns=["kata", "frekuensi"]).sort_values("frekuensi")
            st.plotly_chart(px.bar(t, x="frekuensi", y="kata", orientation="h", title=f"Top kata - {name}",
                                   color_discrete_sequence=[COLORS[name]]), width="stretch")

    st.subheader("Word Cloud")
    w = st.columns(3)
    for col, k, cap in zip(w, ["all", "positive", "negative"], ["Semua ulasan", "Positif", "Negatif"]):
        col.image(str(BASE / "assets" / f"wc_{k}.png"), caption=cap, width="stretch")

    st.subheader("Contoh ulasan paling ekstrem")
    a, b = st.columns(2)
    with a:
        st.markdown("**Paling positif**")
        for r, s in D["top_pos_reviews"][:5]:
            st.info(f"{r[:300]}{'…' if len(r) > 300 else ''}  \n_skor {s}_")
    with b:
        st.markdown("**Paling negatif**")
        for r, s in D["top_neg_reviews"][:5]:
            st.error(f"{r[:300]}{'…' if len(r) > 300 else ''}  \n_skor {s}_")
    st.warning("Catatan: label dibuat dari lexicon, dengan skor ≥ 0 dianggap positif (tidak ada kelas netral), "
               "sehingga ulasan tanpa kata sentimen ikut terhitung positif.")

# ---------------- Prediksi ----------------
with tab2:
    st.subheader("Prediksi satu ulasan")
    txt = st.text_area("Tulis ulasan (bahasa Indonesia):", height=120,
                       placeholder="contoh: game nya seru banget tapi sering lag dan loading lama")
    if st.button("Prediksi", type="primary") and txt.strip():
        pred, proba = predict([txt])
        tokens, final = preprocess(txt)
        score, lex_label = lexicon_sentiment(tokens)
        icon = "😊" if pred[0] == "positive" else "😠"
        st.markdown(f"### {icon} Sentimen: **{pred[0].upper()}** ({proba[0]:.1%} keyakinan model)")
        st.caption(f"Teks setelah preprocessing: `{final or '(kosong)'}` · skor lexicon: {score} ({lex_label})")

    st.divider()
    st.subheader("Prediksi massal (CSV)")
    up = st.file_uploader("Unggah CSV dengan kolom `Review`", type="csv")
    if up:
        df = pd.read_csv(up)
        if "Review" not in df.columns:
            st.error("Kolom `Review` tidak ditemukan.")
        else:
            df = df.dropna(subset=["Review"]).head(5000).copy()
            df["sentimen"], df["keyakinan"] = predict(df["Review"].tolist())
            st.plotly_chart(px.pie(df, names="sentimen", color="sentimen", color_discrete_map=COLORS, hole=0.45),
                            width="stretch")
            st.dataframe(df, width="stretch")
            st.download_button("Unduh hasil", df.to_csv(index=False).encode("utf-8"), "hasil_sentimen.csv", "text/csv")
            st.caption("Maksimal 5.000 baris diproses.")

# ---------------- Model ----------------
with tab3:
    m = pd.DataFrame(D["metrics"]).rename(columns={"model": "Model", "acc_train": "Akurasi Train",
                     "acc_test": "Akurasi Test", "precision": "Precision", "recall": "Recall", "f1": "F1"})
    long = m.melt(id_vars="Model", value_vars=["Akurasi Train", "Akurasi Test"], var_name="Data", value_name="Akurasi")
    st.plotly_chart(px.bar(long, x="Model", y="Akurasi", color="Data", barmode="group",
                           title="Perbandingan akurasi model"), width="stretch")
    st.dataframe(m.sort_values("Akurasi Test", ascending=False).style.format(
        {c: "{:.2%}" for c in m.columns if c != "Model"}), width="stretch", hide_index=True)
    cm = D["confusion"]
    st.plotly_chart(px.imshow(cm["matrix"], x=cm["labels"], y=cm["labels"], text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Prediksi", y="Aktual"), title="Confusion matrix - Logistic Regression (data uji)"),
                    width="stretch")
    st.caption("Model dilatih untuk meniru label lexicon, jadi akurasi mengukur kecocokan dengan lexicon, bukan dengan label manusia.")
