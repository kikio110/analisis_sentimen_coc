"""Latih model + hasilkan data dashboard. Pakai: python train.py clash_of_clans_reviews.csv"""
import sys, json, joblib, time
from collections import Counter
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from wordcloud import WordCloud
from preprocessing import preprocess, lexicon_sentiment

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "clash_of_clans_reviews.csv")
n_raw = len(df)
df = df.dropna().drop_duplicates().reset_index(drop=True)
n_clean = len(df)

pre = df["Review"].apply(preprocess)
df["tokens"] = [p[0] for p in pre]
df["text_akhir"] = [p[1] for p in pre]
lex = df["tokens"].apply(lexicon_sentiment)
df["score"] = [l[0] for l in lex]
df["polarity"] = [l[1] for l in lex]
df["n_words"] = df["tokens"].apply(len)

# ---- model (konfigurasi sama dengan notebook) ----
tfidf = TfidfVectorizer(max_features=5000, min_df=17, max_df=0.8)
X = tfidf.fit_transform(df["text_akhir"])
y = df["polarity"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Naive Bayes": BernoulliNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}
metrics = []
for name, m in models.items():
    t = time.time(); m.fit(Xtr, ytr); p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    metrics.append(dict(model=name, acc_train=accuracy_score(ytr, m.predict(Xtr)),
                        acc_test=accuracy_score(yte, p), precision=pr, recall=rc, f1=f1))
    print(name, round(metrics[-1]["acc_test"], 4), f"{time.time()-t:.0f}s")

best = models["Logistic Regression"]  # dipakai untuk inferensi, seperti di notebook
labels = ["negative", "positive"]
cm = confusion_matrix(yte, best.predict(Xte), labels=labels).tolist()
joblib.dump({"tfidf": tfidf, "model": best, "labels": list(best.classes_)}, "model.joblib", compress=3)

# ---- data dashboard (EDA) ----
def top_words(series, n=15):
    c = Counter(w for toks in series for w in toks)
    return c.most_common(n)

hist, edges = np.histogram(df["n_words"].clip(upper=100), bins=20)
bar = lambda d: d.sort_values("score")
dash = dict(
    n_raw=n_raw, n_clean=n_clean,
    polarity=df["polarity"].value_counts().to_dict(),
    score_dist={str(k): int(v) for k, v in df["score"].clip(-15, 15).value_counts().sort_index().items()},
    len_hist=dict(counts=hist.tolist(), edges=edges.tolist()),
    len_mean=float(df["n_words"].mean()), len_median=float(df["n_words"].median()),
    len_by_polarity=df.groupby("polarity")["n_words"].mean().to_dict(),
    top_words_all=top_words(df["tokens"], 20),
    top_words_pos=top_words(df.loc[df.polarity == "positive", "tokens"]),
    top_words_neg=top_words(df.loc[df.polarity == "negative", "tokens"]),
    top_pos_reviews=df[df.polarity == "positive"].nlargest(8, "score")[["Review", "score"]].values.tolist(),
    top_neg_reviews=df[df.polarity == "negative"].nsmallest(8, "score")[["Review", "score"]].values.tolist(),
    metrics=metrics, confusion=dict(labels=labels, matrix=cm),
)
json.dump(dash, open("dashboard_data.json", "w", encoding="utf-8"), ensure_ascii=False)

for name, mask in [("all", df.polarity.notna()), ("positive", df.polarity == "positive"), ("negative", df.polarity == "negative")]:
    text = " ".join(df.loc[mask, "text_akhir"])
    WordCloud(width=800, height=450, background_color="white", min_font_size=10).generate(text).to_file(f"assets/wc_{name}.png")
print("selesai:", n_raw, "->", n_clean, dash["polarity"])
