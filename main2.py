import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Sentimen Ulasan", layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Mobile Banking")

# Fungsi Analisis Sentimen
def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment_label(score):
    if score > 0:
        return 'Positif'
    elif score < 0:
        return 'Negatif'
    else:
        return 'Netral'

# Sidebar Upload File
st.sidebar.header("📥 Upload File CSV")
uploaded_file = st.sidebar.file_uploader("Pilih file ulasan", type=["csv"])

# Proses jika file diupload
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_columns = {'polarity', 'text_stopword', 'text_akhir'} # Memastikan kolom yang dibutuhkan ada
    if not required_columns.issubset(df.columns):
        st.error(f"❌ Dataset harus mengandung kolom: {', '.join(required_columns)}")
    else:
        st.success("✅ File berhasil dimuat!")

        # Analisis Sentimen dengan TextBlob
        df['polarity'] = df['polarity'].apply(get_polarity)
        df['sentiment'] = df['polarity'].apply(get_sentiment_label)

        # Visualisasi Distribusi Sentimen
        st.subheader("📊 Distribusi Sentimen")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='sentiment', palette='Set2', order=['Positif', 'Netral', 'Negatif'], ax=ax)
            ax.set_ylabel("Jumlah")
            ax.set_xlabel("Sentimen")
            ax.set_title("Distribusi Sentimen Pengguna")
            st.pyplot(fig)
        with col2:
            st.metric("Total Ulasan", len(df))
            st.metric("Positif", (df['sentiment'] == 'Positif').sum())
            st.metric("Netral", (df['sentiment'] == 'Netral').sum())
            st.metric("Negatif", (df['sentiment'] == 'Negatif').sum())

        # WordCloud berdasarkan Sentimen
        st.subheader("☁️ WordCloud Berdasarkan Sentimen")
        tab1, tab2, tab3 = st.tabs(["😊 Positif", "😐 Netral", "😞 Negatif"])
        with tab1:
            st.markdown("**WordCloud Ulasan Positif**")
            text_pos = " ".join(df[df['sentiment'] == 'Positif']['text_stopword'].dropna())
            if text_pos:
                wordcloud_pos = WordCloud(width=1000, height=500, background_color='white').generate(text_pos)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud_pos, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("Tidak ada data ulasan positif.")
        with tab2:
            st.markdown("**WordCloud Ulasan Netral**")
            text_neutral = " ".join(df[df['sentiment'] == 'Netral']['text_stopword'].dropna())
            if text_neutral:
                wordcloud_neutral = WordCloud(width=1000, height=500, background_color='white').generate(text_neutral)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud_neutral, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("Tidak ada data ulasan netral.")
        with tab3:
            st.markdown("**WordCloud Ulasan Negatif**")
            text_neg = " ".join(df[df['sentiment'] == 'Negatif']['text_stopword'].dropna())
            if text_neg:
                wordcloud_neg = WordCloud(width=1000, height=500, background_color='white').generate(text_neg)
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud_neg, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("Tidak ada data ulasan negatif.")

        # Menampilkan Ulasan Positif dan Negatif
        st.subheader("📝 Top Ulasan")
        df_sorted = df.sort_values(by='text_akhir', ascending=False)
        st.markdown("#### 👍 Ulasan Positif")
        for i, row in df_sorted[df_sorted['sentiment'] == 'Positif'].head(5).iterrows():
            st.success(f"{row['text_akhir']}")
        st.markdown("#### 👎 Ulasan Negatif")
        for i, row in df_sorted[df_sorted['sentiment'] == 'Negatif'].tail(5).iterrows():
            st.error(f"{row['text_akhir']}")
        st.markdown("#### 😐 Ulasan Netral")
        for i, row in df_sorted[df_sorted['sentiment'] == 'Netral'].head(5).iterrows():
            st.info(f"{row['text_akhir']}")

        # Logistic Regression untuk Analisis Sentimen
        st.subheader("🤖 Analisis Sentimen dengan Logistic Regression")
        text_data = df['text_stopword'].astype(str) # Menggunakan 'text_stopword' yang sudah diproses
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['sentiment'])  # Mengubah label sentimen menjadi numerik

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

        # Vectorize teks menggunakan TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)  # Batasi fitur untuk efisiensi
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Melatih model Logistic Regression
        model = LogisticRegression(solver='liblinear')  # Gunakan solver yang sesuai untuk data kecil
        model.fit(X_train_vec, y_train)

        # Membuat prediksi
        y_pred = model.predict(X_test_vec)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_) # menggunakan label_encoder
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Menampilkan hasil
        st.write(f"**Akurasi:** {accuracy:.2f}")
        st.text("**Laporan Klasifikasi:**")
        st.text(report)

        # Visualisasi Confusion Matrix
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_) # menggunakan label_encoder
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
else:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
