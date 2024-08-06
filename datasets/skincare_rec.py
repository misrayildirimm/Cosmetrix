import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Bellek kullanımını optimize etmek için veri okuma ve temizleme işlemleri
df_products = pd.read_csv("pythonProject3/product_info.csv")

# Büyük veri dosyalarını parça parça okuyarak birleştiriyoruz
chunks = [pd.read_csv(f"pythonProject3/reviews_{i}-{j}.csv") for i, j in
          [(0, 250), (250, 500), (500, 750), (750, 1250), (1250, 'end')]]
df_reviews = pd.concat(chunks, ignore_index=True)
del chunks  # Belleği boşalt

df_skincare = pd.read_excel('pythonProject3/output.xlsx')

# İlgili sütunları filtreleyerek belleği optimize etme
skincare_products = df_products[df_products['primary_category'] == 'Skincare']
skincare_products = skincare_products[['product_id', 'product_name', 'price_usd', 'rating', 'loves_count', 'reviews']]

# Veri setlerini birleştirme
skincaredf = pd.merge(skincare_products, df_skincare, on='product_id', how='inner')
skincaredf = skincaredf.drop(columns=["primary_category", "secondary_category", "product_name_y", "tertiary_category", "category", "Unnamed: 9"])

# Eksik değerleri doldurma
skincaredf[['problem2', 'problem3']] = skincaredf[['problem2', 'problem3']].fillna('')

# Cilt problemlerini birleştirerek yeni bir sütun oluşturma
skincaredf['problems'] = skincaredf[['problem1', 'problem2', 'problem3']].agg(','.join, axis=1)

# İncelemeleri ürün bilgileri ile birleştirme
skincare_reviews = pd.merge(df_reviews, skincaredf, on='product_id', how='inner')
skincare_reviews = skincare_reviews.drop(columns=['Unnamed: 0', 'author_id', 'helpfulness',
                                                   'total_feedback_count', 'total_neg_feedback_count',
                                                   'total_pos_feedback_count', 'submission_time',
                                                   'review_title', 'skin_tone', 'product_name_x', 'price_usd_x', 'brand_name', 'rating_y'])
del df_reviews  # Belleği boşalt

# Cilt tipi dağılımını hesaplama ve veri setine ekleme
skin_type_counts = skincare_reviews.groupby(['product_id', 'skin_type']).size().unstack(fill_value=0)
skincaredf = pd.merge(skincaredf, skin_type_counts, on='product_id', how='inner')

# Ürün başına ortalama puanları hesaplama ve veri setine ekleme
average_ratings = skincare_reviews.groupby('product_id')['rating_x'].mean().reset_index()
skincaredf = pd.merge(skincaredf, average_ratings, on='product_id', how='inner')

# Cilt tipine göre her bir ürün için ortalama puanı hesaplama
skin_type_ratings = skincare_reviews.groupby(['product_id', 'skin_type'])['rating_x'].mean().unstack(fill_value=0)
skincaredf = pd.merge(skincaredf, skin_type_ratings, on='product_id', how='left', suffixes=('', '_skin_type'))

del skincare_reviews  # Belleği boşalt

# Etiket gruplarını kullanarak problem etiketlerini gruplama
etiket_gruplari = {
    'Nemlendirme-Cilt Kuruluğu': ['Cilt kuruluğu', 'Nemlendirme', 'Hızlı nemlendirme', 'Cilt Kuruluğu', 'Nem bombası', 'Dudak nemlendirme', 'Besleyici', 'Kuru cilt', 'Dengeleyici nemlendirici'],
    'Elastikiyet ve Sıkılaştırma': ['Cilt elastikiyeti kaybı', 'Sıkılaştırıcı serum', 'Sıkılaştırma', 'Cilt sıkılaştırma', 'Cilt Elastikiyeti Kaybı', 'Kollajen desteği', 'Yeniden yapılandırma ve aydınlatma', 'Kolajen desteği', 'Kolajen'],
    'Göz Altı Problemleri': ['Göz altı torbaları', 'Göz altı sorunları', 'Göz Altı Sorunları'],
    'Pigmentasyon ve Koyu Lekeler': ['Pigmentasyon sorunları (Hiperpigmentasyon)', 'Hiperpigmentasyon (Koyu lekeler)', 'Koyu lekeler', 'Pigmentasyon Sorunları', 'Pigmentasyon sorunları (Koyu lekeler)', 'Hiperpigmentasyon', 'Toner', 'Bronzlaştırma hatalarını düzeltme'],
    'Akne-Sivilce': ['Sivilce (Akne)', 'Sivilce (Kistik akne)', 'Akne', 'Akne tedavisi', 'Sivilce', 'Sivilce izleri', 'Salisilik asit'],
    'Aydınlatma ve Beyazlatma': ['Cilt aydınlatma', 'Aydınlatıcı nemlendirici', 'Aydınlatıcı serum', 'Aydınlatıcı yağ', 'Beyazlatma', 'Aydınlatma'],
    'Siyah Nokta - Peeling': ['Temizleme', 'Eksfoliasyon', 'Peeling', 'Gözenek temizleme', 'Siyah nokta temizleme', 'Cilt temizliği', 'Gözenek temizliği', 'Temizleme uçları', 'Siyah nokta', 'Gözenek tıkanıklığı', 'Gözenek Sorunları', 'Eksfoliasyon ve dolgunlaştırma'],
    'Gece Bakımı ve Ürünleri': ['Gece kremi', 'Gece serumu', 'Gece bakımı', 'Gece maskesi', 'Gece tedavisi', 'Uyku desteği', 'Uyku kalitesi'],
    'Güneş Koruması ve UV Hasarı': ['Güneş koruması', 'UV hasarı onarımı', 'Güneş koruma'],
    'Cilt Hassasiyeti ve Kızarıklık': ['Cilt hassasiyeti (Kızarıklık', 'Kızarıklık', 'Cilt hassasiyeti'],
    'Maskeler': ['Dengeleyici maske', 'Maske', 'Yüz maskesi', 'Maske tedavisi', 'Maske uygulama'],
    'Cilt Yenileme ve Onarım': ['Cilt yenileme', 'Yenileyici terapi', 'Yenileme', 'Onarım', 'Bariyer güçlendirme', 'Bariyer onarımı'],
    'Soluk ve Pürüzlü Cilt Problemleri': ['Dolgunlaştırma', 'Cilt parlaklığı', 'Canlandırma', 'Pürüzsüzleştirme'],
    'Antioksidan ve Koruma': ['Antioksidan koruma', 'Antioksidan', 'Koruyucu'],
    'Yağ Kontrolü': ['Yağ kontrolü', 'Yağlı cilt'],
    'Tüy Sorunları': ['Tüy temizleme', 'Tüy alma', 'Tıraş'],
    'Detoks ve Arındırma': ['Detoks', 'Detox', 'Karaciğer detoksu', 'Arındırma'],
    'Sindirim ve Metabolizma Sorunları': ['Sindirim sağlığı', 'Sindirim desteği', 'Metabolizma artırma'],
    'Kadın Sağlığı': ['Vajinal sağlık', 'Menopoz desteği', 'PMS desteği', 'Prenatal destek'],
    'Enerji ve Stres Sorunları': ['Enerji artırma', 'Stres yönetimi', 'Adrenal yorgunluk'],
    'Takviye-Sağlık Destekleri': ['Bağışıklık desteği', 'Omega-3 desteği', 'Adaptogen desteği', 'Beyin sağlığı', 'Vitamin ve takviye saklama'],
    'Genel Sağlık': ['Genel cilt sağlığı', 'Su tüketimi', 'Saç ve cilt sağlığı', 'Yorgunluk karşıtı', 'Rahatlama'],
    'Yaşlanma-Kırışıklık': ['Yaşlanma karşıtı', 'Kırışıklıklar (İnce çizgiler)', 'Boyun kırışıklıkları', 'Kırışıklıklar', 'İnce çizgiler', 'Anti-aging']
}

for yeni_etiket, eski_etiketler in etiket_gruplari.items():
    skincaredf['problem1'] = skincaredf['problem1'].replace(eski_etiketler, yeni_etiket)

# TF-IDF hesaplaması için metin alanını oluşturma
skincaredf['combined_problems'] = skincaredf[['problem1', 'problem2', 'problem3', ]].fillna('').agg(' '.join, axis=1)

# TF-IDF hesaplaması
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(skincaredf['combined_problems'])

# Cosine similarity hesaplaması
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Yeni puan hesaplama fonksiyonu
def calculate_new_score(row, min_reviews, max_reviews, min_loves, max_loves):
    # Min-Max Normalizasyonu
    normalized_reviews = (row['reviews'] - min_reviews) / (max_reviews - min_reviews)
    normalized_loves = 1 + 4 * (row['loves_count'] - min_loves) / (max_loves - min_loves)

    # Yeni Skoru Hesaplama (rating kolonuna daha fazla ağırlık verilerek)
    raw_score = row['rating'] * 2 * (1 + normalized_reviews) * normalized_loves

    return raw_score

# Skoru 1 ile 5 arasında ölçeklendirme fonksiyonu
def scale_score(raw_score, min_score, max_score):
    return 1 + 4 * (raw_score - min_score) / (max_score - min_score)

# Cilt problemi girdiğinde benzer ürünleri önerme fonksiyonu
def get_recommendations_by_problem(problem, cosine_sim=cosine_sim, top_n=15):
    # Kullanıcının girdiği cilt problemine göre TF-IDF vektörünü hesapla
    problem_tfidf = tfidf_vectorizer.transform([problem])

    # Problemle tüm ürünler arasındaki benzerlik puanlarını al
    sim_scores = cosine_similarity(problem_tfidf, tfidf_matrix).flatten()

    # Benzerlik puanlarına göre ürünleri sırala
    sim_scores_indices = sim_scores.argsort()[::-1][:top_n]

    # En benzer ürünleri al
    recommended_products = skincaredf.iloc[sim_scores_indices].copy()

    # Yeni skoru hesaplama ve veri setine ekleme
    recommended_products = normalize_and_calculate_scores(recommended_products)

    # Yeni skora göre sıralama yap
    recommended_products = recommended_products.sort_values(by='new_score', ascending=False)

    return recommended_products[['product_name_x', 'price_usd', 'rating',"reviews","loves_count", 'problems']]

# Örnek olarak bir cilt problemi vererek benzer ürünleri bulma
problem = 'yaşlanma'
recommended_products = get_recommendations_by_problem(problem)

print(recommended_products)
