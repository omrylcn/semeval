# %%
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import torch
import numpy as np
import os 

# Cihazı belirle (Mac için MPS, CUDA yoksa CPU)
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
print(f"Kullanılan cihaz: {device}")

# %%
# Model yükle
model_name = 'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr'
#model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_name = "models/nezahat-turkce-embedding-bge-m3"
model_name = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
model_name = "Alibaba-NLP/gte-multilingual-base"
print(f"Model yükleniyor: {model_name}")
matryoshka_dim = 768

model = SentenceTransformer(model_name,trust_remote_code=True)
print(f"Model başarıyla yüklendi!")

model_dict = {}

model_dict[model_name] = model

# %% [markdown]
# ### Test 1: Bilgi Erişimi ve Sıralama (Information Retrieval & Ranking)
# 
# 

# %%
# =============================================================================
# CORPUS: 35 Finansal Doküman (KAP, Haber, SSS, Analiz karışık)
# =============================================================================

corpus_docs_list = [
    # --- ENFLASYON & PARA POLİTİKASI (c0-c5) ---
    "Türkiye Cumhuriyet Merkez Bankası (TCMB), Para Politikası Kurulu toplantısında politika faizini 500 baz puan artırarak yüzde 30'a yükseltti. Enflasyonla mücadele kapsamında sıkı para politikası sürdürülecek.", # c0
    "Tüketici Fiyat Endeksi (TÜFE) Ekim ayında aylık bazda yüzde 3.4 artış gösterdi. Yıllık enflasyon yüzde 61.5 seviyesinde gerçekleşti. En yüksek artış gıda grubunda gözlendi.", # c1
    "Merkez Bankası faiz artırımı piyasalarda olumlu karşılandı. Analistler, sıkı para politikasının enflasyon beklentilerini düşüreceğini öngörüyor.", # c2
    "Enflasyon hedeflemesi rejimi, merkez bankalarının fiyat istikrarını sağlamak için kullandığı bir para politikası stratejisidir. Türkiye 2006'dan beri bu rejimi uygulamaktadır.", # c3
    "Yüksek enflasyon ortamında tüketicilerin satın alma gücü azalır. Sabit gelirli vatandaşlar en çok etkilenen kesimdir. Devlet, enflasyonla mücadele için sıkı maliye politikası uygulamaktadır.", # c4
    "Üretici Fiyat Endeksi (ÜFE) Ekim ayında yıllık bazda yüzde 55.8 arttı. ÜFE artışının TÜFE'ye yansıması birkaç ay içinde bekleniyor.", # c5
    
    # --- HİSSE SENEDİ & BORSA (c6-c12) ---
    "BIST 100 endeksi günü yüzde 2.8 artışla 8.450 seviyesinden kapattı. Bankacılık endeksi günün yıldızı oldu. Yabancı yatırımcılar net alıcı konumundaydı.", # c6
    "X Bankası hisseleri bugün yüzde 5.2 değer kazandı. Bankanın açıkladığı güçlü bilanço verileri yatırımcılar tarafından olumlu karşılandı. Öz sermaye karlılığı yüzde 18'e yükseldi.", # c7
    "Halka arz tarihi açıklanan Y Teknoloji şirketi, 2 milyar TL değerleme ile Borsa İstanbul'da işlem görmeye başlayacak. Talep toplaması önümüzdeki hafta başlıyor.", # c8
    "Temettü dağıtımı açısından en cömert şirketler listesinde enerji ve bankacılık sektörü öne çıkıyor. Z Bankası, hisse başına 1.5 TL temettü dağıtacağını açıkladı.", # c9
    "Teknik analizde direnç seviyesi 8.500'de görülüyor. Bu seviye aşılırsa BIST 100'ün 9.000'e doğru yükseliş potansiyeli var. RSI göstergesi 65 seviyesinde.", # c10
    "Hisse senedi yatırımında temel analiz yöntemi, şirketlerin finansal tablolarını, bilanço verilerini ve sektör dinamiklerini incelemeyi içerir. F/K oranı önemli bir göstergedir.", # c11
    "Borsa İstanbul'da işlem gören şirketlerin toplam piyasa değeri 7.2 trilyon TL'ye ulaştı. Yabancı yatırımcı oranı yüzde 45 seviyesinde.", # c12
    
    # --- KREDİ & BANKACILIK (c13-c18) ---
    "Konut kredisi faiz oranları son 3 ayda 200 baz puan arttı. Bankalar, yüksek enflasyon nedeniyle kredi faizlerini yukarı çekti. Ortalama konut kredisi faizi yüzde 36 seviyesinde.", # c13
    "Kredi kartı borçlarını ödeyemeyen vatandaş sayısı artıyor. Bankalar, taksitlendirme imkanları sunuyor. Asgari ödeme tuzağına düşmemek için tam ödeme öneriliyor.", # c14
    "Findeks kredi notu 1000-1900 aralığında belirleniyor. 1400'ün üzerindeki notlar iyi kabul ediliyor. Düzenli ödeme yapanların notu yükseliyor.", # c15
    "Ticari kredilerde KGF (Kredi Garanti Fonu) kefaleti ile KOBİ'lere düşük faizli finansman imkanı sağlanıyor. Başvurular bankalar aracılığıyla yapılıyor.", # c16
    "Tüketici kredisi faizleri yüzde 40 seviyesini aştı. Bireysel ihtiyaç kredisi talepleri düşüş gösteriyor. Bankalar riskli kredilerde teminat istiyor.", # c17
    "Kredili mevduat hesabı (kredi kartından farklı), müşterilerin belirli bir limite kadar çek kesmesine izin verir. Sadece kullanılan tutar için faiz ödenir.", # c18
    
    # --- DÖVİZ & ALTIN (c19-c24) ---
    "Dolar/TL kuru bugün yüzde 0.8 değer kazanarak 32.50 seviyesine yükseldi. TCMB'nin döviz rezervlerinde azalma görüldü. Piyasa, yeni faiz kararını bekliyor.", # c19
    "Euro/TL paritesi 35.20 seviyesinde işlem görüyor. Avrupa Merkez Bankası'nın faiz kararları Türk lirasını da etkiliyor.", # c20
    "Altın ons fiyatı 2.050 dolar seviyesini test ediyor. Gram altın ise 1.850 TL'den işlem görüyor. Yatırımcılar enflasyona karşı altına yöneliyor.", # c21
    "Döviz tevdiat hesaplarında tutulan toplam miktar 280 milyar dolara yaklaştı. Vatandaşlar dolarizasyon eğilimi gösteriyor.", # c22
    "TCMB, döviz piyasalarında istikrarı sağlamak için swap ihaleleri düzenliyor. Döviz likiditesi kontrol altında tutuluyor.", # c23
    "Kripto para piyasasında Bitcoin 43.000 dolar seviyesini gördü. Türkiye'de kripto para kullanımı yaygınlaşıyor ancak düzenleme belirsizliği sürüyor.", # c24
    
    # --- TAHVİL & BORÇLANMA (c25-c29) ---
    "Hazine, 10 yıllık tahvil ihalesinde 18 milyar TL borçlandı. Bileşik faiz yüzde 28.5 olarak gerçekleşti. Talep/teklif oranı 2.1 seviyesinde.", # c25
    "Eurobond faizleri geriledi. Türkiye'nin 2034 vadeli dolar cinsi tahvilinin getirisi yüzde 7.8'e düştü. Risk algısı iyileşiyor.", # c26
    "Şirket tahvilleri piyasasında ihraç hacmi arttı. A+ notu olan şirketler yatırımcılar tarafından tercih ediliyor. Getiri oranları yüzde 30-35 arasında.", # c27
    "Devlet İç Borçlanma Senetleri (DİBS) portföyünde ağırlık artıyor. Bankalar, kredi vermek yerine tahvil almayı tercih ediyor.", # c28
    "Tahvil piyasasında vade uzadıkça getiri yükseliyor. 5 yıllık tahvil faizi yüzde 27 iken, 10 yıllık yüzde 29 seviyesinde.", # c29
    
    # --- MAKRO EKONOMİ (c30-c34) ---
    "Cari açık Eylül ayında 4.2 milyar dolar olarak gerçekleşti. Enerji ithalatı cari açığın ana nedeni. Turizm gelirlerindeki artış açığı sınırladı.", # c30
    "Gayri Safi Yurt İçi Hasıla (GSYİH) üçüncü çeyde yıllık bazda yüzde 4.1 büyüdü. İmalat sanayi ve inşaat sektörü büyümeye katkı sağladı.", # c31
    "İşsizlik oranı yüzde 9.4 seviyesinde. Genç işsizlik ise yüzde 16.8 olarak açıklandı. Hizmet sektöründe istihdam artıyor.", # c32
    "Merkez Bankası brüt döviz rezervleri 140 milyar dolar seviyesinde. Net rezervler ise tartışmalı, swap hariç hesaplama yapılıyor.", # c33
    "Bütçe açığı hedefin üzerinde seyrediyor. Hazine, yılsonu hedefini revize edebilir. Vergi gelirleri beklentilerin altında kaldı.", # c34
]

# Sözlük formatına çevir
corpus = {f'c{i}': doc for i, doc in enumerate(corpus_docs_list)}
print(f"✅ Corpus yüklendi: {len(corpus)} finansal doküman")

# %%
# =============================================================================
# QUERIES: 12 Finansal Sorgu (Her biri 3-5 alakalı dökümana sahip)
# =============================================================================

queries_list = [
    "Merkez Bankası faiz kararları enflasyonu nasıl etkiler?",           
    "BIST 100 endeksinin son durumu nedir?",                             
    "Konut kredisi faiz oranları ne durumda?",                           
    "Dolar kuru bugün nasıl hareket etti?",                              
    "Temettü ödeyen bankalar hangileri?",                                
    "Hazine tahvil ihalesi sonuçları nedir?",                            
    "Findeks kredi notu nasıl hesaplanır?",                              
    "Cari açık verileri açıklandı mı?",                                  
    "Halka arz edilecek şirketler hangileri?",                           
    "Altın fiyatları yükseliyor mu?",                                    
    "Ekonomik büyüme rakamları nasıl?",                                  
    "Enflasyon verileri ve merkez bankası politikaları",                 
]

queries = {f'q{i}': query for i, query in enumerate(queries_list)}

# =============================================================================
# GROUND TRUTH: Puanlı Eşleştirme (Manuel Değerlendirme İçin)
# Format: Dict[query_id, Dict[doc_id, score]]
# Score: 2 = Çok alakalı (tam cevap), 1 = Az alakalı (dolaylı/tanjant geçen)
# =============================================================================

relevant_docs_with_scores = {
    # q0: Merkez Bankası faiz kararları enflasyonu nasıl etkiler?
    'q0': {
        'c0': 2,  # TCMB faiz artırımı + enflasyonla mücadele (TAM CEVAP)
        'c2': 2,  # MB faiz artırımı + enflasyon beklentileri (TAM CEVAP)
        'c3': 1,  # Enflasyon hedeflemesi rejimi (DOLAYLI)
        'c4': 1,  # Yüksek enflasyon etkileri (TANJANT)
        'c1': 1,  # TÜFE verileri (AZ ALAKALI)
    },
    
    # q1: BIST 100 endeksinin son durumu nedir?
    'q1': {
        'c6': 2,  # BIST 100 günlük verisi (TAM CEVAP)
        'c10': 2, # Teknik analiz + BIST 100 direnç seviyeleri (TAM CEVAP)
        'c12': 1, # Borsa İstanbul genel bilgi (DOLAYLI)
        'c7': 1,  # Banka hisseleri (bankacılık endeksi) (TANJANT)
    },
    
    # q2: Konut kredisi faiz oranları ne durumda?
    'q2': {
        'c13': 2, # Konut kredisi faizleri (TAM CEVAP)
        'c17': 1, # Tüketici kredisi faizleri (BENZER AMA FARKLI)
        'c14': 1, # Kredi kartı borçları (KREDİ İLE İLGİLİ)
        'c16': 1, # Ticari krediler (KREDİ İLE İLGİLİ)
    },
    
    # q3: Dolar kuru bugün nasıl hareket etti?
    'q3': {
        'c19': 2, # Dolar/TL kuru günlük (TAM CEVAP)
        'c23': 1, # TCMB döviz piyasası müdahalesi (DOLAYLI)
        'c22': 1, # Döviz tevdiat hesapları (DÖVİZ İLE İLGİLİ)
        'c20': 1, # Euro/TL (DÖVİZ AMA FARKLI PAR)
    },
    
    # q4: Temettü ödeyen bankalar hangileri?
    'q4': {
        'c9': 2,  # Z Bankası temettü dağıtımı (TAM CEVAP)
        'c7': 1,  # X Bankası güçlü bilanço (İLGİLİ AMA TEMETTÜden bahsetmiyor)
    },
    
    # q5: Hazine tahvil ihalesi sonuçları nedir?
    'q5': {
        'c25': 2, # Hazine 10 yıllık tahvil ihalesi (TAM CEVAP)
        'c29': 1, # Tahvil vade-getiri ilişkisi (GENEL BİLGİ)
        'c28': 1, # DİBS portföyü (İLGİLİ)
        'c27': 1, # Şirket tahvilleri (TAHVİL AMA FARKLI)
    },
    
    # q6: Findeks kredi notu nasıl hesaplanır?
    'q6': {
        'c15': 2, # Findeks kredi notu (TAM CEVAP)
        'c14': 1, # Kredi kartı borçları (KREDİ İLE İLGİLİ)
    },
    
    # q7: Cari açık verileri açıklandı mı?
    'q7': {
        'c30': 2, # Cari açık Eylül verileri (TAM CEVAP)
        'c31': 1, # GSYİH büyüme (MAKRO EKONOMİ)
        'c33': 1, # MB döviz rezervleri (İLGİLİ)
    },
    
    # q8: Halka arz edilecek şirketler hangileri?
    'q8': {
        'c8': 2,  # Y Teknoloji halka arz (TAM CEVAP)
        'c12': 1, # Borsa İstanbul piyasa değeri (BORSA İLE İLGİLİ)
    },
    
    # q9: Altın fiyatları yükseliyor mu?
    'q9': {
        'c21': 2, # Altın ons ve gram fiyatları (TAM CEVAP)
    },
    
    # q10: Ekonomik büyüme rakamları nasıl?
    'q10': {
        'c31': 2, # GSYİH büyüme verileri (TAM CEVAP)
        'c32': 1, # İşsizlik oranı (MAKRO EKONOMİ)
        'c30': 1, # Cari açık (MAKRO EKONOMİ)
    },
    
    # q11: Enflasyon verileri ve merkez bankası politikaları (GENİŞ SORGU)
    'q11': {
        'c0': 2,  # TCMB faiz artırımı + enflasyon (TAM CEVAP)
        'c1': 2,  # TÜFE verileri (TAM CEVAP)
        'c2': 2,  # MB faiz + enflasyon beklentileri (TAM CEVAP)
        'c5': 2,  # ÜFE verileri (TAM CEVAP)
        'c3': 1,  # Enflasyon hedeflemesi rejimi (İLGİLİ)
        'c4': 1,  # Enflasyon etkileri (İLGİLİ)
    },
}

# InformationRetrievalEvaluator için Set formatına çevir (skor bilgisi kaybolur)
# Ancak NDCG hesabı için skorları ayrıca saklayacağız
relevant_docs = {qid: set(docs.keys()) for qid, docs in relevant_docs_with_scores.items()}

print(f"✅ {len(queries)} sorgu hazırlandı")
print(f"✅ Ground truth oluşturuldu (puanlı format)")
print(f"\n📊 İstatistikler:")
total_relevant = sum(len(docs) for docs in relevant_docs_with_scores.values())
score_2_count = sum(1 for docs in relevant_docs_with_scores.values() for score in docs.values() if score == 2)
score_1_count = sum(1 for docs in relevant_docs_with_scores.values() for score in docs.values() if score == 1)
print(f"  - Toplam alakalı doküman eşleşmesi: {total_relevant}")
print(f"  - Çok alakalı (skor=2): {score_2_count}")
print(f"  - Az alakalı (skor=1): {score_1_count}")
print(f"  - Ortalama alakalı doküman/sorgu: {total_relevant/len(queries):.1f}")

# %%
print("\n" + "="*70)
print("--- TEST 1: Bilgi Erişimi ve Sıralama (Information Retrieval) ---")
print("="*70)

# InformationRetrievalEvaluator ile test
evaluator = InformationRetrievalEvaluator(
    queries=queries,           # Dict[str, str]: query_id -> query_text
    corpus=corpus,             # Dict[str, str]: doc_id -> doc_text
    relevant_docs=relevant_docs, # Dict[str, Set[str]]: query_id -> set of relevant doc_ids
    name="turkish-financial-ir-test",
    show_progress_bar=True,
    ndcg_at_k=[1, 3, 5, 10],
    map_at_k=[1, 3, 5, 10],
    mrr_at_k=[1, 3, 5, 10],
)

# Modeli değerlendir
print(f"\nModel değerlendiriliyor: {model_name}")
results = evaluator(model, output_path="./")

print("\n" + "="*70)
print("--- TEST 1 SONUÇLARI ---")
print("="*70)
print(f"Model: {model_name}\n")

# Sonuçları güzel formatta yazdır
for metric_name, score in results.items():
    if isinstance(score, float):
        print(f"  {metric_name:30s}: {score:.4f}")
    else:
        print(f"  {metric_name:30s}: {score}")
        
print("="*70)

# Detaylı analiz için manuel arama da yapalım
print("\n" + "="*70)
print("--- Detaylı Sorgu Analizi ---")
print("="*70)

# Corpus ve query'leri encode et
corpus_docs = list(corpus.values())
queries_list = list(queries.values())

corpus_embeddings = model.encode(corpus_docs, convert_to_tensor=True, show_progress_bar=False)
query_embeddings = model.encode(queries_list, convert_to_tensor=True, show_progress_bar=False)

# Semantic search yap
all_hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=5)

# Her sorgu için detay göster
for query_id, hits in zip(queries.keys(), all_hits):
    query_text = queries[query_id]
    ground_truth_ids = relevant_docs[query_id]
    
    print(f"\n📊 Sorgu [{query_id}]: '{query_text}'")
    print(f"   ✓ Doğru Cevap(lar): {sorted(list(ground_truth_ids))}")
    print(f"   → Top-3 Sonuçlar:")
    
    for rank, hit in enumerate(hits[:3], 1):
        doc_id = f"c{hit['corpus_id']}"
        score = hit['score']
        is_correct = "✓" if doc_id in ground_truth_ids else "✗"
        doc_preview = corpus[doc_id][:80] + "..." if len(corpus[doc_id]) > 80 else corpus[doc_id]
        print(f"      {rank}. [{is_correct}] {doc_id} (skor: {score:.3f})")
        print(f"         └─ {doc_preview}")

print("\n" + "="*70)

# %% [markdown]
# ---
# 
# ## Test 2: Anlamsal Yakınlık (Semantic Similarity - STS)
# 
# **Amaç:** Modelin, kelime eşleşmesi olmasa bile, anlamsal olarak yakın cümleleri ayırt edebilmesini test etmek.
# 
# **Format:** Triplet (Üçlü) testler
# - **Anchor (Ana cümle):** Test edilecek cümle
# - **Positive (Pozitif):** Anchor'a anlamsal olarak yakın (farklı kelimelerle aynı anlam)
# - **Negative (Negatif):** Anchor'a alakasız
# 
# **Metrik:** 
# - **Triplet Accuracy:** `Sim(Anchor, Positive) > Sim(Anchor, Negative)` koşulu kaç triplet'te sağlanıyor?
# - **Margin Analizi:** Pozitif ve negatif skorlar arasındaki fark ne kadar?
# 
# **Test Kategorileri:**
# 1. Genel Türkçe (20 triplet)
# 2. Finansal Domain (20 triplet)

# %%
# =============================================================================
# TEST 2 - FİNANSAL DOMAIN TRİPLET'LER (20 adet)
# =============================================================================
# Format: (anchor, positive, negative, difficulty, category)

financial_triplets = [
    # --- ENFLASYON & FİYATLAR (Kolay-Orta) ---
    ("Enflasyon oranları yükseliyor.", 
     "Fiyat artış hızı ivme kazandı.", 
     "Şirket yeni bir CEO atadı.",
     "kolay", "enflasyon"),
    
    ("TÜFE bu ay yüzde 3 arttı.", 
     "Tüketici fiyat endeksi aylık bazda %3 yükseldi.", 
     "ÜFE verileri açıklandı.",
     "kolay", "enflasyon"),
    
    ("Merkez bankası enflasyonla mücadele ediyor.", 
     "TCMB fiyat istikrarı için çalışıyor.", 
     "Hazine tahvil ihalesine çıktı.",
     "orta", "enflasyon"),
    
    # --- FAİZ & PARA POLİTİKASI (Orta) ---
    ("Merkez bankası faiz oranlarını artırdı.", 
     "TCMB politika faizini yükseltti.", 
     "Dolar kuru düştü.",
     "kolay", "faiz"),
    
    ("Sıkı para politikası uygulanıyor.", 
     "Merkez bankası para politikasını sıkılaştırdı.", 
     "Bütçe açığı arttı.",
     "orta", "faiz"),
    
    ("Kredi faizleri yükseldi.", 
     "Borçlanma maliyetleri arttı.", 
     "Mevduat faizleri sabit kaldı.",
     "kolay", "faiz"),
    
    # --- BORSA & HİSSE (Orta-Zor) ---
    ("BIST 100 endeksi düştü.", 
     "Borsa İstanbul ana endeksi değer kaybetti.", 
     "Dolar/TL yükseldi.",
     "orta", "borsa"),
    
    ("Hisse senetleri ralli yaptı.", 
     "Pay piyasası yükselişe geçti.", 
     "Tahvil faizleri düştü.",
     "orta", "borsa"),
    
    ("X Bankası hisseleri yüzde 5 değer kazandı.", 
     "X Bankası'nın payları %5 yükseldi.", 
     "Y Şirketi temettü dağıtacak.",
     "kolay", "borsa"),
    
    ("Şirket halka arz ediliyor.", 
     "Firma borsaya açılıyor.", 
     "Şirket satın alma yapacak.",
     "kolay", "borsa"),
    
    # --- DÖVİZ & KUR (Orta-Zor) ---
    ("Dolar/TL kuru yükseldi.", 
     "Türk lirası dolar karşısında değer kaybetti.", 
     "Euro/TL düştü.",
     "orta", "döviz"),
    
    ("Döviz rezervleri azaldı.", 
     "Merkez bankasının döviz stokları düştü.", 
     "Cari açık arttı.",
     "orta", "döviz"),
    
    ("TL güçlendi.", 
     "Türk parası değer kazandı.", 
     "Dolar rekor kırdı.",
     "kolay", "döviz"),
    
    # --- KREDİ & BORÇLANMA (Orta) ---
    ("Konut kredisi çekmek istiyorum.", 
     "Ev almak için bankadan kredi kullanacağım.", 
     "Kredi kartı borcumu kapatacağım.",
     "kolay", "kredi"),
    
    ("Kredi notu yüksek olanlar daha düşük faiz alıyor.", 
     "Findeks skoru iyi olanlara uygun faiz veriliyor.", 
     "Tüketici kredisi talebi arttı.",
     "orta", "kredi"),
    
    # --- TAHVİL & BORÇLANMA (Zor) ---
    ("Hazine tahvil faizleri yükseldi.", 
     "Devlet borçlanma senetlerinin getirisi arttı.", 
     "Şirket tahvili ihraç edildi.",
     "orta", "tahvil"),
    
    ("Eurobond getirisi düştü.", 
     "Dış borçlanma senetlerinin faizi geriledi.", 
     "Hazine iç borçlanmaya yöneldi.",
     "zor", "tahvil"),
    
    # --- MAKRO EKONOMİ (Zor) ---
    ("Cari açık daraldı.", 
     "Dış ticaret açığı azaldı.", 
     "Bütçe fazlası arttı.",
     "orta", "makro"),
    
    ("Ekonomi resesyona girdi.", 
     "Ekonomik daralma başladı.", 
     "Büyüme hızlandı.",
     "orta", "makro"),
    
    ("GSYİH büyümesi beklenenden iyi geldi.", 
     "Milli gelir artışı tahminleri aştı.", 
     "İşsizlik oranı açıklandı.",
     "orta", "makro"),
]

print(f"✅ {len(financial_triplets)} Finansal Domain triplet hazırlandı")
print(f"\n📊 Zorluk dağılımı:")
difficulty_count = {}
category_count = {}
for _, _, _, diff, cat in financial_triplets:
    difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
    category_count[cat] = category_count.get(cat, 0) + 1

print("\n  Zorluk:")
for diff, count in difficulty_count.items():
    print(f"    {diff}: {count} adet")
    
print("\n  Kategori:")
for cat, count in category_count.items():
    print(f"    {cat}: {count} adet")

# %%
# =============================================================================
# TEST 2 - GENEL TÜRKÇE TRİPLET'LER (20 adet)
# =============================================================================
# Format: (anchor, positive, negative, difficulty, category)

general_turkish_triplets = [
    # --- COĞRAFYA & GENEL BİLGİ (Kolay) ---
    ("Türkiye'nin başkenti neresidir?", 
     "Ankara hangi ülkenin başkentidir?", 
     "İstanbul'da hava durumu nasıl?",
     "kolay", "coğrafya"),
    
    ("İstanbul Türkiye'nin en kalabalık şehridir.", 
     "Türkiye'de nüfusu en fazla olan il İstanbul'dur.", 
     "Ankara 1923 yılında başkent oldu.",
     "kolay", "coğrafya"),
    
    # --- GÜNLÜK HAYAT (Kolay-Orta) ---
    ("Bugün hava çok soğuk.", 
     "Dışarıda hava oldukça serin.", 
     "Yarın yağmur yağacak.",
     "kolay", "hava_durumu"),
    
    ("Akşam yemeğinde pizza yedik.", 
     "Akşamleyin pizza siparişi verdik.", 
     "Sabah kahvaltıda börek vardı.",
     "kolay", "yemek"),
    
    ("Arabam bozuldu, tamirciye götürmem lazım.", 
     "Aracımı servise götürmeliyim, arızalı.", 
     "Yeni bir araba almayı düşünüyorum.",
     "orta", "günlük"),
    
    # --- SPOR (Orta) ---
    ("Galatasaray maçı kazandı.", 
     "Galatasaray galibiyetle ayrıldı.", 
     "Fenerbahçe transfere başladı.",
     "kolay", "spor"),
    
    ("Cristiano Ronaldo gol attı.", 
     "Ronaldo ağları havalandırdı.", 
     "Messi en iyi futbolcudur.",
     "orta", "spor"),
    
    # --- SAĞLIK (Orta) ---
    ("Başım çok ağrıyor, ağrı kesici almalıyım.", 
     "Baş ağrım var, ilaç içeceğim.", 
     "Doktora gitmem gerekiyor.",
     "kolay", "sağlık"),
    
    ("Düzenli egzersiz yapmak sağlıklı yaşam için önemlidir.", 
     "Sağlıklı olmak için spor yapmak gerekir.", 
     "Beslenmede protein önemlidir.",
     "orta", "sağlık"),
    
    # --- EĞİTİM (Orta) ---
    ("Üniversite sınavına hazırlanıyorum.", 
     "YKS için ders çalışıyorum.", 
     "Liseyi geçen sene bitirdim.",
     "orta", "eğitim"),
    
    ("Bu kitabı okumak çok zevkli.", 
     "Bu eseri okumak keyifli.", 
     "Kitap almayı seviyorum.",
     "kolay", "eğitim"),
    
    # --- TEKNOLOJİ (Orta-Zor) ---
    ("Telefonum şarjı bitti.", 
     "Cep telefonumun pili tükendi.", 
     "Yeni telefon almam gerekiyor.",
     "kolay", "teknoloji"),
    
    ("Yapay zeka teknolojisi hızla gelişiyor.", 
     "AI alanında büyük ilerlemeler kaydediliyor.", 
     "Robotlar gelecekte yaygınlaşacak.",
     "orta", "teknoloji"),
    
    ("İnternet bağlantım çok yavaş.", 
     "Wifi hızım düşük.", 
     "Bilgisayarım çok eski.",
     "kolay", "teknoloji"),
    
    # --- DUYGUSAL İFADELER (Orta-Zor) ---
    ("Çok mutluyum bugün.", 
     "Bugün kendimi çok iyi hissediyorum.", 
     "Dün üzgündüm.",
     "kolay", "duygu"),
    
    ("Bu film beni çok duygulandırdı.", 
     "Film izlerken gözyaşlarımı tutamadım.", 
     "Film çok uzundu.",
     "orta", "duygu"),
    
    # --- ZAMAN & TARİH (Orta) ---
    ("Toplantı yarın saat 10'da.", 
     "Meeting yarın sabah onda başlıyor.", 
     "Dün toplantı iptal edildi.",
     "kolay", "zaman"),
    
    ("Cumhuriyet 1923 yılında ilan edildi.", 
     "Türkiye Cumhuriyeti 1923'te kuruldu.", 
     "Atatürk 1881 yılında doğdu.",
     "orta", "tarih"),
    
    # --- KARMAŞIK ANLAMLAR (Zor) ---
    ("Bu sorunu çözmek için daha fazla zamana ihtiyacım var.", 
     "Bu problemi halletmek için ek süre gerekiyor.", 
     "Bu sorunun cevabını biliyorum.",
     "orta", "karmaşık"),
    
    ("Projeyi zamanında teslim edemeyeceğim.", 
     "Çalışmayı son tarihe yetiştiremem.", 
     "Proje çok başarılı oldu.",
     "orta", "karmaşık"),
]

print(f"✅ {len(general_turkish_triplets)} Genel Türkçe triplet hazırlandı")
print(f"\n📊 Zorluk dağılımı:")
difficulty_count = {}
for _, _, _, diff, _ in general_turkish_triplets:
    difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
for diff, count in difficulty_count.items():
    print(f"  {diff}: {count} adet")

# %%
print("\n" + "="*70)
print("--- TEST 2: Anlamsal Yakınlık (Semantic Similarity) ---")
print("="*70)

# Tüm triplet'leri birleştir
all_triplets = general_turkish_triplets + financial_triplets

# Her triplet için embedding hesapla ve değerlendir
correct_triplets = 0
total_triplets = len(all_triplets)

# Kategori bazlı istatistikler
category_stats = {
    'genel_turkce': {'correct': 0, 'total': 0},
    'finansal': {'correct': 0, 'total': 0}
}

# Zorluk bazlı istatistikler
difficulty_stats = {}

# Margin (skor farkı) bilgileri
margins = []
positive_scores = []
negative_scores = []

# Hata analizi için
failed_triplets = []

print(f"\nToplam {total_triplets} triplet test ediliyor...")
print("(Genel Türkçe: {}, Finansal: {})".format(
    len(general_turkish_triplets), len(financial_triplets)))

# Her triplet'i değerlendir
for idx, triplet in enumerate(all_triplets):
    anchor, positive, negative, difficulty, category = triplet
    
    # Embedding hesapla
    anchor_emb = model.encode(anchor, convert_to_tensor=True)
    positive_emb = model.encode(positive, convert_to_tensor=True)
    negative_emb = model.encode(negative, convert_to_tensor=True)
    
    # Benzerlik skorları hesapla
    sim_anchor_positive = util.cos_sim(anchor_emb, positive_emb).item()
    sim_anchor_negative = util.cos_sim(anchor_emb, negative_emb).item()
    
    # Margin (skor farkı)
    margin = sim_anchor_positive - sim_anchor_negative
    margins.append(margin)
    positive_scores.append(sim_anchor_positive)
    negative_scores.append(sim_anchor_negative)
    
    # Doğru mu kontrol et
    is_correct = sim_anchor_positive > sim_anchor_negative
    
    if is_correct:
        correct_triplets += 1
    else:
        # Hata analizi için kaydet
        failed_triplets.append({
            'idx': idx,
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'sim_pos': sim_anchor_positive,
            'sim_neg': sim_anchor_negative,
            'margin': margin,
            'difficulty': difficulty,
            'category': category
        })
    
    # Kategori istatistikleri
    if idx < len(general_turkish_triplets):
        category_stats['genel_turkce']['total'] += 1
        if is_correct:
            category_stats['genel_turkce']['correct'] += 1
    else:
        category_stats['finansal']['total'] += 1
        if is_correct:
            category_stats['finansal']['correct'] += 1
    
    # Zorluk istatistikleri
    if difficulty not in difficulty_stats:
        difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
    difficulty_stats[difficulty]['total'] += 1
    if is_correct:
        difficulty_stats[difficulty]['correct'] += 1

# Accuracy hesapla
accuracy = (correct_triplets / total_triplets) * 100

# Sonuçları yazdır
print("\n" + "="*70)
print("--- TEST 2 SONUÇLARI ---")
print("="*70)

print(f"\n🎯 GENEL SONUÇLAR:")
print(f"  Toplam Triplet:        {total_triplets}")
print(f"  ✓ Başarılı:            {correct_triplets}")
print(f"  ✗ Başarısız:           {total_triplets - correct_triplets}")
print(f"  ")
print(f"  📊 Triplet Accuracy:   {accuracy:.2f}%")
print(f"  Ortalama Margin:       {np.mean(margins):+.3f}")
print(f"  Minimum Margin:        {np.min(margins):+.3f}")
print(f"  Maximum Margin:        {np.max(margins):+.3f}")
print(f"  Margin > 0.1 oranı:    {(np.array(margins) > 0.1).mean() * 100:.1f}%")

print(f"\n📊 SKOR DAĞILIMI:")
print(f"  Avg Sim(Anchor, Positive): {np.mean(positive_scores):.3f}")
print(f"  Avg Sim(Anchor, Negative): {np.mean(negative_scores):.3f}")
print(f"  Skor Ayrışması:            {np.mean(positive_scores) - np.mean(negative_scores):.3f}")

print(f"\n📊 KATEGORİ BAZLI ACCURACY:")
for cat_name, stats in category_stats.items():
    cat_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"  {cat_name:20s}: {cat_acc:5.1f}% ({stats['correct']}/{stats['total']})")

print(f"\n📊 ZORLUK SEVİYESİ BAZLI ACCURACY:")
for diff_name in sorted(difficulty_stats.keys()):
    stats = difficulty_stats[diff_name]
    diff_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"  {diff_name:20s}: {diff_acc:5.1f}% ({stats['correct']}/{stats['total']})")

# Hata analizi
if failed_triplets:
    print(f"\n🔍 HATA ANALİZİ (Başarısız Triplet'ler):")
    print("="*70)
    for fail in failed_triplets[:5]:  # İlk 5 hatayı göster
        print(f"\n  ✗ Triplet #{fail['idx']} (Margin: {fail['margin']:+.3f}, {fail['difficulty']}, {fail['category']})")
        print(f"      Anchor:   \"{fail['anchor']}\"")
        print(f"      Positive: \"{fail['positive']}\" (skor: {fail['sim_pos']:.3f})")
        print(f"      Negative: \"{fail['negative']}\" (skor: {fail['sim_neg']:.3f}) ← YANLIŞLIKLA DAHA YÜKSEK!")
    
    if len(failed_triplets) > 5:
        print(f"\n  ... ve {len(failed_triplets) - 5} hata daha.")
else:
    print(f"\n✅ MÜKEMMEL! Tüm triplet'ler doğru değerlendirildi!")

print("\n" + "="*70)
print("✅ Test 2 tamamlandı!")
print("="*70)

# %%


# %%


# %% [markdown]
# ## Test 3: Dilsel Sağlamlık ve Anlamsal Nüans (Linguistic Robustness & Nuance)
# 
# Bu, sizin "türkçe hassasiyet ve olumlusuz olumsuz" maddenizi detaylandırır.
# 
# * **Amaç:** Modelin, Türkçe'nin morfolojik zenginliği, olası yazım hataları ve anlamsal zıtlıklar (olumsuzluk) karşısında ne kadar sağlam durduğunu ölçmek.
# * **Test Verisi (Karşılaştırma Çiftleri):**
#   * **a) Morfoloji (Ekler):**
#     * Çift 1: ("mevduat", "mevduatım")
#     * Çift 2: ("hisse senedi", "hisse senetlerindeki")
#   * **b) Yazım Hataları (Typos):**
#     * Çift 1: ("enflasyon", "enfilasyon")
#     * Çift 2: ("kredi faizi", "kredi faizi")
#   * **c) Olumsuzluk (Negation):**
#     * Çift 1: ("Kredi çekmek mantıklı bir karar.", "Kredi çekmek mantıklı bir karar değil.")
#     * Çift 2: ("Borsanın düşüşü bekleniyor.", "Borsanın yükselişi bekleniyor.")
# * **Nasıl Ölçülür? (Metrik):**
#   * **Kosinüs Benzerliği (Cosine Similarity) Analizi:**
#     * **a) ve b) için:** Benzerlik skoru *çok yüksek* olmalı (örn: > 0.90). Anlam değişmemeli.
#     * **c) için:** Benzerlik skoru *düşük* veya hatta *negatif* olmalı. Anlam tamamen zıt. (Örn: `Sim(Cümle1, Cümle2)` < 0.3)
# 

# %%
# =============================================================================
# A) MORFOLOJİ TESTLERİ (20 çift)
# =============================================================================
# Format: (original, with_suffix, category)

morphology_pairs = [
    # --- İYELİK EKLERİ (5 çift) ---
    ("mevduat", "mevduatım", "iyelik"),
    ("kredi", "kredim", "iyelik"),
    ("hesap", "hesabım", "iyelik"),
    ("yatırım", "yatırımım", "iyelik"),
    ("birikim", "birikimim", "iyelik"),

    # --- ÇOĞUL EKLERİ (5 çift) ---
    ("hisse", "hisseler", "çoğul"),
    ("tahvil", "tahviller", "çoğul"),
    ("dolar", "dolarlar", "çoğul"),
    ("yatırımcı", "yatırımcılar", "çoğul"),
    ("şirket", "şirketler", "çoğul"),

    # --- HAL EKLERİ (5 çift) ---
    ("banka", "bankada", "hal_eki"),
    ("borsa", "borsada", "hal_eki"),
    ("faiz", "faizden", "hal_eki"),
    ("piyasa", "piyasaya", "hal_eki"),
    ("kur", "kurda", "hal_eki"),

    # --- İLGİ EKLERİ (3 çift) ---
    ("enflasyon", "enflasyonun", "ilgi"),
    ("büyüme", "büyümenin", "ilgi"),
    ("gelir", "gelirin", "ilgi"),

    # --- KARMAŞIK EK KOMBİNASYONLARI (2 çift) ---
    ("hisse senedi", "hisse senetlerindeki", "karmaşık"),
    ("merkez bankası", "merkez bankasının", "karmaşık"),
]

print(f"✅ {len(morphology_pairs)} Morfoloji çifti hazırlandı")

# =============================================================================
# B) YAZIM HATALARI TESTLERİ (15 çift)
# =============================================================================
# Format: (correct, typo, typo_type)

typo_pairs = [
    # --- TÜRKÇE'YE ÖZGÜ HATALAR (5 çift) ---
    ("enflasyon", "enfilasyon", "türkçe_ses"),
    ("mevduat", "mevdüat", "türkçe_ses"),
    ("ekonomi", "ekenomi", "türkçe_ses"),
    ("yatırım", "yatirim", "türkçe_ses"),
    ("gelişme", "gelişeme", "türkçe_ses"),

    # --- KLAVYE HATALARI (5 çift) ---
    ("kredi", "kreid", "klavye"),
    ("faiz", "faiiz", "klavye"),
    ("borsa", "boras", "klavye"),
    ("tahvil", "tahivl", "klavye"),
    ("döviz", "döviz", "klavye"),

    # --- EKSİK/FAZLA HARF (5 çift) ---
    ("büyüme", "byüme", "eksik_harf"),
    ("gelir", "geliir", "fazla_harf"),
    ("risk", "rsk", "eksik_harf"),
    ("kâr", "kar", "eksik_işaret"),
    ("portföy", "portfoy", "eksik_işaret"),
]

print(f"✅ {len(typo_pairs)} Yazım hatası çifti hazırlandı")

# =============================================================================
# C) OLUMSUZLUK & ZITLIK TESTLERİ (20 çift)
# =============================================================================
# Format: (sentence1, sentence2, negation_type)

negation_pairs = [
    # --- OLUMSUZLUK EKLERİ (7 çift) ---
    ("Kredi çekmek mantıklı bir karar.",
     "Kredi çekmek mantıklı bir karar değil.",
     "olumsuzluk_eki"),

    ("Borsa yükseldi.",
     "Borsa yükselmedi.",
     "olumsuzluk_eki"),

    ("Faiz oranları düştü.",
     "Faiz oranları düşmedi.",
     "olumsuzluk_eki"),

    ("Bu yatırım karlı.",
     "Bu yatırım karlı değil.",
     "olumsuzluk_eki"),

    ("Enflasyon kontrol altında.",
     "Enflasyon kontrol altında değil.",
     "olumsuzluk_eki"),

    ("Ekonomi büyüyor.",
     "Ekonomi büyümüyor.",
     "olumsuzluk_eki"),

    ("Piyasa istikrarlı.",
     "Piyasa istikrarlı değil.",
     "olumsuzluk_eki"),

    # --- ZIT KELİMELER - FİNANSAL (7 çift) ---
    ("Enflasyon yükseldi.",
     "Enflasyon düştü.",
     "zıt_kelime"),

    ("Dolar güçlendi.",
     "Dolar zayıfladı.",
     "zıt_kelime"),

    ("Borsa kazandırdı.",
     "Borsa kaybettirdi.",
     "zıt_kelime"),

    ("Faiz arttı.",
     "Faiz azaldı.",
     "zıt_kelime"),

    ("Büyüme hızlandı.",
     "Büyüme yavaşladı.",
     "zıt_kelime"),

    ("Gelir arttı.",
     "Gelir düştü.",
     "zıt_kelime"),

    ("Cari açık daraldı.",
     "Cari açık genişledi.",
     "zıt_kelime"),

    # --- BAĞLAÇ DEĞİŞİMİ (6 çift) ---
    ("Enflasyon arttı ama büyüme devam etti.",
     "Enflasyon arttı ve büyüme durdu.",
     "bağlaç"),

    ("Faiz düştü ancak kredi talebi arttı.",
     "Faiz düştü ve kredi talebi azaldı.",
     "bağlaç"),

    ("Borsa yükseldi çünkü veriler iyiydi.",
     "Borsa düştü çünkü veriler kötüydü.",
     "bağlaç"),

    ("Dolar arttı fakat ihracat olumlu etkilendi.",
     "Dolar arttı ve ihracat olumsuz etkilendi.",
     "bağlaç"),

    ("Merkez bankası faiz artırdı yine de enflasyon yükseldi.",
     "Merkez bankası faiz artırdı bu yüzden enflasyon düştü.",
     "bağlaç"),

    ("Ekonomi büyüdü ama işsizlik arttı.",
     "Ekonomi büyüdü ve işsizlik azaldı.",
     "bağlaç"),
]

print(f"✅ {len(negation_pairs)} Olumsuzluk/Zıtlık çifti hazırlandı")


# %%
print("\n" + "="*70)
print("--- TEST 3: Dilsel Sağlamlık ve Anlamsal Nüans ---")
print("="*70)

results = {
    'morphology': {},
    'typo': {},
    'negation': {}
}

# =========================================================================
# A) MORFOLOJİ TESTİ
# =========================================================================
print("\n📊 A) MORFOLOJİ TESTİ (Ekler)")
print("-" * 70)

morph_scores = []
morph_categories = {}
problem_pairs_morph = []

for original, with_suffix, category in morphology_pairs:
    # Embedding'leri al
    emb1 = model.encode(original, convert_to_tensor=True)
    emb2 = model.encode(with_suffix, convert_to_tensor=True)

    # Benzerlik hesapla
    similarity = util.cos_sim(emb1, emb2).item()
    morph_scores.append(similarity)

    # Kategori istatistikleri
    if category not in morph_categories:
        morph_categories[category] = []
    morph_categories[category].append(similarity)

    # Problem çiftleri (skor < 0.85)
    if similarity < 0.85:
        problem_pairs_morph.append({
            'pair': (original, with_suffix),
            'score': similarity,
            'category': category
        })

# Morfoloji sonuçları
avg_morph = np.mean(morph_scores)
success_rate_morph = (np.array(morph_scores) > 0.85).mean() * 100

results['morphology'] = {
    'avg_similarity': avg_morph,
    'min_similarity': np.min(morph_scores),
    'max_similarity': np.max(morph_scores),
    'std_similarity': np.std(morph_scores),
    'success_rate': success_rate_morph,
    'total_pairs': len(morphology_pairs),
    'problem_pairs': problem_pairs_morph,
    'category_scores': {cat: np.mean(scores) for cat, scores in morph_categories.items()}
}

print(f"\n  Toplam Çift:           {len(morphology_pairs)}")
print(f"  Ortalama Benzerlik:    {avg_morph:.3f}")
print(f"  Min/Max:               {np.min(morph_scores):.3f} / {np.max(morph_scores):.3f}")
print(f"  Std Dev:               {np.std(morph_scores):.3f}")
print(f"  Başarı Oranı (>0.85):  {success_rate_morph:.1f}%")

print(f"\n  📊 Kategori Bazlı:")
for cat, scores in morph_categories.items():
    print(f"    {cat:20s}: {np.mean(scores):.3f}")

if problem_pairs_morph:
    print(f"\n  ⚠️ Problem Çiftler (skor < 0.85):")
    for prob in problem_pairs_morph[:3]:
        print(f"    '{prob['pair'][0]}' ↔ '{prob['pair'][1]}' ({prob['category']}): {prob['score']:.3f}")

# =========================================================================
# B) YAZIM HATALARI TESTİ
# =========================================================================
print("\n" + "="*70)
print("📊 B) YAZIM HATALARI TESTİ (Typos)")
print("-" * 70)

typo_scores = []
typo_categories = {}
problem_pairs_typo = []

for correct, typo, typo_type in typo_pairs:
    # Embedding'leri al
    emb1 = model.encode(correct, convert_to_tensor=True)
    emb2 = model.encode(typo, convert_to_tensor=True)

    # Benzerlik hesapla
    similarity = util.cos_sim(emb1, emb2).item()
    typo_scores.append(similarity)

    # Kategori istatistikleri
    if typo_type not in typo_categories:
        typo_categories[typo_type] = []
    typo_categories[typo_type].append(similarity)

    # Problem çiftleri (skor < 0.75)
    if similarity < 0.75:
        problem_pairs_typo.append({
            'pair': (correct, typo),
            'score': similarity,
            'type': typo_type
        })

# Typo sonuçları
avg_typo = np.mean(typo_scores)
success_rate_typo = (np.array(typo_scores) > 0.75).mean() * 100

results['typo'] = {
    'avg_similarity': avg_typo,
    'min_similarity': np.min(typo_scores),
    'max_similarity': np.max(typo_scores),
    'std_similarity': np.std(typo_scores),
    'success_rate': success_rate_typo,
    'total_pairs': len(typo_pairs),
    'problem_pairs': problem_pairs_typo,
    'category_scores': {cat: np.mean(scores) for cat, scores in typo_categories.items()}
}

print(f"\n  Toplam Çift:           {len(typo_pairs)}")
print(f"  Ortalama Benzerlik:    {avg_typo:.3f}")
print(f"  Min/Max:               {np.min(typo_scores):.3f} / {np.max(typo_scores):.3f}")
print(f"  Std Dev:               {np.std(typo_scores):.3f}")
print(f"  Başarı Oranı (>0.75):  {success_rate_typo:.1f}%")

print(f"\n  📊 Hata Tipi Bazlı:")
for typ, scores in typo_categories.items():
    print(f"    {typ:20s}: {np.mean(scores):.3f}")

if problem_pairs_typo:
    print(f"\n  ⚠️ Problem Çiftler (skor < 0.75):")
    for prob in problem_pairs_typo[:3]:
        print(f"    '{prob['pair'][0]}' ↔ '{prob['pair'][1]}' ({prob['type']}): {prob['score']:.3f}")

# =========================================================================
# C) OLUMSUZLUK & ZITLIK TESTİ
# =========================================================================
print("\n" + "="*70)
print("📊 C) OLUMSUZLUK & ZITLIK TESTİ (Negation & Antonyms)")
print("-" * 70)

neg_scores = []
neg_categories = {}
problem_pairs_neg = []

for sent1, sent2, neg_type in negation_pairs:
    # Embedding'leri al
    emb1 = model.encode(sent1, convert_to_tensor=True)
    emb2 = model.encode(sent2, convert_to_tensor=True)

    # Benzerlik hesapla
    similarity = util.cos_sim(emb1, emb2).item()
    neg_scores.append(similarity)

    # Kategori istatistikleri
    if neg_type not in neg_categories:
        neg_categories[neg_type] = []
    neg_categories[neg_type].append(similarity)

    # Problem çiftleri (skor > 0.50 - zıt anlamları ayırt edememiş!)
    if similarity > 0.50:
        problem_pairs_neg.append({
            'pair': (sent1, sent2),
            'score': similarity,
            'type': neg_type
        })

# Negation sonuçları
avg_neg = np.mean(neg_scores)
success_rate_neg = (np.array(neg_scores) < 0.50).mean() * 100

results['negation'] = {
    'avg_similarity': avg_neg,
    'min_similarity': np.min(neg_scores),
    'max_similarity': np.max(neg_scores),
    'std_similarity': np.std(neg_scores),
    'success_rate': success_rate_neg,
    'total_pairs': len(negation_pairs),
    'problem_pairs': problem_pairs_neg,
    'category_scores': {cat: np.mean(scores) for cat, scores in neg_categories.items()}
}

print(f"\n  Toplam Çift:           {len(negation_pairs)}")
print(f"  Ortalama Benzerlik:    {avg_neg:.3f} (düşük olmalı!)")
print(f"  Min/Max:               {np.min(neg_scores):.3f} / {np.max(neg_scores):.3f}")
print(f"  Std Dev:               {np.std(neg_scores):.3f}")
print(f"  Başarı Oranı (<0.50):  {success_rate_neg:.1f}%")

print(f"\n  📊 Kategori Bazlı:")
for typ, scores in neg_categories.items():
    print(f"    {typ:20s}: {np.mean(scores):.3f}")

if problem_pairs_neg:
    print(f"\n  ⚠️ TEHLİKELİ Çiftler (skor > 0.50 - Zıt anlamları ayırt edemiyor!):")
    for prob in problem_pairs_neg[:5]:
        print(f"    Skor: {prob['score']:.3f} ({prob['type']})")
        print(f"      → '{prob['pair'][0]}'")
        print(f"      → '{prob['pair'][1]}'")

# =========================================================================
# GENEL ÖZET
# =========================================================================
print("\n" + "="*70)
print("--- TEST 3 ÖZET ---")
print("="*70)

print(f"\n📊 A) Morfoloji:    Avg={avg_morph:.3f}, Başarı={success_rate_morph:.1f}%")
if success_rate_morph >= 80:
    print(f"   ✅ Mükemmel! Model Türkçe ekleri iyi anlıyor.")
elif success_rate_morph >= 60:
    print(f"   ⚠️ Orta. Model bazı eklerde zorlanıyor.")
else:
    print(f"   ❌ Zayıf! Model Türkçe morfolojiyi öğrenememiş.")

print(f"\n📊 B) Yazım Hataları: Avg={avg_typo:.3f}, Başarı={success_rate_typo:.1f}%")
if success_rate_typo >= 70:
    print(f"   ✅ İyi! Model yazım hatalarına dayanıklı.")
elif success_rate_typo >= 50:
    print(f"   ⚠️ Orta. Bazı typo'ları anlamıyor.")
else:
    print(f"   ❌ Zayıf! Yazım hatalarına çok hassas.")

print(f"\n📊 C) Olumsuzluk & Zıtlık: Avg={avg_neg:.3f}, Başarı={success_rate_neg:.1f}%")
if success_rate_neg >= 70:
    print(f"   ✅ Mükemmel! Zıt anlamları iyi ayırt ediyor.")
elif success_rate_neg >= 50:
    print(f"   ⚠️ Orta. Bazı zıtlıkları yakalayamıyor.")
else:
    print(f"   ❌ TEHLİKELİ! Zıt anlamları ayırt edemiyor!")
    print(f"      Production'da 'düştü' ile 'yükseldi' aynı çıkabilir!")

print("\n" + "="*70)
print("✅ Test 3 tamamlandı!")
print("="*70)



# %%


# %%


# %%


# %% [markdown]
# ---
# 
# ## Test 4: Vektör Aritmetiği / Anlamsal Uzay (Analogies)
# 
# **Amaç:** Embedding uzayının anlamsal ilişkileri (analojileri) ne kadar iyi öğrendiğini test etmek.
# 
# **Format:** Analoji dörtlüleri `(A, B, C, D)` → `A - B + C ≈ D`
# - Örnek: `Kral - Erkek + Kadın ≈ Kraliçe`
# - Vektör: `embedding(A) - embedding(B) + embedding(C)` → En yakın kelime D olmalı
# 
# **Metrik:** 
# - **Top-k Accuracy:** Hesaplanan vektöre en yakın k kelime içinde D var mı?
# - **Ranking:** D ortalama kaçıncı sırada?
# 
# **Test Kategorileri:**
# 1. Genel Türkçe Analojiler (20 dörtlü)
#    - Coğrafya (Başkent-Ülke)
#    - Zaman (Fiil Çekimi)
#    - Cinsiyet
#    - Büyüklük/Derece
#    
# 2. Finansal Domain Analojiler (20 dörtlü)
#    - Para Birimi - Ülke
#    - Borsa - Ülke
#    - Kurum - Pozisyon
#    - Enstrüman - Piyasa

# %%
# =============================================================================
# TEST 4 - GENEL TÜRKÇE ANALOJİ DÖRTLÜLERI (20 adet)
# =============================================================================
# Format: (A, B, C, D, category)
# Mantık: A - B + C ≈ D
# baba - erkek + kadın ≈ anne
# oğul - erkek + kadın ≈ kız


general_analogies = [
    # --- CİNSİYET ---
    ("baba", "erkek", "kadın", "anne", "cinsiyet"),
    ("oğul", "erkek", "kadın", "kız", "cinsiyet"),
    ("kral", "erkek", "kadın", "kraliçe", "cinsiyet"),
    ("prens", "erkek", "kadın", "prenses", "cinsiyet"),
    ("amca", "erkek", "kadın", "hala", "cinsiyet"),

    # --- COĞRAFYA: Başkent - Ülke ---
    ("Ankara", "Türkiye", "Almanya", "Berlin", "coğrafya"),
    ("Paris", "Fransa", "İngiltere", "Londra", "coğrafya"),
    ("Roma", "İtalya", "İspanya", "Madrid", "coğrafya"),
    ("Tokyo", "Japonya", "Çin", "Pekin", "coğrafya"),
    ("Atina", "Yunanistan", "Mısır", "Kahire", "coğrafya"),

    # --- ZAMAN: Fiil Çekimi (Mastar - Geçmiş) ---
    ("gelmek", "geldi", "gitti", "gitmek", "fiil_zaman"),
    ("okumak", "okudu", "yazdı", "yazmak", "fiil_zaman"),
    ("almak", "aldı", "verdi", "vermek", "fiil_zaman"),
    ("yemek", "yedi", "içti", "içmek", "fiil_zaman"),
    ("düşünmek", "düşündü", "anladı", "anlamak", "fiil_zaman"),

    # --- BÜYÜKLÜK / DERECE ---
    ("küçük", "büyük", "çok", "az", "derece"),
    ("soğuk", "sıcak", "hızlı", "yavaş", "derece"),
    ("eski", "yeni", "taze", "köhne", "derece"),

    # --- HAYVAN - YAVRU ---
    ("kedi", "kedi yavrusu", "köpek yavrusu", "köpek", "hayvan"),
    ("inek", "buzağı", "tay", "at", "hayvan"),
]

print(f"✅ {len(general_analogies)} Genel Türkçe analoji")

# =============================================================================
# FİNANSAL DOMAIN ANALOJİLER (DÜZELTİLMİŞ) - 20 adet
# =============================================================================

financial_analogies = [
    # --- PARA BİRİMİ - ÜLKE/BÖLGE ---
    ("dolar", "Amerika", "Avrupa", "euro", "para_ülke"),
    ("dolar", "ABD", "Japonya", "yen", "para_ülke"),
    ("sterlin", "İngiltere", "İsviçre", "frank", "para_ülke"),
    ("TL", "Türkiye", "Rusya", "ruble", "para_ülke"),
    ("yuan", "Çin", "Kore", "won", "para_ülke"),

    # --- BORSA ENDEKSİ - ÜLKE ---
    ("BIST 100", "Türkiye", "Amerika", "NASDAQ", "borsa_ülke"),
    ("BIST 100", "İstanbul", "Almanya", "DAX", "borsa_ülke"),
    ("Nikkei", "Japonya", "İngiltere", "FTSE", "borsa_ülke"),
    ("S&P 500", "Amerika", "Fransa", "CAC 40", "borsa_ülke"),

    # --- FİNANSAL KAVRAMLAR - ZITLIK ---
    ("yükseliş", "düşüş", "kayıp", "kazanç", "zıtlık"),
    ("alıcı", "satıcı", "arz", "talep", "zıtlık"),
    ("enflasyon", "deflasyon", "küçülme", "büyüme", "zıtlık"),
    ("boğa", "ayı", "düşüş", "yükseliş", "zıtlık"),

    # --- ENSTRÜMAN - PİYASA ---
    ("hisse", "borsa", "tahvil piyasası", "tahvil", "enstrüman_piyasa"),
    ("döviz", "forex", "emtia", "altın", "enstrüman_piyasa"),
    ("vadeli", "VİOP", "spot piyasa", "spot", "enstrüman_piyasa"),

    # --- FİNANSAL ORAN - BİLEŞEN ---
    ("F/K", "fiyat", "piyasa değeri", "PD/DD", "oran"),
    ("ROE", "öz sermaye", "aktif", "ROA", "oran"),

    # --- KURUM TİPİ - POZİSYON ---
    ("banka", "banka müdürü", "şirket müdürü", "şirket", "kurum_pozisyon"),
]



print(f"✅ {len(financial_analogies)} Finansal Domain analoji dörtlüsü hazırlandı")
print(f"\n📊 Kategori dağılımı:")
category_count = {}
for _, _, _, _, cat in financial_analogies:
    category_count[cat] = category_count.get(cat, 0) + 1
for cat, count in sorted(category_count.items()):
    print(f"  {cat}: {count} adet")
    
print(f"\n📊 Toplam Analoji: {len(general_analogies) + len(financial_analogies)}")
print(f"  - Genel Türkçe: {len(general_analogies)}")
print(f"  - Finansal Domain: {len(financial_analogies)}")

# %%
# =============================================================================
# TEST 4 - FİNANSAL DOMAIN ANALOJİ DÖRTLÜLERI (20 adet)
# =============================================================================
# Format: (A, B, C, D, category)
# Mantık: A - B + C ≈ D



# %%
# =============================================================================
# TEST 4 - VOCABULARY OLUŞTURMA
# =============================================================================
# Tüm analoji kelimelerini vocabulary olarak kullan

all_analogies = general_analogies + financial_analogies

# Vocabulary: Tüm benzersiz kelimeleri topla
vocabulary = set()
for a, b, c, d, _ in all_analogies:
    vocabulary.update([a, b, c, d])

# Listeye çevir ve sırala
vocabulary = sorted(list(vocabulary))

print(f"✅ Vocabulary oluşturuldu: {len(vocabulary)} benzersiz kelime")
print(f"\nÖrnek kelimeler: {vocabulary[:10]}")

# Vocabulary'yi encode et (bir kez yapılır, hızlandırır)
print("\nVocabulary encode ediliyor...")
vocab_embeddings = model.encode(vocabulary, convert_to_tensor=True, show_progress_bar=True)
print(f"✅ Vocabulary embeddings hazır: {vocab_embeddings.shape}")

# %%
print("\n" + "="*70)
print("--- TEST 4: Vektör Aritmetiği / Anlamsal Uzay (Analogies) ---")
print("="*70)

# İstatistikler
top1_correct = 0
top5_correct = 0
top10_correct = 0
total_analogies = len(all_analogies)

# Kategori bazlı istatistikler
category_stats = {
    'genel_turkce': {'top1': 0, 'top5': 0, 'top10': 0, 'total': 0},
    'finansal': {'top1': 0, 'top5': 0, 'top10': 0, 'total': 0}
}

# Alt kategori istatistikleri
subcategory_stats = {}

# Ranking bilgileri
ranks = []

# Hata analizi
failed_top5 = []
successful_examples = []

print(f"\nToplam {total_analogies} analoji test ediliyor...")
print("(Genel Türkçe: {}, Finansal: {})".format(
    len(general_analogies), len(financial_analogies)))

# Her analoji dörtlüsünü test et
for idx, analogy in enumerate(all_analogies):
    a, b, c, d, category = analogy
    
    # Embedding'leri al
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    emb_c = model.encode(c, convert_to_tensor=True)
    
    # Vektör aritmetiği: A - B + C
    result_vector = emb_a - emb_b + emb_c
    
    # Vocabulary'deki tüm kelimelerle benzerlik hesapla
    similarities = util.cos_sim(result_vector, vocab_embeddings)[0]
    
    # En yakın k kelimeyi bul
    top_k_indices = torch.topk(similarities, k=10).indices.cpu().numpy()
    top_k_words = [vocabulary[i] for i in top_k_indices]
    top_k_scores = [similarities[i].item() for i in top_k_indices]
    
    # D'nin rank'ini bul
    if d in top_k_words:
        rank = top_k_words.index(d) + 1
    else:
        rank = 999  # Bulunamadı
    
    ranks.append(rank)
    
    # Top-1, Top-5, Top-10 kontrolü
    in_top1 = (rank == 1)
    in_top5 = (rank <= 5)
    in_top10 = (rank <= 10)
    
    if in_top1:
        top1_correct += 1
    if in_top5:
        top5_correct += 1
    if in_top10:
        top10_correct += 1
    
    # Kategori istatistikleri
    if idx < len(general_analogies):
        cat_key = 'genel_turkce'
    else:
        cat_key = 'finansal'
    
    category_stats[cat_key]['total'] += 1
    if in_top1:
        category_stats[cat_key]['top1'] += 1
    if in_top5:
        category_stats[cat_key]['top5'] += 1
    if in_top10:
        category_stats[cat_key]['top10'] += 1
    
    # Alt kategori istatistikleri
    if category not in subcategory_stats:
        subcategory_stats[category] = {'top5': 0, 'total': 0}
    subcategory_stats[category]['total'] += 1
    if in_top5:
        subcategory_stats[category]['top5'] += 1
    
    # Örnek kaydet
    if in_top5 and len(successful_examples) < 3:
        successful_examples.append({
            'analogy': f'"{a}" - "{b}" + "{c}" → "{d}"',
            'rank': rank,
            'top5': top_k_words[:5],
            'scores': top_k_scores[:5]
        })
    
    if not in_top5 and len(failed_top5) < 5:
        failed_top5.append({
            'analogy': f'"{a}" - "{b}" + "{c}" → "{d}"',
            'rank': rank if rank < 999 else "Bulunamadı",
            'top5': top_k_words[:5],
            'scores': top_k_scores[:5],
            'category': category
        })

# Accuracy hesapla
top1_acc = (top1_correct / total_analogies) * 100
top5_acc = (top5_correct / total_analogies) * 100
top10_acc = (top10_correct / total_analogies) * 100

# Valid ranks (bulunanlar)
valid_ranks = [r for r in ranks if r < 999]
avg_rank = np.mean(valid_ranks) if valid_ranks else 999

# Sonuçları yazdır
print("\n" + "="*70)
print("--- TEST 4 SONUÇLARI ---")
print("="*70)

print(f"\n🎯 GENEL SONUÇLAR:")
print(f"  Toplam Analoji:        {total_analogies}")
print(f"  ")
print(f"  📊 Top-1 Accuracy:     {top1_acc:.1f}% ({top1_correct}/{total_analogies})")
print(f"  📊 Top-5 Accuracy:     {top5_acc:.1f}% ({top5_correct}/{total_analogies})")
print(f"  📊 Top-10 Accuracy:    {top10_acc:.1f}% ({top10_correct}/{total_analogies})")
print(f"  ")
print(f"  Ortalama Rank:         {avg_rank:.1f}")
print(f"  Bulunamayan:           {ranks.count(999)}")

print(f"\n📊 KATEGORİ BAZLI (Top-5 Accuracy):")
for cat_name, stats in category_stats.items():
    acc_top5 = (stats['top5'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"  {cat_name:20s}: {acc_top5:5.1f}% ({stats['top5']}/{stats['total']})")

print(f"\n📊 ALT KATEGORİ BAZLI (Top-5 Accuracy):")
for subcat, stats in sorted(subcategory_stats.items()):
    acc = (stats['top5'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"  {subcat:25s}: {acc:5.1f}% ({stats['top5']}/{stats['total']})")

# Başarılı örnekler
if successful_examples:
    print(f"\n🔍 BAŞARILI ÖRNEKLER:")
    print("="*70)
    for ex in successful_examples:
        print(f"\n  ✓ {ex['analogy']}")
        print(f"     Rank: #{ex['rank']}")
        print(f"     Top-5: {ex['top5']}")

# Başarısız örnekler
if failed_top5:
    print(f"\n🔍 BAŞARISIZ ÖRNEKLER (Top-5'te bulunamayan):")
    print("="*70)
    for fail in failed_top5:
        print(f"\n  ✗ {fail['analogy']} [{fail['category']}]")
        print(f"     Rank: {fail['rank']}")
        print(f"     Top-5: {fail['top5']}")
        print(f"     (Beklenen: {fail['analogy'].split('→')[1].strip()})")

print("\n" + "="*70)
print("✅ Test 4 tamamlandı!")
print("="*70)

# %%


# %%


# %%



