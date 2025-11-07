"""Convert existing test data from Python to JSON format.

This script extracts test data from test_suite_script.py and converts
it to the new JSON schema format.
"""

import json
from datetime import date
from pathlib import Path

# Output path
output_path = Path(__file__).parent.parent / "data" / "test_data.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CORPUS (from test_suite_script.py lines 45-92)
# ==============================================================================

corpus_docs_list = [
    # --- ENFLASYON & PARA POLİTİKASI (c0-c5) ---
    "Türkiye Cumhuriyet Merkez Bankası (TCMB), Para Politikası Kurulu toplantısında politika faizini 500 baz puan artırarak yüzde 30'a yükseltti. Enflasyonla mücadele kapsamında sıkı para politikası sürdürülecek.",
    "Tüketici Fiyat Endeksi (TÜFE) Ekim ayında aylık bazda yüzde 3.4 artış gösterdi. Yıllık enflasyon yüzde 61.5 seviyesinde gerçekleşti. En yüksek artış gıda grubunda gözlendi.",
    "Merkez Bankası faiz artırımı piyasalarda olumlu karşılandı. Analistler, sıkı para politikasının enflasyon beklentilerini düşüreceğini öngörüyor.",
    "Enflasyon hedeflemesi rejimi, merkez bankalarının fiyat istikrarını sağlamak için kullandığı bir para politikası stratejisidir. Türkiye 2006'dan beri bu rejimi uygulamaktadır.",
    "Yüksek enflasyon ortamında tüketicilerin satın alma gücü azalır. Sabit gelirli vatandaşlar en çok etkilenen kesimdir. Devlet, enflasyonla mücadele için sıkı maliye politikası uygulamaktadır.",
    "Üretici Fiyat Endeksi (ÜFE) Ekim ayında yıllık bazda yüzde 55.8 arttı. ÜFE artışının TÜFE'ye yansıması birkaç ay içinde bekleniyor.",

    # --- HİSSE SENEDİ & BORSA (c6-c12) ---
    "BIST 100 endeksi günü yüzde 2.8 artışla 8.450 seviyesinden kapattı. Bankacılık endeksi günün yıldızı oldu. Yabancı yatırımcılar net alıcı konumundaydı.",
    "X Bankası hisseleri bugün yüzde 5.2 değer kazandı. Bankanın açıkladığı güçlü bilanço verileri yatırımcılar tarafından olumlu karşılandı. Öz sermaye karlılığı yüzde 18'e yükseldi.",
    "Halka arz tarihi açıklanan Y Teknoloji şirketi, 2 milyar TL değerleme ile Borsa İstanbul'da işlem görmeye başlayacak. Talep toplaması önümüzdeki hafta başlıyor.",
    "Temettü dağıtımı açısından en cömert şirketler listesinde enerji ve bankacılık sektörü öne çıkıyor. Z Bankası, hisse başına 1.5 TL temettü dağıtacağını açıkladı.",
    "Teknik analizde direnç seviyesi 8.500'de görülüyor. Bu seviye aşılırsa BIST 100'ün 9.000'e doğru yükseliş potansiyeli var. RSI göstergesi 65 seviyesinde.",
    "Hisse senedi yatırımında temel analiz yöntemi, şirketlerin finansal tablolarını, bilanço verilerini ve sektör dinamiklerini incelemeyi içerir. F/K oranı önemli bir göstergedir.",
    "Borsa İstanbul'da işlem gören şirketlerin toplam piyasa değeri 7.2 trilyon TL'ye ulaştı. Yabancı yatırımcı oranı yüzde 45 seviyesinde.",

    # --- KREDİ & BANKACILIK (c13-c18) ---
    "Konut kredisi faiz oranları son 3 ayda 200 baz puan arttı. Bankalar, yüksek enflasyon nedeniyle kredi faizlerini yukarı çekti. Ortalama konut kredisi faizi yüzde 36 seviyesinde.",
    "Kredi kartı borçlarını ödeyemeyen vatandaş sayısı artıyor. Bankalar, taksitlendirme imkanları sunuyor. Asgari ödeme tuzağına düşmemek için tam ödeme öneriliyor.",
    "Findeks kredi notu 1000-1900 aralığında belirleniyor. 1400'ün üzerindeki notlar iyi kabul ediliyor. Düzenli ödeme yapanların notu yükseliyor.",
    "Ticari kredilerde KGF (Kredi Garanti Fonu) kefaleti ile KOBİ'lere düşük faizli finansman imkanı sağlanıyor. Başvurular bankalar aracılığıyla yapılıyor.",
    "Tüketici kredisi faizleri yüzde 40 seviyesini aştı. Bireysel ihtiyaç kredisi talepleri düşüş gösteriyor. Bankalar riskli kredilerde teminat istiyor.",
    "Kredili mevduat hesabı (kredi kartından farklı), müşterilerin belirli bir limite kadar çek kesmesine izin verir. Sadece kullanılan tutar için faiz ödenir.",

    # --- DÖVİZ & ALTIN (c19-c24) ---
    "Dolar/TL kuru bugün yüzde 0.8 değer kazanarak 32.50 seviyesine yükseldi. TCMB'nin döviz rezervlerinde azalma görüldü. Piyasa, yeni faiz kararını bekliyor.",
    "Euro/TL paritesi 35.20 seviyesinde işlem görüyor. Avrupa Merkez Bankası'nın faiz kararları Türk lirasını da etkiliyor.",
    "Altın ons fiyatı 2.050 dolar seviyesini test ediyor. Gram altın ise 1.850 TL'den işlem görüyor. Yatırımcılar enflasyona karşı altına yöneliyor.",
    "Döviz tevdiat hesaplarında tutulan toplam miktar 280 milyar dolara yaklaştı. Vatandaşlar dolarizasyon eğilimi gösteriyor.",
    "TCMB, döviz piyasalarında istikrarı sağlamak için swap ihaleleri düzenliyor. Döviz likiditesi kontrol altında tutuluyor.",
    "Kripto para piyasasında Bitcoin 43.000 dolar seviyesini gördü. Türkiye'de kripto para kullanımı yaygınlaşıyor ancak düzenleme belirsizliği sürüyor.",

    # --- TAHVİL & BORÇLANMA (c25-c29) ---
    "Hazine, 10 yıllık tahvil ihalesinde 18 milyar TL borçlandı. Bileşik faiz yüzde 28.5 olarak gerçekleşti. Talep/teklif oranı 2.1 seviyesinde.",
    "Eurobond faizleri geriledi. Türkiye'nin 2034 vadeli dolar cinsi tahvilinin getirisi yüzde 7.8'e düştü. Risk algısı iyileşiyor.",
    "Şirket tahvilleri piyasasında ihraç hacmi arttı. A+ notu olan şirketler yatırımcılar tarafından tercih ediliyor. Getiri oranları yüzde 30-35 arasında.",
    "Devlet İç Borçlanma Senetleri (DİBS) portföyünde ağırlık artıyor. Bankalar, kredi vermek yerine tahvil almayı tercih ediyor.",
    "Tahvil piyasasında vade uzadıkça getiri yükseliyor. 5 yıllık tahvil faizi yüzde 27 iken, 10 yıllık yüzde 29 seviyesinde.",

    # --- MAKRO EKONOMİ (c30-c34) ---
    "Cari açık Eylül ayında 4.2 milyar dolar olarak gerçekleşti. Enerji ithalatı cari açığın ana nedeni. Turizm gelirlerindeki artış açığı sınırladı.",
    "Gayri Safi Yurt İçi Hasıla (GSYİH) üçüncü çeyde yıllık bazda yüzde 4.1 büyüdü. İmalat sanayi ve inşaat sektörü büyümeye katkı sağladı.",
    "İşsizlik oranı yüzde 9.4 seviyesinde. Genç işsizlik ise yüzde 16.8 olarak açıklandı. Hizmet sektöründe istihdam artıyor.",
    "Merkez Bankası brüt döviz rezervleri 140 milyar dolar seviyesinde. Net rezervler ise tartışmalı, swap hariç hesaplama yapılıyor.",
    "Bütçe açığı hedefin üzerinde seyrediyor. Hazine, yılsonu hedefini revize edebilir. Vergi gelirleri beklentilerin altında kaldı.",
]

corpus = {f'c{i}': doc for i, doc in enumerate(corpus_docs_list)}

# ==============================================================================
# QUERIES (from test_suite_script.py lines 103-116)
# ==============================================================================

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

# ==============================================================================
# RELEVANT DOCS (from test_suite_script.py lines 126-214)
# ==============================================================================

relevant_docs_with_scores = {
    'q0': {'c0': 2, 'c2': 2, 'c3': 1, 'c4': 1, 'c1': 1},
    'q1': {'c6': 2, 'c10': 2, 'c12': 1, 'c7': 1},
    'q2': {'c13': 2, 'c17': 1, 'c14': 1, 'c16': 1},
    'q3': {'c19': 2, 'c23': 1, 'c22': 1, 'c20': 1},
    'q4': {'c9': 2, 'c7': 1},
    'q5': {'c25': 2, 'c29': 1, 'c28': 1, 'c27': 1},
    'q6': {'c15': 2, 'c14': 1},
    'q7': {'c30': 2, 'c31': 1, 'c33': 1},
    'q8': {'c8': 2, 'c12': 1},
    'q9': {'c21': 2},
    'q10': {'c31': 2, 'c32': 1, 'c30': 1},
    'q11': {'c0': 2, 'c1': 2, 'c2': 2, 'c5': 2, 'c3': 1, 'c4': 1},
}

# ==============================================================================
# Create JSON structure
# ==============================================================================

test_data = {
    "metadata": {
        "version": "1.0",
        "description": "Turkish Financial Domain Semantic Search Evaluation Suite",
        "language": "tr",
        "domain": "finance",
        "created_date": str(date.today()),
        "total_tasks": 1  # Only IR for now
    },

    "information_retrieval": {
        "name": "Information Retrieval & Ranking",
        "enabled": True,
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs_with_scores,
        "config": {
            "ndcg_at_k": [1, 3, 5, 10],
            "map_at_k": [1, 3, 5, 10],
            "mrr_at_k": [1, 3, 5, 10]
        }
    }
}

# Save to file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"✅ Test data JSON created successfully!")
print(f"📍 Location: {output_path}")
print(f"📊 Stats:")
print(f"   - Corpus size: {len(corpus)}")
print(f"   - Query count: {len(queries)}")
print(f"   - Total relevance judgments: {sum(len(docs) for docs in relevant_docs_with_scores.values())}")
