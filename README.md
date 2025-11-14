## Türkçe Spam SMS Sentetik Veri Üretimi

Bu depo, eğitim/araştırma amaçlı olarak Türkçe SMS spam tespiti için sentetik veri üretir.

### Özet
- Kod, şablon + yer tutucu + gürültü yaklaşımıyla gerçekçi görünen ancak sentetik mesajlar üretir.
- Girdi: `Test/seed.csv` (sütunlar: `label,text`).
- Çıktı: `Test/augmented_sms.csv`.

### Veri Seti
- Kaggle: [Turkish Mail Spam Dataset](https://www.kaggle.com/datasets/alpersah11/turkish-mail-spam-dataset)

### Kullanım
1. `Test/generate_sms_dataset.ipynb` dosyasını açın ve hücreleri sırayla çalıştırın.
2. Parametreleri ilk hücreden güncelleyebilirsiniz (`RANDOM_SEED`, `N_PER_SEED`, `DEDUP_THR`).
3. Colab uyumlu yardımcılar son hücrede mevcuttur.

### Veri Politikası ve Uyarı
- Bu depodaki tüm veriler sentetiktir; gerçek kişi, kurum, telefon, kimlik bilgisi veya gerçek domain içermez.
- Örnek linkler `https://example.com/...` veya `[LINK]` biçimindedir.
- Gerçek marka/kurum isimleri kullanılmaz; `[BANKA]`, `Kargo Firması`, `Operatörünüz` gibi yer tutucular tercih edilir.
- Amaç eğitim ve araştırmadır; kötüye kullanım amacıyla kullanılmamalıdır.

### Büyük Dosyalar
- `Test/augmented_sms.csv` büyüyebilir. Gerekirse Git LFS kullanın veya repoda takip etmeyin.
