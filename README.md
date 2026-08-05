# Demand Planning AI — Enterprise Dashboard v4

Bu sürüm, rol bazlı ve 13 ekranlı bir talep/stok planlama uygulamasıdır.

## Menü

1. Ana Sayfa  
2. Veri Yükleme  
3. Veri Kalitesi  
4. Talep Tahmini  
5. Tahmin Performansı  
6. Stok Riskleri  
7. Stok Dağıtım Önerileri  
8. Mağaza / Ürün Detayı  
9. ABC–XYZ Önceliklendirme  
10. Senaryo Analizi  
11. Manuel Düzeltme ve FVA  
12. Raporlar  
13. Model ve Veri Bilgileri  

## Tahmin modelleri

Final tahmin modeli yalnızca zero-shot model kataloğundan seçilir:

- Amazon Chronos Bolt Small
- Amazon Chronos 2
- TimesFM 2.5 — opsiyonel kurulum

Aşağıdaki yöntemler final model değildir; benchmark olarak kullanılır:

- Geçen dönem değeri
- Sezonsal Naïve
- Hareketli ortalama

## Veri akışı

1. Geçmiş satış dosyası yüklenir.
2. Gelecek stok dağıtım planı yüklenir.
3. Veri kalitesi ve stokta-yok dönemleri kontrol edilir.
4. Zero-shot modeller geçmiş verinin son dönemlerinde backtest edilir.
5. En iyi zero-shot model, plan dosyasındaki gelecek tarihler için talep tahmini üretir.
6. Başlangıç stoğu, beklenen giriş tarihi, planlanan gönderim ve tahmini talep birlikte simüle edilir.
7. Stok açığı, fazla stok, servis seviyesi, ciro riski, transfer ve ek gönderim önerileri hesaplanır.

## Kurulum

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

TimesFM desteği:

```bash
pip install -r requirements-timesfm.txt
```

## Örnek dosyalar

- `ornek_gecmis_satis_v4.csv`
- `ornek_gelecek_stok_plani_v4.csv`

## Uygulama notları

- Tahmin ufku uygulama tarafından rastgele oluşturulmaz; gelecek stok planındaki tarih aralığından alınır.
- Stokta-yok dönemlerinde satış ile gerçek talep ayrıştırılır.
- Güven aralıkları, seçilen zero-shot modelin backtest artıklarından ampirik olarak hesaplanır.
- FVA, tahmin dönemi gerçekleşen satışları geldikten sonra hesaplanabilir.
- Senaryo analizi talep, sevkiyat, gecikme, güvenlik stoğu ve servis seviyesi varsayımlarını değiştirir.
