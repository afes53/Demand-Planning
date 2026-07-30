# Demand Planning AI — Streamlit

Bu uygulama:

- CSV/XLSX/Parquet perakende verisi yükler,
- sütunları standart şemaya eşler,
- stokout dönemlerindeki gözlenemeyen talebi düzeltir,
- Chronos veya opsiyonel TimesFM ile zero-shot tahmin üretir,
- tarihsel kayıp satış ve kayıp ciroyu hesaplar,
- stok açığı ve ikmal önerisi oluşturur,
- fiyat/ciro bazlı öncelik verir,
- ürün ABC analizi yapar,
- sonuçları CSV/ZIP olarak indirir.

Ayrıca FreshRetailNet demo modu bulunur. Bu modda gerçek adet, fiyat ve stok bulunmadığı için demo amaçlı global adet katsayısı, ürün fiyatı ve stok senaryosu oluşturulur.

## Dosyalar

- `app.py`: Streamlit arayüzü
- `zero_shot_demand_mvp_core_generic_v2.py`: veri pipeline'ı ve tahmin modelleri
- `demand_business_analytics_fixed.py`: stokout, ciro, ikmal ve ABC analizleri
- `freshretail_demo_preprocessing.py`: yalnızca FreshRetailNet demo dönüşümleri
- `requirements.txt`: temel bağımlılıklar
- `requirements-timesfm.txt`: opsiyonel TimesFM bağımlılığı

## Yerelde çalıştırma

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

Kurulum:

```bash
pip install -r requirements.txt
```

Çalıştırma:

```bash
streamlit run app.py
```

TimesFM desteği de gerekiyorsa:

```bash
pip install -r requirements-timesfm.txt
```

## Streamlit Community Cloud

Repo kökünde `app.py` ve `requirements.txt` bulunmalıdır. Uygulamanın ilk model çalıştırmasında model dosyaları Hugging Face üzerinden indirilir. CPU ortamında önce `Chronos Bolt Small` ve düşük seri sayısıyla deneme yapılması önerilir.

## Şirket verisi için beklenen temel alanlar

Zorunlu:

- tarih
- mağaza ID
- ürün ID
- satış
- stok

Opsiyonel:

- fiyat
- kategori 1/2/3
- stokout bayrağı

Stok sütunu yoksa uygulamada “Stok sütunum yok” seçilebilir; bu durumda stokout bayrağı gerekir ve gerçek ikmal miktarı hesaplanamaz.

## Önemli ayrım

FreshRetailNet demo modundaki adet, fiyat ve stoklar gerçek kayıt değildir. Şirket verisi modunda uygulama doğrudan yüklenen gerçek satış, fiyat ve stok değerlerini kullanır.
