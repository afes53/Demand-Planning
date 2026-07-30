# Demand Planning AI — Dashboard v2

Bu sürümde demo ayar ekranı kaldırılmıştır. Ana akış:

1. Örnek veri setini indir veya kendi CSV/XLSX/Parquet dosyanı yükle.
2. Satış, stok, fiyat ve kimlik sütunlarını eşleştir.
3. Zero-shot tahmin ve stok analizlerini çalıştır.
4. Karar destek dashboardunda ikmal, ciro riski ve ABC sonuçlarını incele.

## Dashboard özellikleri

- Geçmiş satış + düzeltilmiş talep + gelecek tahmini çizgisi
- Geçmiş stok + tahmini kalan stok grafiği
- Risk seviyesi donut grafiği
- Stokout oranı ve iş etkisi baloncuk grafiği
- En yüksek ikmal ihtiyacı yatay bar grafiği
- Mağaza–ürün ikmal heatmap'i
- Tarihsel ve gelecek ciro riski karşılaştırması
- ABC treemap ve donut grafikleri
- Sadeleştirilmiş Türkçe tablolar
- Tüm sonuçları ZIP olarak indirme

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
```

TimesFM desteği:

```bash
pip install -r requirements-timesfm.txt
```

## Önemli

Gelecek tahminleri her zaman kullanıcının yüklediği veri setindeki satış geçmişinden üretilir. Stok sütunu varsa aynı veri setindeki son stok seviyesi kullanılarak tahmini kalan stok, stokout tarihi ve ikmal ihtiyacı hesaplanır.
