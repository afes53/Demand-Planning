# Demand Planning AI — Dağıtım Planı Sürümü

Bu sürümde gelecek tarihleri uygulama üretmez. Kullanıcı iki dosya yükler:

1. Geçmiş satış verisi
2. Gelecek stok dağıtım planı

Model, dağıtım planındaki mağaza–ürün ve tarihler için talep tahmini üretir. Daha sonra:

- başlangıç stoğu,
- planlanan sevkiyat,
- tahmini talep,
- güvenlik stoğu

birlikte kullanılarak planın yeterliliği hesaplanır.

## Gelecek plan dosyası

Zorunlu alanlar:

- tarih
- mağaza ID
- ürün ID
- başlangıç stoğu
- planlanan sevkiyat

Başlangıç stoğu her mağaza–ürünün ilk plan tarihinde dolu olmalıdır. Sonraki tarihlerde boş bırakılabilir.

Opsiyonel:

- fiyat

Plan tarihleri geçmiş verinin hemen sonraki döneminde başlamalı ve kesintisiz olmalıdır.

## Hesaplama

Mevcut plan:

```text
Dönem sonu stok
= dönem başı stok
+ planlanan sevkiyat
- tahmini talep
```

Önerilen ek sevkiyat, her dönemde talebi karşılayıp güvenlik stoğunu koruyacak minimum miktardır.

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
```

TimesFM desteği:

```bash
pip install -r requirements-timesfm.txt
```
