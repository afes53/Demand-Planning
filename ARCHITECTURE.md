# Uygulama Mimarisi

## Tahmin katmanı

Final tahmin modeli yalnızca zero-shot modellerden seçilir:

- Chronos Bolt Small
- Chronos 2
- TimesFM 2.5 — kuruluysa

Naïve, sezonsal naïve ve hareketli ortalama yöntemleri yalnızca benchmark olarak kullanılır.

## İş katmanı

- Stokta-yok dönemlerinde gözlenemeyen talep düzeltmesi
- Gelecek plan tarihleriyle bire bir tahmin eşleştirmesi
- Beklenen giriş tarihine göre stok simülasyonu
- Güvenlik stoğu ve servis seviyesi
- Kayıp talep, kayıp satış değeri ve fazla stok
- Depo tahsisi ve mağazalar arası transfer önerisi
- ABC–XYZ önceliklendirme
- Senaryo karşılaştırması
- Manuel tahmin versiyonları ve gerçekleşen veri sonrası FVA

## Uygulama durumu

Bu MVP Streamlit `session_state` üzerinde çalışır. Kurumsal kullanımda aşağıdaki servislerin ayrıca eklenmesi gerekir:

- kullanıcı kimlik doğrulama
- rol ve yetki yönetimi
- kalıcı veritabanı
- tahmin/model versiyon deposu
- işlem ve değişiklik denetim kayıtları
- zamanlanmış model çalıştırmaları
- kurumsal PDF rapor servisi
