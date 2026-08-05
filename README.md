# Demand Planning AI

Zero-shot zaman serisi modelleriyle mağaza–ürün seviyesinde talep tahmini üreten,
kullanıcının yüklediği gelecek stok dağıtım planını test eden ve stok açığı,
fazla stok, servis seviyesi, kayıp satış, transfer ve ek sevkiyat önerileri sunan
Streamlit tabanlı karar destek uygulaması.

> Final tahmin modeli yalnızca zero-shot modeller arasından seçilir.
> Naïve, sezonsal naïve ve hareketli ortalama yöntemleri benchmark amacıyla kullanılır.

## Canlı demo

- Streamlit: https://irrs8jkrjx2awizgo7lzgo.streamlit.app
- GitHub: https://github.com/afes53/Demand-Planning

Repo public yapıldıktan sonra bağlantıları gizli sekmede test edin.

## Temel özellikler

- Geçmiş satış ve gelecek stok planı için ayrı dosya yükleme
- Veri kalitesi raporu ve stokta-yok dönemlerinin işaretlenmesi
- Stokout dönemlerinde gözlenemeyen talebin düzeltilmesi
- Chronos Bolt Small, Chronos 2 ve opsiyonel TimesFM zero-shot tahmini
- Sezonsal naïve ve hareketli ortalama benchmarkları
- WMAPE, Bias, MAE, RMSE ve benchmark iyileşmesi
- Plan tarihleri için stok projeksiyonu
- Güvenlik stoğu, servis seviyesi ve stok tükenme tarihi
- Kayıp talep, kayıp satış değeri ve fazla stok analizi
- Depo tahsisi ve mağazalar arası transfer önerileri
- ABC–XYZ önceliklendirme
- Senaryo analizi
- Manuel tahmin düzenleme ve gerçekleşen veri sonrası FVA
- HTML yönetim özeti, Excel operasyon dosyası ve analitik ZIP çıktısı

## Örnek çıktılar

### Geçmiş satış görünümü

![Örnek geçmiş satış görünümü](docs/screenshots/ornek_gecmis_satis.png)

### Gelecek stok dağıtım planı

![Örnek gelecek dağıtım planı](docs/screenshots/ornek_dagitim_plani.png)

Canlı uygulamadan alınan ekran görüntülerini aynı klasöre ekleyerek bu bölümü
`app_ana_sayfa.png` ve `app_stok_riskleri.png` görselleriyle genişletebilirsiniz.

## Kullanılan modeller

Final tahmin modeli:

- Amazon Chronos Bolt Small
- Amazon Chronos 2
- TimesFM 2.5 — opsiyonel

Benchmarklar:

- Geçen dönem değeri
- Sezonsal naïve
- Hareketli ortalama

## Veri dosyaları

Repoda doğrudan çalıştırılabilen iki küçük örnek veri bulunur:

- `ornek_gecmis_satis_v4.csv`
- `ornek_gelecek_stok_plani_v4.csv`

### Geçmiş veri zorunlu alanları

- Tarih
- Mağaza ID
- Ürün ID
- Satılan miktar
- Stok

Fiyat, promosyon, gelen stok, tedarik süresi, kategori, bölge, maliyet,
stokout, iade ve yeni ürün bilgileri opsiyoneldir.

### Gelecek plan zorunlu alanları

- Tarih
- Mağaza ID
- Ürün ID
- Başlangıç/mevcut stok
- Planlanan gönderim

Beklenen giriş tarihi, dağıtılabilir depo stoğu, mağaza kapasitesi ve fiyat
opsiyoneldir.

Büyük veya gizli şirket verileri repoya yüklenmemelidir. README içinde veri
kaynağı ve indirme adımları açıklanmalı; hassas veriler `.gitignore` kapsamındaki
`data/raw/` dizininde tutulmalıdır.

## Kurulum

Önerilen Python sürümü: **3.11**

```bash
git clone GITHUB_REPO_LINKINI_BURAYA_EKLE
cd demand-planning-ai
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Bağımlılıkları yükleyin:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Uygulamayı çalıştırın:

```bash
streamlit run app.py
```

Tarayıcıda varsayılan olarak `http://localhost:8501` açılır.

## TimesFM kurulumu

TimesFM, Streamlit Community Cloud başlangıcını ve bellek kullanımını
ağırlaştırabileceği için ayrı tutulmuştur:

```bash
pip install -r requirements-timesfm.txt
```

Kurulduğunda uygulamadaki model listesinde otomatik görünür.

## Hızlı test

Model indirmeden repo yapısını, örnek dosyaları ve secret taramasını kontrol edin:

```bash
python scripts/validate_repository.py
python -m compileall -q .
```

İlk gerçek tahmin çalıştırmasında model ağırlıkları Hugging Face üzerinden
indirilebilir. İlk çalıştırma daha uzun sürebilir.

## Streamlit Community Cloud dağıtımı

1. Repoyu GitHub'a public olarak push edin.
2. Streamlit Community Cloud'da **Create app** seçin.
3. Repository ve `main` branch'ini seçin.
4. Main file path olarak `app.py` girin.
5. Python 3.11 kullanın.
6. Deploy sonrası Cloud logs ekranını kontrol edin.

TimesFM zorunlu değilse `requirements.txt` dosyasına eklemeyin; ayrı
`requirements-timesfm.txt` dosyası yerel veya daha güçlü ortamlarda kullanılabilir.

## Ortam değişkenleri ve güvenlik

Bu MVP varsayılan olarak API key gerektirmez.

```bash
cp .env.example .env
```

- `.env`, `.env.*` ve `.streamlit/secrets.toml` repoya girmez.
- Gerçek token veya şifreyi `.env.example` içine yazmayın.
- Daha önce bir anahtar push edildiyse dosyayı silmek yeterli değildir;
  anahtarı sağlayıcı panelinden iptal edip yenisini oluşturun.
- Ayrıntılar için `SECURITY.md` dosyasına bakın.

## Proje yapısı

```text
.
├── app.py
├── enterprise_analytics.py
├── demand_business_analytics_fixed.py
├── zero_shot_demand_mvp_core_generic_v2.py
├── requirements.txt
├── requirements-timesfm.txt
├── ornek_gecmis_satis_v4.csv
├── ornek_gelecek_stok_plani_v4.csv
├── docs/
│   └── screenshots/
├── scripts/
│   └── validate_repository.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── ARCHITECTURE.md
├── SECURITY.md
└── LICENSE
```

## Uygulama menüsü

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

## Notebook durumu

Bu repo uygulama tabanlıdır ve notebook içermez. Bu nedenle çalıştırılmış
notebook çıktısı gereksinimi uygulanmaz. İleride notebook eklenirse hücreler
çalıştırılmış ve çıktıları kayıtlı biçimde push edilmelidir.

## Sınırlamalar

- Streamlit oturum verileri kalıcı veritabanında tutulmaz.
- Kurumsal rol/yetki kontrolü henüz uygulanmamıştır.
- Gerçek FVA için tahmin döneminin gerçekleşen satışları yüklenmelidir.
- Model indirme ve tahmin süresi kullanılan donanıma göre değişir.

## Lisans

MIT — `LICENSE`
