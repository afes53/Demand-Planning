# Security

Bu proje varsayılan olarak API anahtarı gerektirmez.

- `.env`, `.env.*` ve `.streamlit/secrets.toml` Git tarafından takip edilmez.
- Örnek değişkenler `.env.example` dosyasında tutulur.
- Token, şifre veya API anahtarı yanlışlıkla push edilirse yalnızca dosyayı silmek yeterli değildir:
  anahtar sağlayıcı panelinden iptal edilmeli ve yenisi oluşturulmalıdır.
- Hassas şirket verileri örnek CSV dosyalarıyla değiştirilmeden public repoya eklenmemelidir.
