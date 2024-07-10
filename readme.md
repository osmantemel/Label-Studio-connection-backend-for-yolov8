## Proje Yapısı

İşte düzenlenmiş plan ve açıklamaları:

```plaintext
Label-Studio-connection-backend-for-yolov8/
├── iri_images/
├── Linux-ubuntu22.04-Conn/
│   ├── linux_conn_backend.py
├── backend_app.py
├── best.pt
├── http_server_with_cors.py
├── iris.json
├── README.md
```

- `iris_images/`: iris fotoğraflarının olduğu klasör
- `Linux-ubuntu22.04-Conn/`: Linux ortamı için Label studio ile bağlantı kurulabilmesi için gerekli dosyalar.
  - `linux_conn_backend.py`: Ubuntu 22.04 için backend dosyası
- `backend_app.py`: Windows işletim sistemi için backend kodları
- `best.pt`: Yolov8 ile eğitilmiş iris  tespit için yapay zeka
- `http_server_with_cors`: Localde Server açıp fotoğrafları yapyalşamaya yarayan flask uygulaması
- `iris.json`: iris fotoğraflarının yollarının tutulduğu json dosyası
- `README.md`: Projenin genel açıklamasını ve kurulum talimatlarını içeren dosya.
```

## Kurulum

Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### Gerekli Yazılımlar

- Python (>= 3.6)
- npm veya yarn paket yöneticisi
- Label Studio

### Adımlar

1. Bu projeyi klonlayın veya indirin:

   ```bash
   git clone https://github.com/osmantemel/Label-Studio-connection-backend-for-yolov8.git
   cd Label-Studio-connection-backend-for-yolov8
   ```

2. Gerekli bağımlılıkları yükleyin:

   ```bash
   pip install ultralytics
   pip install Flask
   pip install label_studio_ml
   pip install label_studio_sdk
   pip install json
   pip install datetime   
   pip install io 
   pip install PIL
   pip install requests
   ```

3. Uygulamayı çalıştırın:
   ```bash
    python backend_app.py
    # veya
    python3 backend_app.py
   ```

4. Tarayıcınızda `http://localhost:5000`adresine giderek uygulamayı görüntüleyin.
