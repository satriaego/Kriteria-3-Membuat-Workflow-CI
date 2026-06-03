## CI/CD Pipeline & Deployment Status

Proyek ini telah diintegrasikan dengan sistem Otomatisasi CI/CD (GitHub Actions) dan sistem Containerization (Docker Hub) untuk mendukung siklus MLOps yang berkelanjutan.

### 📦 Artefak Hasil Training & Kontainerisasi
* **GitHub Actions Artifact (Level Skilled):** Setiap proses training otomatis akan mengeluarkan file biner `model.pkl` yang dapat diunduh langsung pada tab **Actions** > **Summary** > **Artifacts** (Asset: `titanic-trained-model`).
* **Docker Hub Registry (Level Advanced):** Image hasil build otomatis dari pipeline CI telah berhasil didorong ke repositori publik dan siap digunakan untuk proses deployment / serving.

📌 **URL Docker Hub:**  
👉 [https://hub.docker.com/r/satriaego/titanic-model-ci](https://hub.docker.com/r/satriaego/titanic-model-ci)

---
