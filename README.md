# 🛰️ Real-Time Object Detection & Alert System

A Python-based system that captures live video from a webcam, detects **Humans**, **Animals**, and **Drones** using YOLOv8, enhances image quality, saves results in **MongoDB**, and sends **email alerts** with detections.  

> 📡 Built for surveillance, security, and smart monitoring applications.

---

## 📌 Features

- 🎯 Real-time object detection (Human, Animal, Drone)
- 🖼️ Image enhancement (denoising + contrast)
- 📧 Email alerts with detection images
- 🗃️ MongoDB + GridFS integration for storage
- 🌐 Automatic location tagging (via IP)
- 🔁 Fallback logic if drone model fails
- 🖥️ Live webcam preview with bounding boxes

---

## 🧰 Tech Stack

| Component      | Tech/Tool                    |
|----------------|------------------------------|
| Detection      | YOLOv8 (Ultralytics)         |
| Language       | Python 3.8+                  |
| Image Processing | OpenCV, Scikit-Image        |
| Database       | MongoDB + GridFS             |
| Location       | Geocoder                     |
| Email Alerts   | SMTP (Gmail)                 |

---
## 🏗️ Project Structure

```
├── main.py
├── yolov8n.pt                  # General model (humans/animals)
├── yolov8m-drone.pt            # Optional: custom drone model
├── requirements.txt
├── sample_output.jpg
└── README.md
```

---

## 🚀 How It Works

1. Captures video from webcam
2. Enhances frames using wavelet denoising + histogram equalization
3. Runs YOLOv8 model(s) to detect:
   - Human (class 0)
   - Animal (classes 15–25)
   - Drones (via custom model or fallback)
4. Draws labeled bounding boxes
5. Saves detection metadata + image to MongoDB
6. Sends email alert with image (cooldown = 30s)
7. Live display in OpenCV window

---

## 🧪 Detection Categories

| Object Type | Color | Confidence Threshold |
|-------------|-------|----------------------|
| Human       | Green | 0.7                  |
| Animal      | Yellow| 0.7                  |
| Drone       | Red   | 0.3 (or fallback)    |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/object-detector.git
cd object-detector
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add YOLOv8 Models
- Download [YOLOv8n](https://github.com/ultralytics/ultralytics) and place `yolov8n.pt` in the root directory
- Add your custom drone model as `yolov8m-drone.pt` *(optional)*

### 4. MongoDB Setup
Make sure MongoDB is running locally at:
```bash
mongodb://localhost:27017
```

It will auto-create:
- DB: `pjt2`
- Collections: `human_detections`, `animal_detections`, `drone_detections`

### 5. Email Alert Setup
Inside `send_email_with_image()` function:
```python
msg['From'] = 'your_email@gmail.com'
msg['To'] = 'recipient_email@gmail.com'
smtp.login('your_email@gmail.com', 'your_app_password')
```

Use **App Password** for Gmail if 2FA is enabled.

---

## 🧠 MongoDB Entry Sample

```json
{
  "timestamp": "2025-07-08 12:30:45",
  "objects_detected": ["Human (0.89)"],
  "location": "Lat: 28.61, Lon: 77.23",
  "image_id": "64a7f6d4567c3b0a3c0f4e9b"
}
```

---

## ⌨️ Run the App

```bash
python main.py
```

- Press **`q`** to quit webcam display
- Detection preview shown with bounding boxes

---

## 📅 Cooldown Logic

- Email alerts are throttled to **1 alert every 30 seconds**
- Helps avoid spam from repeated detections

---

## 📌 TODO (Future Improvements)

- 🖥 Web dashboard to browse detections
- 🔊 Audio alerts
- 📲 WhatsApp / SMS integration
- ☁️ Cloud deployment (AWS/GCP)
- 🔐 Secure login for email config

---

## 👤 Author

**Anshik Mantri**  
📧 [anshik77mantri@gmail.com](mailto:anshik77mantri@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/anshikmantri/)  
💻 [GitHub](https://github.com/anshikmantri77)

---


