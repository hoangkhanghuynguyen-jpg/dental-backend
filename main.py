import base64
import io
import os
import gdown
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Cavity Detector API")

# Cấu hình CORS để web gửi ảnh qua được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE ---
MODEL_PATH = "best.pt"
FILE_ID = "1R4aROAg2ZvFu2cscJ9Gi3AWOsIKANKjs" # ID file Drive của bạn

# Kiểm tra nếu chưa có file best.pt thì tải về
if not os.path.exists(MODEL_PATH):
    print("Đang tải model từ Google Drive xuống máy chủ...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model AI
model = YOLO(MODEL_PATH)

# --- API NHẬN DIỆN ---
class DetectRequest(BaseModel):
    image: str

@app.post("/api/detect")
async def detect(req: DetectRequest):
    # Giải mã ảnh base64 từ frontend gửi lên
    image_data = req.image
    if "," in image_data:
        image_data = image_data.split(",")[1]
        
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Chạy YOLO nhận diện (để conf=0.15 cho nhạy)
    results = model.predict(image, conf=0.25)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1] # x, y, width, height
            })
            
    return {"detections": detections}

# API kiểm tra kết nối
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Máy chủ nha khoa đang chạy!"}
