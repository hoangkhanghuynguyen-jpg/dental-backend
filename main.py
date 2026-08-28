"""
Backend nhận diện sâu răng — FastAPI

Chạy thử:
    pip install -r requirements.txt
    uvicorn main:app --reload

Endpoint chính: POST /api/detect
Body: { "image": "data:image/jpeg;base64,...." }
Trả về: { "detections": [ { "label": str, "confidence": float, "bbox": [x, y, w, h] }, ... ] }

Hiện tại hàm detect_cavities() đang dùng logic giả lập (random) để bạn test
luồng frontend <-> backend trước. Khi có model đã train (vd YOLOv8), xem
phần "GẮN MODEL THẬT VÀO ĐÂY" bên dưới để thay thế.
"""

import base64
import io

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Cavity Detector API")

# Cho phép frontend (mở bằng Live Server hoặc file://) gọi tới backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/đồ án: để "*" cho tiện; sản phẩm thật nên giới hạn domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model 1 lần lúc khởi động app (không load lại mỗi request, sẽ rất chậm)
# Đặt file .pt của bạn vào cùng thư mục backend/ và đổi tên bên dưới cho khớp
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


class DetectRequest(BaseModel):
    image: str  # data URL dạng "data:image/jpeg;base64,...."


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x, y, w, h] theo pixel của ảnh gốc


class DetectResponse(BaseModel):
    detections: list[Detection]


def decode_base64_image(data_url: str) -> Image.Image:
    """Chuyển data URL base64 từ frontend thành ảnh PIL."""
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def detect_cavities(image: Image.Image) -> list[Detection]:
    """Chạy model YOLOv8 thật trên ảnh và trả về danh sách phát hiện."""
    results = model.predict(image, conf=0.1, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                label=model.names[int(box.cls[0])],
                confidence=round(float(box.conf[0]), 3),
                bbox=[x1, y1, x2 - x1, y2 - y1],
            )
        )
    return detections


@app.post("/api/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    image = decode_base64_image(req.image)
    detections = detect_cavities(image)
    return DetectResponse(detections=detections)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Cavity Detector API đang chạy"}
