import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ========== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Muat state_dict
state_dict = torch.load("vgg16_brain_tumor_best.pth", map_location=device)

# Buat model VGG16 standar (features saja)
model = models.vgg16(pretrained=False)

# Ambil informasi dari state_dict untuk classifier.7
weight7 = state_dict['classifier.7.weight']
in_features_7 = weight7.shape[1]
out_features_7 = weight7.shape[0]
print(f"✅ Detected classifier.7: Linear({in_features_7}, {out_features_7})")

# Bangun classifier yang sesuai dengan struktur state_dict
# Indeks: 0:Linear,1:ReLU,2:Dropout,3:Linear,4:ReLU,5:Dropout,6:ReLU,7:Linear
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.ReLU(inplace=True),          # layer indeks 6 (tanpa parameter)
    nn.Linear(in_features_7, out_features_7)  # layer indeks 7 (output)
)

# Muat bobot (strict=False mengabaikan ketidakcocokan non-parameter)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print("✅ Model 4 kelas berhasil dimuat!")

# ========== LABEL KELAS (4 kelas) ==========
# Sesuaikan urutan dengan training Anda. Contoh umum untuk tumor otak:
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
# Jika urutan berbeda, ganti sesuai urutan output model Anda.

# ========== PREPROCESSING ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== ENDPOINTS ==========
@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)

        label = CLASS_NAMES[predicted.item()]
        confidence = conf.item() * 100

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)