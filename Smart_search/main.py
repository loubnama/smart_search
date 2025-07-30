from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import os
import base64
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI(title="Image Similarity Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مسار مجلد الصور
images_folder = "smart_search/Smart_search/images"  # للنشر على Render
# images_folder = "./images"  # للاختبار محليًا
image_vectors = []
image_names = []

# تحميل نموذج ResNet50
model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")  # إصلاح التحذير
model.eval()

# تحويل الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vector(img: Image.Image):
    img.thumbnail((256, 256))  # تقليل الحجم قبل المعالجة
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vector = model(img_tensor)
    return vector.numpy().flatten()

# تجهيز قاعدة بيانات الصور (الحد الأقصى 20 صورة)
for filename in os.listdir(images_folder)[:20]:  # الحد الأقصى 20 صورة
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(images_folder, filename)
        img = Image.open(path).convert("RGB")
        vec = extract_vector(img)
        image_vectors.append(vec)
        image_names.append(filename)

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    try:
        # فتح الصورة المدخلة
        input_img = Image.open(file.file).convert("RGB")
        input_img.thumbnail((256, 256))  # تقليل الحجم
        query_vector = extract_vector(input_img)

        # حساب التشابه
        similarities = cosine_similarity([query_vector], image_vectors)[0]
        top_indices = similarities.argsort()[-2:][::-1]  # إرجاع صورتين فقط

        # تجهيز الصور المشابهة
        results = []
        for idx in top_indices:
            image_path = os.path.join(images_folder, image_names[idx])
            with Image.open(image_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=50)  # جودة منخفضة
                encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results.append({
                "filename": image_names[idx],
                "score": float(similarities[idx]),
                "image_base64": encoded_string
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
