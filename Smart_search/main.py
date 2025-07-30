from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import os
import base64
import io

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اسمح لجميع المصادر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعدادات المسارات
images_folder = "images"
vectors_file = "image_vectors.npy"
names_file = "image_names.npy"

# تحميل نموذج CLIP
clip_model = SentenceTransformer("clip-ViT-B-32")

# تحميل الصور أو استرجاعها من ملفات التخزين
if os.path.exists(vectors_file) and os.path.exists(names_file):
    image_vectors = np.load(vectors_file)
    image_names = np.load(names_file)
else:
    image_vectors = []
    image_names = []

    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(images_folder, filename)
            img = Image.open(path).convert("RGB")
            vec = clip_model.encode(img, convert_to_numpy=True, show_progress_bar=False)
            image_vectors.append(vec)
            image_names.append(filename)

    image_vectors = np.array(image_vectors)
    np.save(vectors_file, image_vectors)
    np.save(names_file, image_names)

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    try:
        # فتح الصورة المرسلة من المستخدم
        input_img = Image.open(file.file).convert("RGB")
        query_vector = clip_model.encode(input_img, convert_to_numpy=True)

        # حساب التشابه مع الصور المخزنة
        similarities = cosine_similarity([query_vector], image_vectors)[0]
        top_indices = similarities.argsort()[-5:][::-1]

        # تجهيز الرد مع الصور المشابهة
        results = []
        for idx in top_indices:
            image_path = os.path.join(images_folder, image_names[idx])
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            results.append({
                "filename": image_names[idx],
                "score": float(similarities[idx]),
                "image_base64": encoded_string
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
