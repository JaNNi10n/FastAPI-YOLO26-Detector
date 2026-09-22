from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import shutil

# Load your custom-trained model
model = YOLO("best.pt")
app = FastAPI(title="Object Detection API")

# A simple endpoint to check if the server is running
@app.get("/health")
def health():
    return {"status": "API is live and ready!"}

# The endpoint that receives an image and returns predictions
@app.post("/predict")
async def predict(file: UploadFile):
    # Save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Run the model on the temporary image
    results = model("temp.jpg")
    
    # Return the coordinates and class names of detected objects
    return {
        "boxes": results[0].boxes.xyxy.tolist(),
        "classes": [model.names[int(c)] for c in results[0].boxes.cls.tolist()],
    }