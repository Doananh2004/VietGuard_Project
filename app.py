from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load model
MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# Predict function
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    
    return "Tin thật 🟢" if pred == 1 else "Tin giả 🔴"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    
    if request.method == "POST":
        text = request.form["text"]
        result = predict_text(text)
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)