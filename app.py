import os
import re
import sqlite3

from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from underthesea import word_tokenize

app = Flask(__name__)

# Load model
MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

MAX_TOKENS = 256  # Phải khớp với max_length khi training

# Khởi tạo LIME explainer
explainer = LimeTextExplainer(class_names=["Tin giả", "Tin thật"])

def lime_predictor(texts):
    """Hàm dành riêng cho LIME: Tính xác suất cho nhiều văn bản cùng lúc"""
    probs = []
    for t in texts:
        t = clean_text(t)
        token_ids = tokenizer.encode(t, add_special_tokens=False)
        if not token_ids:
            probs.append([0.5, 0.5])
            continue
            
        # Tối ưu tốc độ: LIME chỉ cần kiểm tra 1 đoạn (chunk) đầu tiên
        chunk = token_ids[:MAX_TOKENS - 2]
        input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([[1] * len(input_ids)])
        }
        with torch.no_grad():
            outputs = model(**inputs)
            p = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
            probs.append(p)
    return np.array(probs)

# ---------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction_label TEXT,
            confidence REAL,
            user_feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)   # Xóa URL
    text = re.sub(r'\s+', ' ', text)      # Chuẩn hóa khoảng trắng
    return text.strip()

def predict_text(text):
    try:
        text = clean_text(text)
        
        # Lấy token ID thô (không có special tokens <s>, </s>)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Nếu bài viết rỗng sau khi clean
        if not token_ids:
            return None

        # Chunking: chia nhỏ bài báo thành nhiều đoạn
        # Bớt đi 2 token cho <s> và </s>
        chunk_size = MAX_TOKENS - 2
        chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
        
        chunk_probs = []
        for chunk in chunks:
            # Thêm special tokens cho từng chunk
            input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
            inputs = {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([[1] * len(input_ids)])
            }
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Tính xác suất (Softmax)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                chunk_probs.append(probs[0])
        
        # Trung bình xác suất của tất cả các chunks
        avg_probs = torch.stack(chunk_probs).mean(dim=0)
        
        pred = torch.argmax(avg_probs).item()
        confidence = avg_probs[pred].item() * 100
        
        label = "Tin thật" if pred == 1 else "Tin giả"
        icon = "🟢" if pred == 1 else "🔴"
        
        # --- Giải thích bằng LIME (Explainable AI) ---
        # Chỉ giải thích 1000 ký tự đầu tiên để giữ tốc độ phản hồi nhanh
        # Tách từ ghép (Word Segmentation) bằng underthesea để LIME nhận diện cụm từ (ví dụ: "chính_phủ")
        text_for_lime = word_tokenize(text[:1000], format="text")
        
        # num_samples=50: tạo 50 mẫu nhiễu (đủ tốt và rất nhanh)
        exp = explainer.explain_instance(text_for_lime, lime_predictor, num_features=6, num_samples=50)
        
        explanation = []
        for word, weight in exp.as_list():
            explanation.append({
                "word": word.replace('_', ' '), # Đổi lại dấu cách để hiện lên UI cho đẹp
                "weight": round(abs(weight), 4),
                "type": "real" if weight > 0 else "fake"
            })
        
        return {
            "label": f"{label} {icon}",
            "confidence": round(confidence, 1),
            "raw_text": text,
            "explanation": explanation
        }

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return None

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    original_text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        original_text = text

        if not text:
            error = "⚠️ Vui lòng nhập nội dung tin tức."
        else:
            result = predict_text(text)
            if not result:
                error = "❌ Lỗi xử lý — vui lòng thử lại."

    return render_template("index.html", result=result, error=error, original_text=original_text)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    text = data.get('text')
    label = data.get('label')
    confidence = data.get('confidence')
    user_feedback = data.get('feedback')

    if not all([text, label, confidence, user_feedback]):
        return jsonify({"success": False, "message": "Thiếu dữ liệu"}), 400

    try:
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO feedback (text, prediction_label, confidence, user_feedback) VALUES (?, ?, ?, ?)",
            (text, label, float(confidence), user_feedback)
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Feedback error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)