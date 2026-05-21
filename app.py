import os
import re
import csv
import io
import sqlite3
from datetime import datetime

from flask import Flask, request, render_template, jsonify, Response
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from underthesea import word_tokenize

# Phase 1: URL crawling + file reading
import requests as req_lib
from bs4 import BeautifulSoup

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
            is_bookmarked INTEGER DEFAULT 0,
            source_type TEXT DEFAULT 'text',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migration: thêm cột is_bookmarked nếu chưa có (cho DB cũ)
    try:
        c.execute("ALTER TABLE feedback ADD COLUMN is_bookmarked INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE feedback ADD COLUMN source_type TEXT DEFAULT 'text'")
    except Exception:
        pass
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
        chunk_size = MAX_TOKENS - 2
        chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]

        chunk_probs = []
        for chunk in chunks:
            input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
            inputs = {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([[1] * len(input_ids)])
            }
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                chunk_probs.append(probs[0])

        # Trung bình xác suất của tất cả các chunks
        avg_probs = torch.stack(chunk_probs).mean(dim=0)

        pred = torch.argmax(avg_probs).item()
        confidence = avg_probs[pred].item() * 100

        label = "Tin thật" if pred == 1 else "Tin giả"
        is_fake = pred == 0

        # --- Giải thích bằng LIME (Explainable AI) ---
        text_for_lime = word_tokenize(text[:1000], format="text")
        exp = explainer.explain_instance(text_for_lime, lime_predictor, num_features=6, num_samples=50)

        explanation = []
        for word, weight in exp.as_list():
            explanation.append({
                "word": word.replace('_', ' '),
                "weight": round(abs(weight), 4),
                "type": "real" if weight > 0 else "fake",
                "raw_weight": round(weight, 4)
            })

        return {
            "label": label,
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "raw_text": text,
            "explanation": explanation
        }

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return None

# ---------------------------------------------------------
# Phase 1.3 — Nhập liệu đa dạng
# ---------------------------------------------------------
def extract_article_from_url(url):
    """Crawl URL bài báo, trả về text nội dung chính"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        res = req_lib.get(url, timeout=10, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        # Xóa nav, header, footer, script, style, ads
        for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                          'aside', 'figure', 'form', 'iframe', 'noscript']):
            tag.decompose()

        # Rút trích tất cả thẻ <p> thay vì cố tìm thẻ chứa (dễ bị match nhầm div nhỏ)
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)

        if len(text) < 100:
            text = soup.get_text(separator=' ', strip=True)

        return text[:8000].strip()  # Giới hạn 8000 ký tự
    except Exception as e:
        raise ValueError(f"Không thể crawl URL: {str(e)}")

def extract_text_from_file(file):
    """Đọc nội dung từ file .txt hoặc .docx"""
    filename = file.filename.lower()
    if filename.endswith('.txt'):
        raw = file.read()
        try:
            return raw.decode('utf-8').strip()
        except UnicodeDecodeError:
            return raw.decode('cp1252', errors='replace').strip()
    elif filename.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(file)
            return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise ValueError(f"Không thể đọc file .docx: {str(e)}")
    else:
        raise ValueError("Chỉ hỗ trợ file .txt hoặc .docx")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def index():
    """Dashboard tổng quan"""
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback WHERE prediction_label LIKE '%Tin giả%'")
    total_fake = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback WHERE prediction_label LIKE '%Tin thật%'")
    total_real = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback WHERE user_feedback='Đúng'")
    correct = c.fetchone()[0]
    # 8 bản ghi gần nhất
    c.execute("SELECT id, text, prediction_label, confidence, timestamp, is_bookmarked FROM feedback ORDER BY id DESC LIMIT 8")
    recent = c.fetchall()
    conn.close()

    acc = round(correct / total * 100, 1) if total > 0 else 0
    return render_template("index.html",
        total=total, total_fake=total_fake, total_real=total_real,
        accuracy=acc, recent=recent)

@app.route("/about")
def about():
    return render_template("about.html")

# ---------------------------------------------------------
# PREDICT ROUTE — hỗ trợ text, URL, file
# ---------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    original_text = ""
    source_type = "text"

    if request.method == "POST":
        source_type = request.form.get("source_type", "text")

        if source_type == "url":
            url = request.form.get("url", "").strip()
            if not url:
                error = "⚠️ Vui lòng nhập URL bài báo."
            else:
                try:
                    original_text = extract_article_from_url(url)
                    if not original_text:
                        error = "❌ Không trích xuất được nội dung từ URL này."
                    else:
                        result = predict_text(original_text)
                        if not result:
                            error = "❌ Lỗi xử lý — vui lòng thử lại."
                except ValueError as ve:
                    error = f"❌ {str(ve)}"

        elif source_type == "file":
            uploaded = request.files.get("article_file")
            if not uploaded or uploaded.filename == "":
                error = "⚠️ Vui lòng chọn file."
            else:
                try:
                    original_text = extract_text_from_file(uploaded)
                    if not original_text:
                        error = "❌ File rỗng hoặc không đọc được nội dung."
                    else:
                        result = predict_text(original_text)
                        if not result:
                            error = "❌ Lỗi xử lý — vui lòng thử lại."
                except ValueError as ve:
                    error = f"❌ {str(ve)}"

        else:  # text
            text = request.form.get("text", "").strip()
            original_text = text
            if not text:
                error = "⚠️ Vui lòng nhập nội dung tin tức."
            else:
                result = predict_text(text)
                if not result:
                    error = "❌ Lỗi xử lý — vui lòng thử lại."

        # Lưu vào DB nếu có kết quả
        if result:
            try:
                conn = sqlite3.connect('feedback.db')
                c = conn.cursor()
                c.execute(
                    "INSERT INTO feedback (text, prediction_label, confidence, source_type) VALUES (?, ?, ?, ?)",
                    (result['raw_text'][:2000], result['label'], result['confidence'], source_type)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                app.logger.error(f"DB insert error: {e}")

    return render_template("predict.html",
        result=result, error=error,
        original_text=original_text, source_type=source_type)

# ---------------------------------------------------------
# FEEDBACK ROUTE
# ---------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    text = data.get('text')
    label = data.get('label')
    confidence = data.get('confidence')
    user_feedback = data.get('feedback')

    if not all([text, label, confidence is not None, user_feedback]):
        return jsonify({"success": False, "message": "Thiếu dữ liệu"}), 400

    try:
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        # Tìm bản ghi gần nhất trùng text + label để cập nhật
        c.execute(
            "SELECT id FROM feedback WHERE text=? AND prediction_label=? ORDER BY id DESC LIMIT 1",
            (text[:2000], label)
        )
        row = c.fetchone()
        if row:
            c.execute("UPDATE feedback SET user_feedback=? WHERE id=?", (user_feedback, row[0]))
        else:
            c.execute(
                "INSERT INTO feedback (text, prediction_label, confidence, user_feedback) VALUES (?, ?, ?, ?)",
                (text[:2000], label, float(confidence), user_feedback)
            )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Feedback error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------
# Phase 1.2 — HISTORY & BOOKMARK
# ---------------------------------------------------------
@app.route("/history")
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    filter_type = request.args.get('filter', 'all')
    search = request.args.get('q', '')

    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()

    where_clauses = []
    params = []

    if filter_type == 'fake':
        where_clauses.append("prediction_label LIKE '%Tin giả%'")
    elif filter_type == 'real':
        where_clauses.append("prediction_label LIKE '%Tin thật%'")
    elif filter_type == 'bookmarked':
        where_clauses.append("is_bookmarked = 1")

    if search:
        where_clauses.append("text LIKE ?")
        params.append(f'%{search}%')

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    c.execute(f"SELECT COUNT(*) FROM feedback {where_sql}", params)
    total_count = c.fetchone()[0]

    offset = (page - 1) * per_page
    c.execute(
        f"SELECT id, text, prediction_label, confidence, user_feedback, is_bookmarked, source_type, timestamp FROM feedback {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    rows = c.fetchall()
    conn.close()

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    return render_template("history.html",
        rows=rows, page=page, total_pages=total_pages,
        total_count=total_count, filter_type=filter_type,
        search=search, per_page=per_page)

@app.route("/bookmark", methods=["POST"])
def bookmark():
    data = request.json
    record_id = data.get('id')
    if not record_id:
        return jsonify({"success": False, "message": "Thiếu ID"}), 400
    try:
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute("SELECT is_bookmarked FROM feedback WHERE id=?", (record_id,))
        row = c.fetchone()
        if not row:
            return jsonify({"success": False, "message": "Không tìm thấy"}), 404
        new_val = 1 - row[0]
        c.execute("UPDATE feedback SET is_bookmarked=? WHERE id=?", (new_val, record_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "is_bookmarked": bool(new_val)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/history/export")
def history_export():
    filter_type = request.args.get('filter', 'all')
    search = request.args.get('q', '')

    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()

    where_clauses = []
    params = []
    if filter_type == 'fake':
        where_clauses.append("prediction_label LIKE '%Tin giả%'")
    elif filter_type == 'real':
        where_clauses.append("prediction_label LIKE '%Tin thật%'")
    elif filter_type == 'bookmarked':
        where_clauses.append("is_bookmarked = 1")
    if search:
        where_clauses.append("text LIKE ?")
        params.append(f'%{search}%')

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    c.execute(f"SELECT id, text, prediction_label, confidence, user_feedback, is_bookmarked, source_type, timestamp FROM feedback {where_sql} ORDER BY id DESC", params)
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Nội dung', 'Nhãn dự đoán', 'Confidence (%)', 'Feedback', 'Bookmark', 'Nguồn', 'Thời gian'])
    for row in rows:
        writer.writerow([
            row[0],
            row[1][:200] if row[1] else '',
            row[2], row[3],
            row[4] or '',
            'Có' if row[5] else 'Không',
            row[6] or 'text',
            row[7]
        ])

    output.seek(0)
    filename = f"vietguard_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue().encode('utf-8-sig'),  # utf-8-sig cho Excel đọc được tiếng Việt
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------
@app.route("/api/recent")
def api_recent():
    """JSON endpoint: 4 bản ghi gần nhất cho sidebar"""
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("SELECT id, text, prediction_label, confidence FROM feedback ORDER BY id DESC LIMIT 4")
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        is_fake = 'Tin giả' in (row[2] or '')
        result.append({
            'id': row[0],
            'text': (row[1] or '')[:80],
            'label': 'fake' if is_fake else 'real',
            'confidence': row[3]
        })
    return jsonify(result)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)