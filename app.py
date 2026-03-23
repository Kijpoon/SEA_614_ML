import streamlit as st
import pickle
from attacut import tokenize

def thai_tokenize(text):
    return tokenize(text)

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Shopping Review Analysis", page_icon="🛍️")

# โหลดโมเดลและ TF-IDF ที่บันทึกไว้
try:
    with open('best_model_attacut.pkl', 'rb') as f:
        tfidf, model = pickle.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล! กรุณารันโค้ด Train ก่อน")

# ส่วนแสดงผลบนหน้าเว็บ
st.title("🔍 ระบบวิเคราะห์รีวิวสินค้า (Thai Sentiment)")
st.markdown("ระบุข้อความรีวิวสินค้า เพื่อให้ AI วิเคราะห์ว่าเป็นด้านบวกหรือด้านลบ")

# ช่องรับข้อมูลจากผู้ใช้
user_input = st.text_area("กรอกรีวิวที่นี่:", placeholder="เช่น สินค้าคุณภาพดีมาก ส่งไวสุดๆ")

if st.button("วิเคราะห์ผล"):
    if user_input.strip() == "":
        st.warning("กรุณากรอกข้อความก่อนกดปุ่มครับ")
    else:
        # แปลงข้อความเป็นตัวเลขและทำนายผล
        vec_input = tfidf.transform([user_input])
        prediction = model.predict(vec_input)[0]

        # แสดงผลลัพธ์
        st.subheader("ผลการวิเคราะห์:")
        if prediction == 'pos':
            st.success("✅ **POSITIVE (ชอบสินค้า)** 😊")
        else:
            st.error("❌ **NEGATIVE (ไม่ชอบสินค้า)** 😡")

st.markdown("---")
st.caption("Model: Multinomial Naive Bayes | Feature: TF-IDF (Char N-grams)")