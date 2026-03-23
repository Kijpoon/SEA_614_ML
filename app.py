import streamlit as st
import pickle
from pythainlp import word_tokenize  # เปลี่ยนมาใช้ตัวนี้

# 🚨 ต้องประกาศฟังก์ชันนี้ให้เหมือนตอนเทรนเป๊ะๆ
def thai_tokenize(text):
    return word_tokenize(text, engine="newmm")

st.title("🛍️ Shopping Review Analysis")

# โหลดโมเดลตัวใหม่
try:
    with open('best_model.pkl', 'rb') as f:
        tfidf, model = pickle.load(f)

    user_input = st.text_area("กรอกรีวิวสินค้า:")
    if st.button("วิเคราะห์"):
        vec_input = tfidf.transform([user_input])
        prediction = model.predict(vec_input)[0]

        if prediction == 'pos':
            st.success("ผลลัพธ์: Positive 😊")
        else:
            st.error("ผลลัพธ์: Negative 😡")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")