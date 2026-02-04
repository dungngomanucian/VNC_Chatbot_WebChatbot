---
title: LLAMA Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 LLAMA Chatbot

Ứng dụng chatbot sử dụng mô hình LLAMA để trả lời câu hỏi của người dùng.

## Tính năng

- ✅ Sử dụng mô hình LLAMA 3.2 1B Instruct
- ✅ Giao diện chat thân thiện với Gradio
- ✅ Tự động tải model khi khởi động
- ✅ Tối ưu tốc độ với quantization và compilation
- ✅ Hỗ trợ GPU và CPU

## Cách sử dụng

1. Nhập câu hỏi của bạn vào ô chat
2. Nhấn Enter hoặc nút "Gửi" để nhận câu trả lời

## Cấu hình

Để thay đổi model, set biến môi trường `MODEL_NAME` trong Settings của Space.

Mặc định: `meta-llama/Llama-3.2-1B-Instruct`

## Yêu cầu

- Python 3.8+
- GPU được khuyến nghị để có tốc độ tốt nhất
- Hugging Face token (nếu model yêu cầu)

## License

MIT
