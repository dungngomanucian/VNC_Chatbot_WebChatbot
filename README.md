# Yêu cầu hệ thống
- Python 3.8 trở lên
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- GPU: Tùy chọn (có GPU sẽ nhanh hơn)
- Disk: Ít nhất 10GB trống để tải model

# Tạo và kích hoạt môi trường ảo
# Windows
py -3.11 -m venv venv
venv\Scripts\activate
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Cấu hình Model
3. Tạo file `.env` trong thư mục gốc của project và thêm token:
HUGGINGFACE_TOKEN=your_token_here

# Chạy ứng dụng:
1. python app.py

3. Nhấn nút "Load Model" để tải model (chỉ cần làm một lần)

4. Nhập câu hỏi và nhấn Enter hoặc nút "Gửi"

## ⚙️ Tùy chỉnh

Có thể chỉnh sửa các tham số trong `config.py`:

- `MAX_NEW_TOKENS`: Số token tối đa cho câu trả lời
- `TEMPERATURE`: Độ sáng tạo (0.0-1.0)
- `TOP_P`: Nucleus sampling
- `REPETITION_PENALTY`: Hệ số phạt lặp lại