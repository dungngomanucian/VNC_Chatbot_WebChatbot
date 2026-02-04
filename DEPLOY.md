# Hướng dẫn Deploy lên Hugging Face Spaces

## Bước 1: Tạo Space mới

1. Truy cập https://huggingface.co/spaces
2. Nhấn "Create new Space"
3. Điền thông tin:
   - **Name**: Tên Space của bạn (ví dụ: `llama-chatbot`)
   - **SDK**: Chọn **Gradio**
   - **Hardware**: Chọn **GPU** (khuyến nghị) hoặc **CPU**
   - **Visibility**: Public hoặc Private

## Bước 2: Upload files

Upload các file sau vào Space:

- `app.py` - File chính của ứng dụng
- `requirements.txt` - Dependencies
- `README.md` - Metadata và mô tả
- `.gitignore` - Git ignore rules

## Bước 3: Cấu hình biến môi trường (nếu cần)

Nếu model yêu cầu token:

1. Vào **Settings** của Space
2. Tìm phần **Repository secrets**
3. Thêm biến môi trường:
   - **Name**: `HF_TOKEN`
   - **Value**: Token của bạn từ https://huggingface.co/settings/tokens

Hoặc nếu muốn dùng model khác:

- **Name**: `MODEL_NAME`
- **Value**: Tên model (ví dụ: `meta-llama/Llama-3.2-3B-Instruct`)

## Bước 4: Chờ build

Space sẽ tự động build và deploy. Quá trình này có thể mất 5-10 phút lần đầu tiên.

## Lưu ý

- **GPU**: Khuyến nghị để có tốc độ tốt nhất
- **CPU**: Có thể chậm hơn nhưng vẫn hoạt động
- **Memory**: Đảm bảo Space có đủ RAM cho model (tối thiểu 4GB cho model 1B)
- **Timeout**: Nếu model quá lớn, có thể cần upgrade Space lên tier cao hơn

## Troubleshooting

### Lỗi "Out of memory"
- Giảm kích thước model
- Upgrade hardware tier
- Sử dụng quantization (đã được tích hợp sẵn)

### Lỗi "Model not found"
- Kiểm tra token đã được set chưa
- Kiểm tra tên model trong biến môi trường

### Build chậm
- Lần đầu build sẽ chậm vì phải tải model
- Các lần sau sẽ nhanh hơn nhờ cache

