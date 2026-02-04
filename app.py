import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
from config import MODEL_NAME_HF, HF_TOKEN

warnings.filterwarnings("ignore")

try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

MODEL_NAME = MODEL_NAME_HF
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZATION = DEVICE == "cuda" and HAS_BITSANDBYTES
USE_COMPILE = DEVICE == "cuda" and hasattr(torch, 'compile')

model = None
tokenizer = None
pipe = None

def load_model():
    global model, tokenizer, pipe
    
    if model is None or tokenizer is None:
        print(f"Đang tải model {MODEL_NAME}...")
        print(f"Sử dụng device: {DEVICE}")
        
        try:
            tokenizer_kwargs = {"trust_remote_code": True}
            if HF_TOKEN:
                tokenizer_kwargs["token"] = HF_TOKEN
                print("Đang sử dụng Hugging Face token từ file .env")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            if USE_QUANTIZATION:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    print("✅ Sử dụng 8-bit quantization để tăng tốc độ")
                except Exception as e:
                    print(f"⚠️ Không thể sử dụng quantization: {e}")
                    model_kwargs["dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            else:
                if DEVICE == "cuda":
                    model_kwargs["dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["dtype"] = torch.float32
                    model_kwargs["device_map"] = None
            
            if HF_TOKEN:
                model_kwargs["token"] = HF_TOKEN
            
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
            
            if DEVICE == "cpu":
                model = model.to(DEVICE)
            
            model.eval()
            
            if USE_COMPILE and not USE_QUANTIZATION:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    print("✅ Đã compile model để tối ưu tốc độ")
                except Exception as e:
                    print(f"⚠️ Không thể compile model: {e}")
            
            if hasattr(model, 'generation_config'):
                model.generation_config.max_length = None
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if DEVICE == "cuda" else -1
            )
            
            print("Model đã được tải thành công!")
            return "Model đã được tải thành công!"
        except Exception as e:
            error_msg = f"Lỗi khi tải model: {str(e)}"
            print(error_msg)
            return error_msg
    
    return "Model đã được tải sẵn!"

def generate_response(message, history):
    global model, tokenizer, pipe
    
    if model is None or tokenizer is None:
        return "Vui lòng load model trước!"
    
    try:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": message}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"Question: {message}\nAnswer: "
        
        with torch.inference_mode():
            outputs = pipe(
                prompt,
                max_new_tokens=64,
                max_length=None,
                temperature=0.2,
                top_p=0.95,
                do_sample=False,
                repetition_penalty=1.1,
                return_full_text=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = outputs[0]['generated_text'].strip()
        
        stop_phrases = [
            "\n\n\n",
            "\nQuestion:",
            "\nUser:",
            "<|endoftext|>",
            "\n\nAnswer:",
            "\n\nQuestion:"
        ]
        
        for stop_phrase in stop_phrases:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0]
                break
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            cleaned_response = lines[0]
            if len(cleaned_response) < 10 and len(lines) > 1:
                cleaned_response = lines[0] + " " + lines[1]
        else:
            cleaned_response = response
        
        sentences = cleaned_response.split('.')
        if len(sentences) > 3:
            cleaned_response = '.'.join(sentences[:3]) + '.'
        elif len(sentences) > 1 and len('.'.join(sentences[:2])) < 200:
            cleaned_response = '.'.join(sentences[:2]) + '.'
        
        if len(cleaned_response) > 300:
            cleaned_response = cleaned_response[:300].rsplit('.', 1)[0] + '.'
        
        return cleaned_response.strip()
        
    except Exception as e:
        return f"Lỗi khi sinh câu trả lời: {str(e)}"

def chat_interface(message, history):
    if not message:
        return history, ""
    
    if history is None:
        history = []

    response = generate_response(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, ""

with gr.Blocks(title="LLAMA Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 LLAMA Chatbot
        Ứng dụng chatbot sử dụng mô hình LLAMA để trả lời câu hỏi của bạn.
        
        **Hướng dẫn sử dụng:**
        1. Nhấn nút "Load Model" để tải mô hình LLAMA (chỉ cần làm một lần)
        2. Nhập câu hỏi của bạn vào ô chat
        3. Nhấn Enter hoặc nút "Gửi" để nhận câu trả lời
        """
    )
    
    with gr.Row():
        load_btn = gr.Button("Load Model", variant="primary", size="lg")
        status = gr.Textbox(label="Trạng thái", interactive=False)
    
    load_btn.click(fn=load_model, outputs=status)
    
    chatbot = gr.Chatbot(label="Chat", height=500)
    
    with gr.Row():
        msg = gr.Textbox(
            label="Câu hỏi của bạn",
            placeholder="Nhập câu hỏi của bạn ở đây...",
            scale=4,
            show_label=False
        )
        submit_btn = gr.Button("Gửi", variant="primary", scale=1)
    
    msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
    submit_btn.click(chat_interface, [msg, chatbot], [chatbot, msg])
    
    gr.Markdown(
        """
        ---
        **Lưu ý:** 
        - Lần đầu tiên load model có thể mất vài phút để tải về
        - Model sẽ được cache và không cần tải lại lần sau
        - Đảm bảo bạn có đủ RAM/VRAM để chạy model
        """
    )

if __name__ == "__main__":
    print("Đang khởi động ứng dụng...")
    print(f"Device: {DEVICE}")
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )
