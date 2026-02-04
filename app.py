import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
import os

warnings.filterwarnings("ignore")

try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZATION = DEVICE == "cuda" and HAS_BITSANDBYTES
USE_COMPILE = DEVICE == "cuda" and hasattr(torch, 'compile')

model = None
tokenizer = None
pipe = None

def load_model():
    global model, tokenizer, pipe
    
    if model is not None and tokenizer is not None:
        return
    
    print(f"Đang tải model {MODEL_NAME}...")
    print(f"Sử dụng device: {DEVICE}")
    
    try:
        tokenizer_kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            tokenizer_kwargs["token"] = HF_TOKEN
        
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
                print("✅ Sử dụng 8-bit quantization")
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
                print("✅ Đã compile model")
            except Exception as e:
                print(f"⚠️ Không thể compile model: {e}")
        
        if hasattr(model, 'generation_config'):
            model.generation_config.max_length = None
            model.generation_config.max_new_tokens = 200
            model.generation_config.temperature = 0.5
            model.generation_config.top_p = 0.95
            model.generation_config.repetition_penalty = 1.1
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            batch_size=1,
            model_kwargs={"use_cache": True}
        )
        
        print("✅ Model đã được tải thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {str(e)}")
        raise

def generate_response(message, history):
    global model, tokenizer, pipe
    
    if model is None or tokenizer is None:
        return "Đang tải model, vui lòng đợi..."
    
    try:
        if tokenizer.chat_template is not None:
            messages = []
            
            if history and len(history) > 0:
                for user_msg, assistant_msg in history[-2:]:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            messages.append({"role": "user", "content": message})
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            context = ""
            if history and len(history) > 0:
                user_msg, assistant_msg = history[-1]
                context = f"Q: {user_msg}\nA: {assistant_msg}\n\n"
            prompt = f"{context}Question: {message}\nAnswer: "
        
        with torch.inference_mode():
            outputs = pipe(
                prompt,
                max_new_tokens=200,
                max_length=None,
                temperature=0.5,
                top_p=0.95,
                do_sample=False,
                repetition_penalty=1.1,
                return_full_text=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = outputs[0]['generated_text'].strip()
        
        stop_phrases = [
            "\n\n\n",
            "\nQuestion:",
            "\nQ:",
            "\nUser:",
            "<|endoftext|>",
            "\n\nAnswer:",
            "\n\nA:",
            "\n\nQuestion:",
            tokenizer.eos_token if tokenizer.eos_token else None
        ]
        
        stop_phrases = [s for s in stop_phrases if s]
        
        for stop_phrase in stop_phrases:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0]
                break
        
        response = response.strip()
        
        if not response:
            return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
        
        if len(response) < 5:
            return response
        
        if response.endswith(('.', '!', '?')):
            return response
        
        sentences = response.split('.')
        if len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            if len(last_sentence) < 10:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                response = '.'.join(sentences) + '.'
        elif not response.endswith('.'):
            response += '.'
        
        return response.strip()
        
    except Exception as e:
        return f"Lỗi khi sinh câu trả lời: {str(e)}"

def chat_interface(message, history):
    if not message:
        return history, ""
    
    if history is None:
        history = []

    response = generate_response(message, history)

    history.append((message, response))

    return history, ""

load_model()

with gr.Blocks(title="LLAMA Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 LLAMA Chatbot
        Ứng dụng chatbot sử dụng mô hình LLAMA để trả lời câu hỏi của bạn.
        
        **Hướng dẫn sử dụng:**
        - Nhập câu hỏi của bạn vào ô chat
        - Nhấn Enter hoặc nút "Gửi" để nhận câu trả lời
        """
    )
    
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
        - Model đã được tải sẵn và sẵn sàng sử dụng
        - Câu trả lời được tối ưu để ngắn gọn và chính xác
        """
    )

if __name__ == "__main__":
    demo.launch()
