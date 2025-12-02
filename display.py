import gradio as gr
from PIL import Image
import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------------------
# Load the LoRA-finetuned model
# -------------------------------
BASE_MODEL = "google/paligemma2-3b-pt-224"
LORA_PATH = "models/paligemma-fashion" 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16
)
# model = PeftModel.from_pretrained(model, LORA_PATH)
# model.to(device)
# model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL)

# -------------------------------
# Gradio prediction function
# -------------------------------
def generate_caption(image: Image, prompt: str = "Describe all garments and their attributes in this image.") -> str:
    """
    Takes an image and optional prompt, returns generated caption.
    """
    model_input = processor(
        images=[image],
        text=[f"<image>{prompt}\n"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(device)

    output_ids = model.generate(**model_input, max_length=256)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption

# -------------------------------
# Gradio interface
# -------------------------------
iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload Fashion Image"),
        gr.Textbox(lines=2, placeholder="Prompt (optional)", label="Prompt")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Fashion VLM (PaliGemma + LoRA)",
    description="Upload an image of fashion items and get detailed captions describing garments and attributes."
)

iface.launch()
