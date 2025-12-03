import gradio as gr
from PIL import Image
import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from peft import PeftModel

# -------------------------------
# Load the LoRA-finetuned model
# -------------------------------
MODEL_BASE = "google/paligemma2-3b-pt-224"
LORA_PATH = "models/paligemma-fashion" 
model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_BASE,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

model = PeftModel.from_pretrained(model, LORA_PATH)
processor = AutoProcessor.from_pretrained(MODEL_BASE,use_fast=True)



def extract_attribute(image, prompt="Describe all garments and their attributes in this image."):
    model.eval()
    
    print('Extracting attribute of fashion image!')
    with torch.no_grad():
        formatted_prompt = f"<image>{prompt}"
        
        model_input = processor(
            images=[image],
            text=[formatted_prompt], 
            return_tensors="pt"
        ).to(model.device)
        
        generation_config = {
            'max_new_tokens': 50,     
            'num_beams': 1,             
            'do_sample': False,         
            'pad_token_id': processor.tokenizer.pad_token_id,
            'eos_token_id': processor.tokenizer.eos_token_id,
            'use_cache': True,
            'output_scores': False,     
        }



        output_ids = model.generate(**model_input, **generation_config)
        result = processor.decode(output_ids[0], skip_special_tokens=True)
        result = result.replace(prompt, "").strip()
        
        return result

# -------------------------------
# Gradio interface
# -------------------------------
with gr.Blocks(title="Fashion VLM - Clothing Analyzer") as demo:
    gr.Markdown("# 👗 Fashion VLM - Clothing Analyzer")
    gr.Markdown("Upload a fashion image to get detailed analysis of fashion item attributes!")

    with gr.Row():
        input_image = gr.Image(
            type="pil", 
            label="📷 Upload Fashion Image",
            scale=1
        )
        
        output_textbox = gr.Textbox(
            label="🧥 Fashion Analysis",
            scale=1,
            lines=10 
        )

    # Attach the function to the components
    input_image.change(
        fn=extract_attribute,
        inputs=input_image,
        outputs=output_textbox
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
    )