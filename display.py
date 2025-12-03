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
    if image is None:
        print("No image provided")
        return "Please upload an image first."
    
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
            'max_new_tokens': 150,     
            'pad_token_id': processor.tokenizer.pad_token_id,
            'eos_token_id': processor.tokenizer.eos_token_id,
            'use_cache': True,
        }



        try:
            print('Generating output')
            output_ids = model.generate(**model_input, **generation_config)

            print('Decoding output_ids:')
            result = processor.decode(output_ids[0], skip_special_tokens=True)

            print('Result is:',result)
            result = result.replace(prompt, "").strip()
            
            if not result or result.isspace():
                return "Could not extract attributes. Try a different image."
            
            print(result)
            return result
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error processing image: {str(e)}"

# -------------------------------
# Gradio interface
# -------------------------------
with gr.Blocks(title="Fashion VLM") as demo:
    gr.Markdown("# 👗 Fashion Clothing Extraction")
    gr.Markdown("Upload an image to extract clothing attributes!")
    
    with gr.Row():
        
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                height=500
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column():
            # Output
            output_box = gr.Textbox(
                label="Extraction Results",
                lines=10,
                interactive=False
            )
            status = gr.Markdown("Ready")
    
    
    async def analyze_image(img):
        if img is None:
            return "Please upload an image first.", "⚠️ No image"
        
        
        result = extract_attribute(img)
        
        return result, "✅ Extraction complete" 
    
    def clear():
        return None, "", "Ready"
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[output_box, status],
        concurrency_limit=1
    )
    
    clear_btn.click(
        fn=clear,
        inputs=[],
        outputs=[image_input, output_box, status]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
    )