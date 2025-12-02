import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataloader import *
from peft import PeftModel
from functools import partial
from torch.utils.data import DataLoader




def evaluate(model, processor, test_dataset, device="cuda"):
    model.eval()
    model.to(device)

    smoothie = SmoothingFunction().method4
    bleu_scores = []

    PROMPT = "<image> Describe all garments and their attributes in this image."
    test_dataloader = DataLoader(test_dataset,collate_fn=partial(eval_collate_fn, processor=processor),batch_size=2,shuffle=True)
    print(f"Starting evaluation on {len(test_dataloader)} batches... \n")

    for i , (batch,answers) in enumerate(test_dataloader):
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items() if k != 'answer'}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,  
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            input_len = inputs["input_ids"].shape[1]
            generated_texts = processor.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            for pred , true in zip(generated_texts,answers):
                pred_tokens = pred.strip().split()
                true_tokens = [true.strip().split()]  
                if len(pred_tokens) == 0:
                    bleu = 0.0
                else:
                    bleu = sentence_bleu(true_tokens, pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)

                print(f"Predicted: {pred}")
                print(f"Answer:    {true}")
                print(f"BLEU:      {bleu:.4f}\n")
        
        if i == 2:
            break




    
   





if __name__ == "__main__":
    MODEL_BASE = "google/paligemma2-3b-pt-224" 
    LORA_PATH = "../../models/paligemma-fashion"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, LORA_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_BASE)
    _ , _ , test_ds = load_dataset()

    evaluate(model, processor, test_ds, device)