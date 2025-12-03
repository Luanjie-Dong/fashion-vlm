from transformers import PaliGemmaForConditionalGeneration, AutoProcessor, Trainer , TrainingArguments , EarlyStoppingCallback 
import torch
import torch.nn as nn
from dataloader import load_dataset, CustomCollator
from functools import partial
from peft import get_peft_model, LoraConfig
import evaluate
from tqdm import tqdm

class FashionTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_metric = evaluate.load("bleu")

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch = self._prepare_inputs(batch)
                
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    max_new_tokens=512,
                    use_cache=True,     
                    pad_token_id=self.processing_class.tokenizer.pad_token_id,
                    eos_token_id=self.processing_class.tokenizer.eos_token_id,
                )
                
                predictions = self.processing_class.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                
                references = batch['answers']
                
                predictions = [p.strip() for p in predictions]
                references = [[r.strip()] for r in references]
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        bleu_result = self.eval_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        
        metrics = {
            f"eval_bleu": bleu_result["bleu"],
        }
        
        self.log(metrics)
        print(f"BLEU Score: {bleu_result['bleu']:.4f}")
        return metrics


def check_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    print()

    

def train_model(model_type='google/paligemma2-3b-pt-224'):
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_type,
    dtype=torch.bfloat16,  
    device_map="auto",
    attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_type) 

    train_ds, val_ds, test_ds = load_dataset()

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    

    lora_config = LoraConfig(
        r=8,  # Rank of the adaptation
        target_modules=[
            "q_proj", "o_proj", "k_proj", 
            "v_proj", "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

        
    steps_per_epoch = 36463 // (2 * 4)  
    max_steps = steps_per_epoch * 3 

    args = TrainingArguments(
        output_dir="../../models/paligemma-fashion",
        max_steps=max_steps,            
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        eval_strategy="steps",
        eval_steps=100,     
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,     

        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,

        bf16=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )


    trainer = FashionTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CustomCollator(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        processing_class=processor
    )

    trainer.train()

    model.save_pretrained("../../models/paligemma-fashion")


if __name__ == "__main__":
    check_cuda()
    train_model()