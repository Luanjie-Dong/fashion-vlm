import evaluate
from sklearn.metrics import f1_score , recall_score , precision_score
import numpy as np

def parse_attributes(attribute_string):
  
    if not attribute_string:
        return []
    
    result = []
    lines = attribute_string.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        if line.startswith('-'):
            line = line[1:].strip()
        
        category_part, attrs_part = line.split(':', 1)
        category = category_part.strip().lower()
        
        categories = [cat.strip() for cat in category.split(',')]
        
        attributes = []
        current_attr = []
        paren_depth = 0
        
        for char in attrs_part:
            if char == '(':
                paren_depth += 1
                current_attr.append(char)
            elif char == ')':
                paren_depth -= 1
                current_attr.append(char)
            elif char == ',' and paren_depth == 0:
                attr_text = ''.join(current_attr).strip()
                if attr_text:
                    attributes.append(attr_text)
                current_attr = []
            else:
                current_attr.append(char)
        
        if current_attr:
            attr_text = ''.join(current_attr).strip()
            if attr_text:
                attributes.append(attr_text)
        
        for cat in categories:
            for attr in attributes:
                if attr:  
                    attr_clean = attr.strip().lower()
                    result.append((cat, attr_clean))
    
    return result



def evaluate_attributes(predictions, actuals):
   
    pred_parsed = [parse_attributes(pred) for pred in predictions]
    actual_parsed = [parse_attributes(act) for act in actuals]
    
    all_pairs = set()
    for pairs in pred_parsed + actual_parsed:
        all_pairs.update(pairs)

    y_true = []
    y_pred = []
    
    for pred_pairs, actual_pairs in zip(pred_parsed, actual_parsed):
        pred_set = set(pred_pairs)
        actual_set = set(actual_pairs)
        
        true_vec = [1 if pair in actual_set else 0 for pair in all_pairs]
        pred_vec = [1 if pair in pred_set else 0 for pair in all_pairs]
        
        y_true.append(true_vec)
        y_pred.append(pred_vec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'recall': recall_score(y_true,y_pred, average='micro', zero_division=0),
        'precision': recall_score(y_true,y_pred, average='micro', zero_division=0)
    }
    


def evaluate_attributes_bleu(predictions,actuals):
    eval_metric = evaluate.load("bleu")

    bleu_result = eval_metric.compute(
            predictions=predictions,
            references=actuals
        )
    
    return bleu_result