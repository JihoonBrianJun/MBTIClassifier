from tqdm import tqdm
import numpy as np
import torch


def test(model, eval_dataloader):
    class_names = np.array(["intj", "intp", "infj", "infp",
                            "istj", "istp", "isfj", "isfp",
                            "entj", "entp", "enfj", "enfp",
                            "estj", "estp", "esfj", "esfp"])

    results = []
    correct, correct_match = 0,0
    correct_each = [0, 0, 0, 0]
    match_scores = [0, 0, 0]
    hits = [0, 0, 0]
    
    model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="Testing Epoch")):
        with torch.no_grad():
            model_device = next(model.parameters()).device
            model_output = model(input_ids=batch["input_ids"].to(model_device)).cpu()
            model_answer_idx = torch.argmax(model_output, dim=1).numpy()
            model_top_answers_idx = torch.argsort(model_output, dim=1, descending=True).numpy()
            
            model_answer = class_names[model_answer_idx[0]]
            model_top_answers = class_names[model_top_answers_idx[0]]
            
            gold_answer = batch["gold_answers"][0]
            
            if model_answer == gold_answer:
                correct += 1
            if model_answer is not None and len(model_answer) == len(gold_answer):
                for i in range(len(gold_answer)):
                    correct_each[i] += (model_answer[i] == gold_answer[i])
                match_score = sum([model_answer[i] == gold_answer[i] for i in range(len(gold_answer))])
                if match_score < len(gold_answer) and match_score > 0:
                    match_scores[match_score-1] += 1
                correct_match += match_score / len(gold_answer)
            
            
            for idx, i in enumerate([3,5,10]):
                if gold_answer in model_top_answers[:i]:
                    hits[idx] += 1
            
            print(f"Model Answer: {model_answer}, Gold Answer: {gold_answer}")
            
            results.append(model_answer)
            
    
    print(f"Correct: {correct} out of {step+1}\nCorrect Rate: {correct / (step+1) * 100}%")
    for i in range(len(gold_answer)-1):
        print(f"{i+1} Wrong Rate: {match_scores[i] / (step+1) * 100}%")

    print(f"Correct Match Rate: {correct_match / (step+1) * 100}%")
    
    for i in range(len(gold_answer)):
        print(f"Answer {i}th idx Correct Rate: {correct_each[i] / (step+1) * 100}%")

    for idx, i in enumerate([3,5,10]):
        print(f"Hit@{i}: {hits[idx] / (step+1) * 100}%")
    
    return results