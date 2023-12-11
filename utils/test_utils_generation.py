from tqdm import tqdm
import torch
import torch.distributed as dist


def test(model, config, eval_dataloader, tokenizer, local_rank, world_size):
    results = []
    correct = 0
    naive_correct = 0
    correct_each, naive_correct_each = 0, 0
    
    model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="Testing Epoch")):
        with torch.no_grad():
            if type(model).__name__ == "DistributedDataParallel":
                outputs = model.module.generate(
                    input_ids=batch["input_ids"].to(local_rank),
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    max_length=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )
            else:
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(local_rank),
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    max_length=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )                  

            result = tokenizer.decode(outputs[0].view(-1))
            naive_answer = tokenizer.decode(torch.argmax(model(input_ids=batch["input_ids"].to(local_rank)).logits[0][-1]))
            

            if "[MBTI]" in result and "<|endoftext|>" in result:
                model_answer = result.split("[MBTI]")[-1].strip().split("<|endoftext|>")[0]
            else:
                model_answer = None

            gold_answer = batch["gold_answers"][0]
            if model_answer == gold_answer:
                correct += 1
            if model_answer is not None and len(model_answer) == len(gold_answer):
                correct_each += sum([model_answer[i] == gold_answer[i] for i in range(len(gold_answer))]) / len(gold_answer)
                
            if naive_answer == gold_answer:
                naive_correct += 1
            if naive_answer is not None and len(naive_answer) == len(gold_answer):
                naive_correct_each += sum([naive_answer[i] == gold_answer[i] for i in range(len(gold_answer))]) / len(gold_answer)
            
            print(f"Model Answer: {model_answer}, Naive Answer: {naive_answer}, Gold Answer: {gold_answer}")
            print(f"Model Answer Each: {model_answer}, Naive Answer: {naive_answer}, Gold Answer: {gold_answer}")
            
            # print(result + f"\nGold Answer: {gold_answer}")
            results.append(result + f"\nGold Answer: {gold_answer}")
    
    print(f"Correct: {correct} out of {step+1}\nCorrect Rate: {correct / (step+1) * 100}%")
    print(f"Naive Correct: {naive_correct} out of {step+1}\nNaive Correct Rate: {naive_correct / (step+1) * 100}%")
    
    print(f"Correct Each Rate: {correct_each / (step+1) * 100}%")
    print(f"Naive Correct Each Rate: {naive_correct_each / (step+1) * 100}%")
    
    if world_size>1:
        gathered_results = [list() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        for i in range(1, world_size):
            gathered_results[0].extend(gathered_results[i])
        return gathered_results[0]
    else:
        return results