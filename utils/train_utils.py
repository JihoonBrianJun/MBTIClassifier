import os
import time
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torch.distributed as dist


def train(model, train_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, config, checkpoint_dir, use_lora):
    train_perp = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = dict()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader,colour="blue", desc=f"Training Epoch{epoch}")):
            model_device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(model_device)
            target_labels = batch["target_labels"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_function = CrossEntropyLoss()
            loss = loss_function(outputs, target_labels)                
        
            loss = loss / gradient_accumulation_steps
            total_loss += loss.detach().float()
            
            loss.backward()
                
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"\n step {step} is completed and loss is {loss.detach().float()}")
        
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_perp.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        lr_scheduler.step()
        
        print("Epoch ended")
        checkpoint_start_time = time.perf_counter()
        if config.save_model:                    
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
            if use_lora:
                torch.save(model.model.classifier.original_module.state_dict(), os.path.join(checkpoint_dir, "classifier.original_module.pt"))
                # torch.save(model.model.score.original_module.state_dict(), os.path.join(checkpoint_dir, "score.original_module.pt"))                
        
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)
        
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
    
    avg_epoch_time = sum(epoch_times)/ len(epoch_times) 
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times)   
    avg_train_prep = sum(train_perp)/len(train_perp)
    avg_train_loss = sum(train_loss)/len(train_loss)    
    
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    return results