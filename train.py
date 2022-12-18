#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:10:21 2020

@author: af1tang
"""
import torch, os, pickle, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from load_configs import model, tokenizer, pretrain_stats, train_stats, opts, device, create_dir, p1_tok, p2_tok, start_tok, act_tok
from utils import *

## model saving ##
def checkpoint(model, tokenizer, optimizer, scheduler, stats, title="", cp=""):
    output_dir=os.path.join(opts.output_dir, cp)
    create_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(opts, os.path.join(output_dir, title+"training_opts.bin"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, title+'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, title+'scheduler.pt'))
    with open(os.path.join(output_dir, title+'stats.pkl'), 'wb') as f: pickle.dump(stats,f)
    
## Training Pipeline ##
def fit_on_batch(batch):
    xx,yy = batch
    try:
        xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
    except:
        xx, yy = to_var(xx), to_var(yy)
    # print()
    # print(f'xx: {xx}')
    # print(f'yy: {yy}')
    # print()
    ## forward on new data batch
    _outp = model(xx)
    past = _outp.past_key_values
    # print(f'past_key_values: {past}')
    # print()
    outp = model(yy, past_key_values=past, labels=yy)
    
    # backward
    loss = outp[0]; del outp
    if opts.gradient_accumulation_steps > 1:
        loss = loss / opts.gradient_accumulation_steps
    loss.backward()
    return loss

def pretrain(data, stats=None): 
    # fine tuning
    dataloader = DataLoader(data, batch_size=opts.batch_size, shuffle=True); del data
    ## optimizer and scheduler ##
    t_total = len(dataloader) // opts.gradient_accumulation_steps * opts.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [ {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                         "weight_decay": opts.weight_decay},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                     "weight_decay": 0.0} ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.lr, eps=opts.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.warmup_steps, 
                                                num_training_steps=t_total)
    # loading optimizer settings
    if (opts.model_name_or_path and os.path.isfile(os.path.join(opts.model_name_or_path, "pretrain_optimizer.pt"))
                                and os.path.isfile(os.path.join(opts.model_name_or_path, "scheduler.pt")) ):
        # load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(opts.model_name_or_path, "pretrain_optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(opts.model_name_or_path, "pretrain_scheduler.pt")))
    # track stats
    if stats is not None:
        global_step = max(stats.keys())
        epochs_trained = global_step // (len(dataloader) // opts.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(dataloader) // opts.gradient_accumulation_steps)
        print("Resuming Training ... ")
    else:
        stats = {}
        global_step, epochs_trained, steps_trained_in_current_epoch = 0,0,0
    # print(global_step)
    # print()
    tr_loss, logging_loss = 0.0, 0.0
    # very important: set model to TRAINING mode
    model.zero_grad(); model.train()
    print("Re-sizing model ... ")
    model.resize_token_embeddings(len(tokenizer))
    start_time = time.time()
    for epoch in range(epochs_trained, opts.num_train_epochs):
        data_iter= iter(dataloader)
        for step in range(len(dataloader)):
            print(f'step: {step}')
            print(f'global step: {global_step}')
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                batch = data_iter.next()
                continue
            ### step ###
            batch = data_iter.next()
            loss = fit_on_batch(batch); del batch
            # logging (new data only)
            tr_loss += loss.item()

            #print()
            #print(f'batch: {batch}')
            #print()

            # gradient accumulation
            if (step+1) % opts.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # reporting 
                if global_step % opts.logging_steps ==0:
                    stats[global_step] = {'pretrain_loss': (tr_loss - logging_loss) / opts.logging_steps, 
                                          'pretrain_lr': scheduler.get_last_lr()[-1]}
                    logging_loss = tr_loss
                    
                    elapsed_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
                    print('Epoch: %d | Iter: [%d/%d] | loss: %.3f | lr: %s | time: %s' %( 
                    epoch, global_step, t_total, stats[global_step]['pretrain_loss'],                             
                            str(stats[global_step]['pretrain_lr']), elapsed_time))
                    start_time = time.time()
                    
                if global_step % opts.save_steps==0:
                    print("Saving stuff ... ")
                    checkpoint(model, tokenizer, optimizer, scheduler, stats, title="pretrain_", cp=str(global_step))
                    plot_losses(stats, title='pretrain_loss' )
                    plot_losses(stats, title='pretrain_lr')
                    print("Done.")
                    
    return stats

def evaluate_loop(data):
    dataloader = DataLoader(data, batch_size=opts.eval_batch_size, shuffle=True); del data
    data_iter = iter(dataloader)
    with torch.no_grad():
        eval_stats, total_steps, val_loss, val_f1_score = {}, 0, 0.0, 0.0
        model.eval()
        for i in range(len(dataloader)):
            batch = data_iter.next()
            xx,yy = batch
            try:
                xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
            except:
                xx, yy = to_var(xx), to_var(yy)
            ## forward on new data batch
            _outp = model(xx)
            past = _outp.past_key_values
            outp = model(yy, past_key_values=past, labels=yy)
            loss = outp[0]
            ytrue=np.array( filter_turn_indices(to_data(yy[...,1:].contiguous().view(-1)) ) )
            ypred=np.array( filter_turn_indices(to_data( outp[1][..., :-1, :].contiguous().topk(1)[1].view(-1)) ) ) 
            min_len = min(len(ypred), len(ytrue))
            hits = [set(ypred[i]).intersection(set(ytrue[i])) for i in range(min_len)]
            prec = [len(hits[i])/len(ypred[i]) for i in range(min_len)]
            rec = [len(hits[i])/len(ytrue[i]) for i in range(min_len)]
            f1 = np.mean([2*(prec[i]*rec[i])/(prec[i] + rec[i]+1e-3) for i in range(min_len)])
            val_f1_score += f1
            val_loss += loss.mean().item()
            total_steps +=1 
            #if total_steps%100 ==0: print("... %d out of %d"%(total_steps, len(dataloader)))
            
    val_loss = val_loss / total_steps 
    val_f1_score = val_f1_score / total_steps
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    eval_stats = {'perplexity': perplexity, 'loss': val_loss, 'f1': val_f1_score}
    print("Done.")
    return eval_stats

if __name__ == '__main__':        
    with open(opts.raw_data_path, 'rb') as f: train_data = pickle.load(f)
    pretrain_stats = pretrain(train_data, pretrain_stats)
    
    # with open(opts.active_data_path, 'rb') as f: active_data = pickle.load(f)
    # train_stats = train_loop(active_data, train_data, train_stats)

    print("="*50)
    print("Evaluating ... ")
    with open(opts.val_data_path, 'rb') as f: eval_data = pickle.load(f)
    eval_stats = evaluate_loop(eval_data)
    print("Done!")
    print()
    print("Perplexity: %.2f" %eval_stats['perplexity'])
    print("F1 Score: %.2f" % eval_stats['f1'])
    print("="*50)
