import argparse
import logging
import os
from pathlib import Path
import json

import spu.utils.distributed as ppd
import jax.numpy as jnp
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    FlaxGPT2LMHeadModel, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer)

from dataset import EvalDataset
LOG_DIR = Path("./logs")
LOGGER = logging.getLogger(__name__)


def get_last_comm():
    #total send bytes 2261740442,
    comm = None
    with open(LOG_DIR/"server.log", "r") as f:
        lines = f.readlines()
    
    for i in reversed(range(len(lines))):
        key = "total send bytes "
        pos1  = lines[i].find(key)
        if pos1 != -1:
            pos2 = lines[i].find(",", pos1)
            comm = int(lines[i][pos1+len(key):pos2-1])
            break
    
    return comm


def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens


def run_on_plaintext(model, inputs, use_gpu, params = None, vocab_length = None):
    if use_gpu:
        inputs = inputs.to("cuda")
        with torch.no_grad():
            logits = model(inputs).logits[:,:,:vocab_length] 
        return logits.argmax(-1).cpu(), logits.cpu()
    else:      
        logits = model(input_ids=inputs, params=params).logits[:,:,:vocab_length]      
        return jnp.argmax(logits, axis=-1), logits

# # global MODEL
# MODEL = None
def run_on_ciphertext(model, seq_ids, spu_model_params, vocab_length = None):
    def text_generation(seq_ids, model_params):
        logits = MODEL(input_ids=seq_ids, params=model_params).logits    
        return logits
    # encode context the generation is conditioned
    input_ids = ppd.device("P1")(lambda x: x)(seq_ids)
    # params = ppd.device("P2")(lambda x: x)(spu_model_params)
    outputs = ppd.device("SPU")(
        text_generation,
    )(input_ids, spu_model_params)

    logits = ppd.get(outputs)
    logits = logits[:,:,:vocab_length]
    preds = jnp.argmax(logits, axis=-1)
    return preds, logits


def process_prediction(pred_ids, inputs, tokenizer):
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")
    all_pred = []
    all_gt = []
    prev_pred = None
    for pred, gt in zip(pred_ids, inputs):
        pred = pred.tolist()
        gt = gt.tolist()

        for i, y in enumerate(gt):
            if i == 0:
                if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    now_gt = [y]
                    now_pred = [0] if prev_pred is None else [prev_pred]
                    all_pred.append(DecodeIds(now_pred).strip().split()[0])
                    all_gt.append(DecodeIds(now_gt).strip())
                    now_gt = []  
                    now_pred = []
                else:
                    now_gt = [y]
                    now_pred = [0] if prev_pred is None else [prev_pred]
            else:
                if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                    if len(now_gt) > 0:
                        try:
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append("<SPACE>")
                        all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = []
                        now_pred = []
                if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT"):
                    if len(now_gt) > 0:
                        try:
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append("<SPACE>")
                        all_gt.append(DecodeIds(now_gt).strip())
                    now_gt = [y]
                    now_pred = [pred[i-1]]
                    try:
                        all_pred.append(DecodeIds(now_pred).strip().split()[0])
                    except IndexError:
                        all_pred.append("<SPACE>")
                    all_gt.append(DecodeIds(now_gt).strip())
                    now_gt = []
                    now_pred = []
                    continue
                now_gt.append(y)
                now_pred.append(pred[i-1])

    assert len(all_pred) == len(all_pred)
    return all_pred, all_gt


def eval_acc(args, model, tokenizer, file_type="test", model_params = None, start_line = -1, end_line = -1):
    use_gpu = args.mode == "gpu"

    eval_dataset = EvalDataset(tokenizer, args, LOGGER, file_type=file_type, block_size=args.block_size, start_line = start_line, end_line = end_line)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.batch_size,
        collate_fn=(lambda batch: torch.tensor(batch)) if use_gpu else (lambda batch: jnp.array(batch))
    )
    if use_gpu:
        model.eval()
        model.to("cuda")
        
    #output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.mode}_{start_line}_{end_line}.jsonl" 

    if args.mode in ["cpu","gpu"]:   
        for step, batch in tqdm(enumerate(eval_dataloader)):
            correct = 0.0
            total = 0
            t1 = time.time()   
            pred_ids, logits = run_on_plaintext(model, batch, use_gpu, model_params, vocab_length=len(tokenizer))
            duration = time.time() - t1   
            pred, gt = process_prediction(pred_ids, batch, tokenizer)

            for x, y in zip(pred, gt):
                if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                    total += 1
                    if x == y:
                        correct += 1
            
            if step % args.logging_steps == 0:
                LOGGER.info(f"{step} are done!")

            with open(output_file, "a+") as f:
                info = {"index": step, "duration": duration, "correct": correct, "total": total}   
                f.write(json.dumps(info) + "\n")      

    elif args.mode  == "spu":
        spu_model_params = ppd.device("SPU")._place_arguments(
        ppd.device("P2")(lambda x: x)(model_params)
        )[0][0]

        for step, batch in tqdm(enumerate(eval_dataloader)):
            correct = 0.0
            total = 0
            t1 = time.time()   
            pred_ids, logits = run_on_ciphertext(model, batch, spu_model_params, vocab_length=len(tokenizer))
            duration = time.time() - t1   
            pred, gt = process_prediction(pred_ids, batch, tokenizer)

            for x, y in zip(pred, gt):
                if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                    total += 1
                    if x == y:
                        correct += 1
            
            if step % args.logging_steps == 0:
                LOGGER.info(f"{step} are done!")

            #save result       
            comm = None
            # comm = get_last_comm()
            # assert comm is not None 

            with open(output_file, "a+") as f:
                info = {"index": step, "duration": duration, "comm": comm, "correct": correct, "total": total}   
                f.write(json.dumps(info) + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument("-c", "--config", default="./3pc.json")
    # parser.add_argument("-r", "--resume", default=0, type=int)     
    parser.add_argument("-m", "--mode", type=str, default="cpu", choices=["cpu","spu","gpu"])
    parser.add_argument("-i", "--data_dir", type=str, default=None) 
    parser.add_argument("-o", "--output_dir", type=str, default="./output") 
    parser.add_argument("--model_file", type=str, default="/mnt/silver/zsj/model/CodeGPT-small-java")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file", default="/mnt/gold/zsj/SecretCodeLM/flax_gpt2/javaCorpus/literals.json")
    parser.add_argument("--block_size", default=64, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--logging_steps", default=100, type=int)
    args = parser.parse_args()

    handler = logging.StreamHandler()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

    LOGGER.info(f"client pid {os.getpid()}")
    use_gpu = args.mode == "gpu"
      
    if args.mode == "spu":
        with open(args.config, "r") as file:
            conf = json.load(file)
        ppd.init(conf["nodes"], conf["devices"])

    special_tokens = get_special_tokens(args.lit_file)
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model_file, 
        do_lower_case=False, 
        sep_token='<EOL>', bos_token='<s>', eos_token='</s>', 
        pad_token='<pad>', unk_token='<|UNKNOWN|>', 
        additional_special_tokens=special_tokens)

    model = GPT2LMHeadModel.from_pretrained(args.model_file) \
                    if use_gpu else FlaxGPT2LMHeadModel.from_pretrained(args.model_file) # , from_pt=True
    
    model_params = None
    if not use_gpu:
        config = GPT2Config(**model.config.to_dict())
        model_params = model.params
        model = FlaxGPT2LMHeadModel(config=config)
        global MODEL
        MODEL = FlaxGPT2LMHeadModel(config=config)


    LOGGER.info(f"------Run on {args.mode}")
    eval_acc(args, model, tokenizer, model_params = model_params, start_line=0, end_line = 10) # , start_line = 0, end_line = 10
    LOGGER.info("finished")
