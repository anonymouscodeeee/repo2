import argparse
import logging
import os
import re
from pathlib import Path
import json
from typing import Union

import spu.utils.distributed as ppd
import jax.numpy as jnp
from human_eval.data import write_jsonl, read_problems
import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import set_seed
from transformers import (
    AutoTokenizer, AutoConfig, GPTNeoConfig,
    AutoModelForCausalLM, FlaxAutoModelForCausalLM, FlaxGPTNeoForCausalLM,
    )

LOG_DIR = Path("./logs")
LOGGER = logging.getLogger(__name__)
set_seed(123)

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


def run_on_plaintext(model: Union[AutoModelForCausalLM, FlaxAutoModelForCausalLM], inputs, use_gpu, params = None, generation_configs = None):
    if use_gpu:
        generated_ids = model.generate(
            inputs,
            **generation_configs
        ).cpu().tolist()
    else:
        generated_ids = model.generate(
            inputs,
            params = params,
            **generation_configs,
        )[0].tolist()
    return generated_ids


def run_on_ciphertext(model, seq_ids, spu_model_params, generation_configs = None):
    def text_generation(seq_ids, model_params):
        max_length = generation_configs["max_new_tokens"] + seq_ids.shape[1]
        model_kwargs = model.prepare_inputs_for_generation(seq_ids, max_length)
        next_token = None
        generated_ids = seq_ids.copy()
        
        while generated_ids.shape[1] <= max_length:  # and next_token != generation_configs["eos_token_id"]
            model_outputs = model(seq_ids, params = model_params, **model_kwargs)
            next_token_logits = model_outputs.logits[0, -1, :] # only support batch size = 1
            next_token = jnp.argmax(next_token_logits)

            seq_ids = jnp.array([[next_token]])
            generated_ids = jnp.concatenate([generated_ids, jnp.array([[next_token]])], axis=1)
            model_kwargs = model.update_inputs_for_generation(model_outputs, model_kwargs)
        return generated_ids
    # encode context the generation is conditioned
    input_ids = ppd.device("P1")(lambda x: x)(seq_ids)
    # params = ppd.device("P2")(lambda x: x)(spu_model_params)
    outputs = ppd.device("SPU")(
        text_generation,
    )(input_ids, spu_model_params)

    generated_ids = ppd.get(outputs)
    return generated_ids.tolist()


def human_eval_dataset(tokenizer, batch_size, num_samples_per_task, start_line = -1, end_line = -1):
    problems = read_problems()
    bos_token = tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token
    task_ids = list(problems.keys())

    if start_line >= 0 and end_line >= 0:
        task_ids = task_ids[start_line: end_line]

    for task_id in tqdm(task_ids):
        prompt = problems[task_id]["prompt"].strip()
        prompt = bos_token + prompt

        for _ in range(num_samples_per_task // batch_size):
            input_batch = [prompt for _ in range(batch_size)]
            inputs = tokenizer(input_batch)
            yield task_id, inputs.input_ids


def extract_function_block(string):
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()


def extract_completion_block(ids: list, input_ids_cutoff: int, eos_token_id: int):
    if eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id)]
    return ids[input_ids_cutoff: ]


def human_eval(args, model, tokenizer, model_params = None, start_line = -1, end_line = -1):
    use_gpu = args.mode == "gpu"
    generation_configs = {
        "do_sample": False,
        # "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        # "top_p": args.top_p,
        # "top_k": 0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    eval_dataloader = human_eval_dataset(
                            tokenizer, 
                            args.batch_size, 
                            args.num_samples_per_task, 
                            start_line=start_line, 
                            end_line=end_line
                        )
    if use_gpu:
        model.eval()
        model.to("cuda")
        
    #output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.mode}_n_{args.num_samples_per_task}_{start_line}_{end_line}_nosample.jsonl" 

    if args.mode in ["cpu","gpu"]:   
        for step, batch in enumerate(eval_dataloader):
            task_id = batch[0]
            if use_gpu:
                inputs = torch.tensor(batch[1]).to("cuda")
            else:
                inputs = jnp.array(batch[1])
            
            input_ids_cutoff = inputs.shape[1]
            t1 = time.time()   
            
            generated_ids = run_on_plaintext(model, inputs, use_gpu, model_params, generation_configs)
            duration = time.time() - t1   

            batch_completions = tokenizer.batch_decode(
                [ids[input_ids_cutoff:] for ids in generated_ids],
                    skip_special_tokens=True,
            )

            batch_completions = [extract_function_block(completion) for completion in batch_completions]
            with open(output_file, "a+") as f:
                for completion in batch_completions:
                    info = {"task_id": task_id, "duration": duration, "completion": completion}   
                    f.write(json.dumps(info) + "\n")      

    elif args.mode  == "spu":
        spu_model_params = ppd.device("SPU")._place_arguments(
        ppd.device("P2")(lambda x: x)(model_params)
        )[0][0]

        for step, batch in enumerate(eval_dataloader):
            task_id = batch[0]
            inputs = jnp.array(batch[1])
            
            input_ids_cutoff = inputs.shape[1]
            t1 = time.time()   
            
            generated_ids = run_on_ciphertext(model, inputs, spu_model_params, generation_configs)
            duration = time.time() - t1

            batch_completions = tokenizer.batch_decode(
                [extract_completion_block(ids, input_ids_cutoff, generation_configs["eos_token_id"]) for ids in generated_ids],
                    skip_special_tokens=True,
            )

            batch_completions = [extract_function_block(completion) for completion in batch_completions]
            #save result       
            # comm = None
            comm = get_last_comm()
            assert comm is not None 

            with open(output_file, "a+") as f:
                for completion in batch_completions:
                    info = {"task_id": task_id, "duration": duration, "comm": comm,  "completion": completion}   
                    f.write(json.dumps(info) + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument("-c", "--config", default="./3pc.json")
    parser.add_argument("-m", "--mode", type=str, default="spu", choices=["cpu","spu","gpu"])
    parser.add_argument("-o", "--output_dir", type=str, default="./human_results") 
    parser.add_argument("--model_file", type=str, default="/home/starmage/projects/PyCodeGPT")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_samples_per_task", default=1, type=int,
                        help="Number of examples per task")
    
    # generation config
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=100)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_file)

    model = AutoModelForCausalLM.from_pretrained(args.model_file) \
                    if use_gpu else FlaxGPTNeoForCausalLM.from_pretrained(args.model_file) # , from_pt=True
    
    model_params = None
    if not use_gpu:
        config = GPTNeoConfig(**model.config.to_dict())
        model_params = model.params
        model = FlaxGPTNeoForCausalLM(config=config)
        global MODEL
        MODEL = FlaxGPTNeoForCausalLM(config=config)


    LOGGER.info(f"------Run on {args.mode}")
    human_eval(args, model, tokenizer, model_params = model_params) # , start_line = 0, end_line = 10
    LOGGER.info("finished")
