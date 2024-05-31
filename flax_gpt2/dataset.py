import os
import pickle
import gc

import jax.numpy as jnp
import torch
from torch.utils.data import Dataset,DataLoader



class EvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024, start_line = -1, end_line = -1):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if start_line >= 0 and end_line >= 0:
            cached_file = os.path.join(args.data_dir, file_type+"_{}_{}_blocksize_{}".format(start_line, end_line, block_size))
        else:
            cached_file = os.path.join(args.data_dir, file_type+"_blocksize_%d"%(block_size))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            with open(datafile) as f:
                data = f.readlines()

            # split the dataset from start_line to end_line
            if start_line >= 0 and end_line >= 0:
                data = data[start_line: end_line]
   
            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("load %d"%(percent))
            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(input_ids, tokenizer, logger, block_size=block_size)
            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.use_gpu = args.mode == "gpu"
    
    def split(self, input_ids, tokenizer, logger, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                # make the last character a full character(considering the BPE tokenization) or </s> , <EOL>
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size-1-j])[0] == '\u0120' or tokenizer.convert_ids_to_tokens(sample[block_size-1-j]).startswith("<NUM_LIT"):
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]
            # print(len(sample))
            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item]