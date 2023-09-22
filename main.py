from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--grad_accum', type=int, default=32)
    parser.add_argument('--worker', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--distil', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--temperature', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=0.5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_mse', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.context_length}_{args.temperature}_{args.alpha}_{args.grad_accum}_{args.bs}'
    logdir = os.path.join(args.logdir, expname)
    print(expname, flush=True)
    if os.path.exists("/project/lt200060-capgen/coco"):
        wiki = '/home/nhongcha/.cache/huggingface/datasets/graelo___wikipedia/20230601.th/1.1.0/fa7b5c4902ab5a491d3fe295e3bf5c519890262c50a0401dcafd108de622068d'
        bs = args.bs
        output_dir = os.path.join('/project/lt200060-capgen/palm/capocr/workdir/', expname)
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = args.worker
        teacher_path = "/project/lt200060-capgen/palm/huggingface/mGPT"
        config_path = "/project/lt200060-capgen/palm/huggingface/tiny-gpt2"
        disable_tqdm = True
    elif os.path.exists("/tarafs/data/project/proj0174-capgen/palm"):
        wiki = '/tarafs/data/project/proj0174-capgen/palm/graelo___wikipedia/20230601.th/1.1.0/fa7b5c4902ab5a491d3fe295e3bf5c519890262c50a0401dcafd108de622068d'
        bs = args.bs
        output_dir = os.path.join('workdir/', expname)
        txt_path = '/tarafs/data/project/proj0174-capgen/palm/caption/data/txt/'
        workers = args.worker
        teacher_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/mGPT"
        config_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/tiny-gpt2"
        disable_tqdm = True
    elif os.path.exists("/media/palm/Data/capgen/"):
        wiki = "graelo/wikipedia"
        bs = 1
        output_dir = '/media/palm/Data/ocr/cp/outs'
        txt_path = '/media/palm/Data/ocr/data/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False
    else:
        wiki = "graelo/wikipedia"
        bs = 2
        output_dir = '/tmp/out/'
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False
    context_length = args.context_length
    config = GPT2Config(
        n_layer=4,
        n_head=4,
        n_embd=768,
        n_ctx=context_length,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


