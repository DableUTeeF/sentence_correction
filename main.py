from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import os
from datasets import load_dataset, Dataset
from dataset import MaskedData
import torch
import evaluate


def data_prepare(debug):
    wikipedia = load_dataset(wiki, '20230601.th', split='train')
    thaisum = open(os.path.join(txt_path, 'thaisum_train.txt')).read().split('\n')
    lst20 = open(os.path.join(txt_path, 'lst20_train.txt')).read().split('\n')
    wisesight = open(os.path.join(txt_path, 'wisesight_train.txt')).read().split('\n')
    wikitext_train = Dataset.from_file(wikitext['train1'])['text'] + Dataset.from_file(wikitext['train2'])['text']
    wikitext_val = Dataset.from_file(wikitext['test'])['text'] + Dataset.from_file(wikitext['val'])['text']

    data = wikipedia['text'] + thaisum + lst20 + wisesight

    num_val = 1000 if debug else 10000
    valid_tokens = data[:num_val] + wikitext_val
    if not debug:
        train_tokens = data[10000:] + wikitext_train
    else:
        train_tokens = valid_tokens
    return train_tokens, valid_tokens


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_result = bleu.compute(predictions=decoded_preds,
                               references=decoded_labels)
    return bleu_result


def collate_fn(batch):
    model_inputs = {'labels': [], 'input_ids': []}
    for obj in batch:
        model_inputs['labels'].append(torch.tensor(obj[1]))
        model_inputs['input_ids'].append(torch.tensor(obj[0]))
    model_inputs['labels'] = torch.stack(model_inputs['labels'])
    model_inputs['input_ids'] = torch.stack(model_inputs['input_ids'])
    return model_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--worker', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_mse', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.bs}'
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
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        wikitext = {
            'train1': '/project/lt200060-capgen/palm/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-train-00000-of-00002.arrow',
            'train2': '/project/lt200060-capgen/palm/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-train-00000-of-00002.arrow',
            'test': '/project/lt200060-capgen/palm/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-test.arrow',
            'val': '/project/lt200060-capgen/palm/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-validation.arrow',
        }
    elif os.path.exists("/tarafs/data/project/proj0174-capgen/palm"):
        wiki = '/tarafs/data/project/proj0174-capgen/palm/graelo___wikipedia/20230601.th/1.1.0/fa7b5c4902ab5a491d3fe295e3bf5c519890262c50a0401dcafd108de622068d'
        bs = args.bs
        output_dir = os.path.join('workdir/', expname)
        txt_path = '/tarafs/data/project/proj0174-capgen/palm/caption/data/txt/'
        workers = args.worker
        teacher_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/mGPT"
        config_path = "/tarafs/data/project/proj0174-capgen/palm/caption/cp/tiny-gpt2"
        disable_tqdm = True
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
    elif os.path.exists("/media/palm/Data/capgen/"):
        wikitext = {
            'train1': '/home/palm/.cache/huggingface/datasets/wikitext/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-train-00000-of-00002.arrow',
            'train2': '/home/palm/.cache/huggingface/datasets/wikitext/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-train-00000-of-00002.arrow',
            'test': '/home/palm/.cache/huggingface/datasets/wikitext/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-test.arrow',
            'val': '/home/palm/.cache/huggingface/datasets/wikitext/wikitext-103-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/wikitext-validation.arrow',
        }
        wiki = "graelo/wikipedia"
        bs = 2
        output_dir = '/media/palm/Data/ocr/cp/outs'
        txt_path = '/media/palm/Data/ocr/data/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False
        bleu_path = 'bleu'
    else:
        wiki = "graelo/wikipedia"
        bs = 2
        output_dir = '/tmp/out/'
        txt_path = '/project/lt200060-capgen/peune/ocr/txt/'
        workers = 0
        teacher_path = "ai-forever/mGPT"
        config_path = "sshleifer/tiny-gpt2"
        disable_tqdm = False
        bleu_path = 'bleu'
    context_length = args.context_length
    bleu = evaluate.load(bleu_path)
    train_tokens, valid_tokens = data_prepare(args.debug)

    train_data = MaskedData(train_tokens)
    valid_data = MaskedData(valid_tokens)
    tokenizer = train_data.tokenizer

    config = GPT2Config(
        n_layer=4,
        n_head=4,
        n_embd=768,
        n_ctx=context_length,
        vocab_size=len(tokenizer),
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.cuda()

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=12,
        output_dir=os.path.join(output_dir),
        logging_dir=logdir,
        dataloader_num_workers=workers,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=disable_tqdm,
        # report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume)
