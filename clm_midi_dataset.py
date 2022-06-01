from re import M
from datasets import load_dataset, load_metric
import os
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from itertools import chain
import math
from miditoolkit import MidiFile
import hydra
from hydra_zen import instantiate

# esta signature no permite procesamiento por batches
def midifn2tokens(x, data_dir, tokenizer, return_dict=True):
    x['midi_filename'] = os.path.join(data_dir, x['midi_filename'])
    midi_file = MidiFile(x['midi_filename'])
    tokens = tokenizer.midi_to_tokens(midi_file)[0]
    x['input_ids'] = tokens
    if return_dict:
        return x
    else:
        return tokens

# from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# Main data processing function that will concatenate all texts from our
# dataset and generate chunks of block_size.
def group_seqs(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(
            chain(
                *
                examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


@hydra.main(config_path=None)
def main(cfg):
    set_seed(42)
    obj = instantiate(cfg)
    train_args = TrainingArguments(**obj.train_args)
    dataset = load_dataset('csv', data_files=obj.csv_dir)
    dataset = dataset['train'].train_test_split(test_size=0.2)
    if obj.max_train_samples is not None:
        dataset['train'] = dataset['train'].select(
            range(obj.max_train_samples))
    if obj.max_test_samples is not None:
        dataset['test'] = dataset['test'].select(range(obj.max_test_samples))

    tokenizer = obj.tokenizer  # was REMI
    column_names = dataset['train'].column_names
    def map_func(x): return midifn2tokens(x, obj.data_root, tokenizer, True)

    dataset = dataset.map(
        map_func,
        num_proc=8,
        load_from_cache_file=True,
        # cache_file_name='./cache/tokenized_dataset',
        # batched=True,
        remove_columns=column_names,
        # removing columns makes it easier to make chunks of block_size
    )

    dataset = dataset.map(
        lambda x: group_seqs(x, obj.block_size),
        num_proc=8,
        batched=True,
        load_from_cache_file=True,
        # cache_file_name='./cache/preprocessed_dataset'
    )

    model = AutoModelForCausalLM.from_pretrained(obj.model_name)
    model.resize_token_embeddings(len(tokenizer.vocab))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'] if train_args.do_train else None,
        eval_dataset=dataset['test'] if train_args.do_eval else None,
        args=train_args,
        # Data collator will default to DataCollatorWithPadding, so we change
        # it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if train_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if train_args.do_eval else None,
    )

    if train_args.do_train:
        train_result = trainer.train()  # resume_from_checkpoint=checkpoint
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            obj.max_train_samples if obj.max_train_samples is not None else len(
                dataset['train']))
        metrics["train_samples"] = min(
            max_train_samples, len(
                dataset['train']))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if train_args.do_eval:
        metrics = trainer.evaluate()
        max_eval_samples = obj.max_eval_samples if obj.max_eval_samples is not None else len(
            dataset['test'])
        metrics["eval_samples"] = min(max_eval_samples, len(dataset['test']))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()
