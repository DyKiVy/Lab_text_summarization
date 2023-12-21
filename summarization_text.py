from datasets import load_dataset
dataset = load_dataset("IlyaGusev/gazeta")

from transformers import AutoTokenizer, T5ForConditionalGeneration
model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

import json
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset


def gen_batch(inputs, batch_size):
    batch_start = 0
    while batch_start < len(inputs):
        yield inputs[batch_start: batch_start + batch_size]
        batch_start += batch_size


def predict(
    model_name,
    input_records,
    output_file,
    max_source_tokens_count=400,
    max_target_tokens_count=200,
    batch_size=96
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).half().to(device)

    predictions = []
    for batch in tqdm(gen_batch(input_records, batch_size)):
        texts = [r["text"] for r in batch]
        input_ids = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_target_tokens_count,
            no_repeat_ngram_size=3,
            early_stopping=True,
            num_beams=2
        )
        summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        #for s in summaries:
        #    print(s)
        predictions.extend(summaries)
    with open(output_file, "w") as w:
        for p in predictions:
            w.write(p.strip().replace("\n", " ") + "\n")

gazeta_test = load_dataset('IlyaGusev/gazeta', revision="v1.0")["test"]
predict("IlyaGusev/rut5_base_sum_gazeta", list(gazeta_test), "t5_predictions.txt")

article_text = input()

input_ids = tokenizer(
    [article_text],
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    max_length=400,
    return_tensors="pt"
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=3,
    num_beams=5,
    early_stopping=True
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print(summary)
