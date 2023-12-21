import json
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

def gen_batch(inputs, batch_size):
    batch_start = 0
    while batch_start < len(inputs):
        yield inputs[batch_start: batch_start + batch_size]
        batch_start += batch_size

def predict(
    model_name,
    input_records,
    max_source_tokens_count=400,
    max_target_tokens_count=200,
    batch_size=96
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch.float).to(device)

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
        predictions.extend(summaries)

    for p in predictions:
        print(p.strip().replace("\n", " "))

if __name__ == "__main__":
    # Считывание входных данных из файла
    with open("input_text.txt", "r") as f:
        input_text = f.read()

    # Преобразование входного текста в резюме
    input_records = [{"text": input_text}]
    predict("IlyaGusev/rut5_base_sum_gazeta", input_records)
