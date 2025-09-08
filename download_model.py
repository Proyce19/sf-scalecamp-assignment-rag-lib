from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "google/flan-t5-small"
SAVE_DIR = "./flan-t5-small-local"

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)

model.generation_config.save_pretrained(SAVE_DIR)

print("Saved to", SAVE_DIR)
