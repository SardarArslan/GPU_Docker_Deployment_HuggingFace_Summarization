from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("Sardar/sql-model-101")
model = AutoModelForSeq2SeqLM.from_pretrained("Sardar/sql-model-101")


model.save_pretrained('./sql')
tokenizer.save_pretrained('./sql')
