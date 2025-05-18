from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "LIUWJ/fine-tuned-home-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 保存到本地（可选）
model.save_pretrained("./fine-tuned-home-bert")
tokenizer.save_pretrained("./fine-tuned-home-bert")