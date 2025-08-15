# SMS-Spam-Detection-with-DistilBERT-and-Hugging-Face-Trainer-

• Built a binary text classification model using distilbert-base-uncased to classify SMS messages as spam or ham.
• Preprocessed and tokenized messages using Hugging Face tokenizer with truncation and dynamic padding.
• Converted tokenized data to Hugging Face Dataset objects and fine-tuned a DistilBERT model using the Trainer API.
• Evaluated model with precision, recall, F1-score, and accuracy using a custom compute_metrics function.
• Managed training configuration with TrainingArguments, including checkpointing, logging, and evaluation strategy.
