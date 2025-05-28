from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import csv

model = SentenceTransformer("all-MiniLM-L6-v2")

def log_callback(score, epoch, step):
    with open("finetunelog/fn_log_0.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([step, epoch, score])
    # print(f"Logged: step={step}, epoch={epoch}, loss={score}")

# Load dataset
df = pd.read_csv("data_finetune_v1.csv")
train_samples = []
for row in df.itertuples():
    train_samples.append(InputExample(texts=[row.query, row.document]))

# packing dataset
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)

# initiate loses
train_loss = losses.MultipleNegativesRankingLoss(model)

model = model.to('cuda')
# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=10,
    show_progress_bar=True,
    callback=log_callback
)

# Save model
model.save('all-MiniLM-L6-v2-tunedbycaesar')
print("✅ Model fine-tuned berhasil disimpan di 'output/my_finetuned_sbert'!")