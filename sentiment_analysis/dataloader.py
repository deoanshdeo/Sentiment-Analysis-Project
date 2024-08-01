
from transformers import BertTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 512
BATCH_SIZE = 16

pre_trained_model_ckpt = 'bert-base-uncased'

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len, include_raw_text=False):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_raw_text = include_raw_text

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

        if self.include_raw_text:
            output['review_text'] = review

        return output


tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, include_raw_text=False):
    ds = ReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.stars.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, collate_fn=collator)