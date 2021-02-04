from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from tqdm import tqdm


class Emotion:
    def __init__(self, threshold=0.8, batch_size=5):
        self.threshold = threshold
        self.batch_size = batch_size
        self.model_path = 'monologg/bert-base-cased-goemotions-ekman'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForMultiLabelClassification.from_pretrained(self.model_path)
        self.pipeline = MultiLabelPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def predict(self, texts):
        labels = []
        pbar = tqdm(total=len(texts))
        for i in range(0, len(texts), self.batch_size):
            probs = self.pipeline(texts[i: i + self.batch_size])
            for p in probs:
                top_idx = [i
                           for i in range(len(p))
                           if p[i] > self.threshold]
                if len(top_idx) != 1:
                    label = 'N/A'
                else:
                    label = self.model.config.id2label[top_idx[0]]
                labels.append(label)
            pbar.update(self.batch_size)

        return labels
