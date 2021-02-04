from typing import Union, Optional
import numpy as np
from transformers.pipelines import ArgumentHandler
from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    ModelCard
)


class MultiLabelPipeline(Pipeline):
    def __init__(
            self,
            model: Union["PreTrainedModel", "TFPreTrainedModel"],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task
        )

    def __call__(self, *args, **kwargs):
        logits = super().__call__(*args, **kwargs)
        probs = 1 / (1 + np.exp(-logits))
        return probs
