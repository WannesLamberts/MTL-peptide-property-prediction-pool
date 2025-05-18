from tape.models.modeling_bert import (
    ProteinBertAbstractModel,
    ProteinBertModel,
)
from tape.models.modeling_utils import MLMHead
from torch import nn
import torch
from src.model_util import ValuePredictionHead,poolprediction,baseprediction


class MTLTransformerEncoder(ProteinBertAbstractModel):
    """
    Multi Task Learning Transformer Encoder based on ProteinBert from TAPE

    Can be pretrained using a Masked Language Model head or can learn properties using multiple task heads
    """

    def __init__(self, bert_config, mode,type):
        """

        :param bert_config:         A ProteinBertConfig containing the model parameters
        :param mode:                'supervised' for property prediction or 'pretrain' for MLM pretraining
        :param tasks:               A list of tasks to use for supervised learning, each task will get its own
                                    PredictionHead. CCS prediction gets a slightly different predictionhead including
                                    the Charge of the peptide.
        """
        super().__init__(bert_config)

        self.mode = mode
        self.type=type
        self.bert = ProteinBertModel(bert_config)

        if self.mode == "supervised":
            if self.type=="pool":
                self.task_head = poolprediction(bert_config)
            elif self.type=="base":
                self.task_head = baseprediction(bert_config)
        elif self.mode != "pool":
            raise RuntimeError(f"Unrecognized mode {mode}")
        self.init_weights()
    def forward(self, input_ids, features = None):
        outputs = self.bert(input_ids)
        sequence_output, pooled_output = outputs[:2]
        if self.mode == "supervised":
            if self.type=="pool":
                result = torch.cat((pooled_output, features), dim=1).to(torch.float16)
                (out,) = self.task_head(result)
            if self.type=="base":
                (out,) = self.task_head(pooled_output)
        elif self.mode == "pool":
            out = pooled_output
        # (loss), prediction_scores, (hidden_states), (attentions)
        return (out,) + outputs[2:]
