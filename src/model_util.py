import torch
from pytorch_lightning.callbacks import EarlyStopping
from tape.models.modeling_bert import ProteinBertConfig, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP, ValuePredictionHead
from torch import nn
from torch.nn.utils import weight_norm

from src.util import resize_token_embeddings


class SimpleMLPFix(SimpleMLP):
    def __init__(
        self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__(in_dim, hid_dim, out_dim, dropout)
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        )


class ValuePredictionHeadFix(ValuePredictionHead):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.hidden_dropout_prob)
        self.value_prediction = SimpleMLPFix(
            config.hidden_size *2,
            int(config.hidden_size * 2 / 3),
            1,
            config.hidden_dropout_prob,
        )


ValuePredictionHead = ValuePredictionHeadFix



class EarlyStoppingLate(EarlyStopping):
    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
            or trainer.current_epoch <= trainer.min_epochs
        )


def create_model(args):
    from src.lit_model import LitMTL

    bert_config = ProteinBertConfig.from_pretrained(
        "bert-base",
        vocab_size=len(args.vocab),
        hidden_act=args.activation,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_hidden_layers=args.num_layers,
    )

    if args.mode == "supervised":
        if args.pretrained_model == "own":
            model = LitMTL.load_from_checkpoint(
                args.checkpoint_path,
                strict=False,
                mtl_config=args,
                bert_config=bert_config,
            )
    else:
        raise ValueError(f"Train mode {args.mode} not supported")

    return model
