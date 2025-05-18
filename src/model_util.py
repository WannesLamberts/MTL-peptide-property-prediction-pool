import torch
from pytorch_lightning.callbacks import EarlyStopping
from tape.models.modeling_bert import ProteinBertConfig, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP, ValuePredictionHead
from torch import nn
from torch.nn.utils import weight_norm



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


def get_activation_function(activation_name):
    """Helper function to get activation function from string name"""
    activations = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    return activations[activation_name]


class poolprediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        activation = get_activation_function(config.activation_mlp)
        hidden_sizes = config.hidden_size_mlp  # List, e.g., [512, 256, 128]

        # Input layer
        layers.append(weight_norm(nn.Linear(config.hidden_size * 2, hidden_sizes[0]), dim=None))
        layers.append(activation)
        layers.append(nn.Dropout(config.dropout_mlp, inplace=False))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(weight_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), dim=None))
            layers.append(activation)
            layers.append(nn.Dropout(config.dropout_mlp, inplace=False))

        # Output layer
        layers.append(weight_norm(nn.Linear(hidden_sizes[-1], 1), dim=None))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return (self.main(x),)

class baseprediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        activation = get_activation_function(config.activation_mlp)
        hidden_sizes = config.hidden_size_mlp  # List, e.g., [512, 256, 128]

        # Input layer
        layers.append(weight_norm(nn.Linear(config.hidden_size, hidden_sizes[0]), dim=None))
        layers.append(activation)
        layers.append(nn.Dropout(config.dropout_mlp, inplace=False))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(weight_norm(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), dim=None))
            layers.append(activation)
            layers.append(nn.Dropout(config.dropout_mlp, inplace=False))

        # Output layer
        layers.append(weight_norm(nn.Linear(hidden_sizes[-1], 1), dim=None))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return (self.main(x),)


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
    bert_config.dropout_mlp = args.dropout_mlp
    bert_config.hidden_size_mlp = args.hidden_size_mlp
    bert_config.activation_mlp = args.activation_mlp

    if args.mode == "supervised":
        model = LitMTL.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            mtl_config=args,
            bert_config=bert_config,
        )
    else:
        raise ValueError(f"Train mode {args.mode} not supported")

    return model
