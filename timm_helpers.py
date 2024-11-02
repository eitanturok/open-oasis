import tinygrad
from tinygrad import Tensor, nn, dtypes

"""
Silly, but gotta be done: Mlp has a different implementation in
timm, and so...gotta match weights :shrug:
"""

class Mlp:
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=Tensor.gelu,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.drop1 = drop_probs[0]
        self.norm = norm_layer(hidden_features) if norm_layer is not None else None
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = drop_probs[1]

    def __call__(self, x:Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        if Tensor.training:
            x = x.dropout(self.drop1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc2(x)
        if Tensor.training:
            x = x.dropout(self.drop2)
        return x

