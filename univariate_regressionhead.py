# Copyright contributors to the Terratorch project

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Classification head"""

    # how to allow cls token?
    def __init__(
        self,
        in_dim: int,
        num_classes: int,  # should be set to 1 to allow for regression
        dim_list: list[int] | None = None,
        dropout: float = 0,
        linear_after_pool: bool = False,
    ) -> None:
        """Constructor

        Args:
            in_dim (int): Input dimensionality
            num_classes (int): Number of output classes
            dim_list (list[int] | None, optional):  List with number of dimensions for each Linear
                layer to be created. Defaults to None.
            dropout (float, optional): Dropout value to apply. Defaults to 0.
            linear_after_pool (bool, optional): Apply pooling first, then apply the linear layer. Defaults to False
        """
        super().__init__()
        self.num_classes = num_classes
        self.linear_after_pool = linear_after_pool
        if dim_list is None:
            pre_head = nn.Identity()
        else:

            def block(in_dim, out_dim):
                return nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim), nn.ReLU())

            dim_list = [in_dim, *dim_list]
            pre_head = nn.Sequential(*[block(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])
            in_dim = dim_list[-1]
        dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(
            pre_head,
            dropout,
            nn.Linear(in_features=in_dim, out_features=1), # changed to accomodate 1 output node
        )

    def forward(self, x: Tensor):
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        if self.linear_after_pool:
            x = x.mean(axis=1)
            out = self.head(x)
        else:
            x = self.head(x)
            out = x.mean(axis=1)
        return out