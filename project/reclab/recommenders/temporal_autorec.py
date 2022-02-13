"""Pytorch implementation of AutoRec recommender."""

import numpy as np
import torch

from . import recommender
from .autorec import Autorec, AutoRecLib


class TemporalAutoRecLib(AutoRecLib):
    def __init__(
        self,
        num_users,
        num_items,
        temporal_window_size,
        seen_users,
        seen_items,
        hidden_neuron,
        dropout=0.05,
        random_seed=0,
    ):
        super().__init__(num_users, num_items, seen_users, seen_items, hidden_neuron, dropout, random_seed)
        self.temporal_window_size = temporal_window_size

    def loss(self, pred, test, mask, lambda_value=1):
        autorec_loss = super().loss(pred, test, mask, lambda_value)
        reg_value_time_enc = torch.mul(lambda_value / 2, list(self.time_encoder.parameters())[0].norm(p="fro") ** 2)
        total_loss = autorec_loss + reg_value_time_enc
        return total_loss

    def prepare_model(self):
        self.time_encoder = torch.nn.Linear(self.temporal_window_size, 1, bias=True)
        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        x = self.time_encoder(x).squeeze(dim=-1)
        return super().forward(x)


class TemporalAutorec(Autorec):
    """The Autorec recommender.

    Parameters
    ----------
    num_users : int
        Number of users in the environment.
    num_items : int
        Number of items in the environment.
    hidden_neuron : int
        Output dimension of hidden layer.
    lambda_value : float
        Coefficient for regularization while training layers.
    train_epoch : int
        Number of epochs to train for each call.
    batch_size : int
        Batch size during initial training phase.
    optimizer_method : str
        Optimizer for training model; either Adam or RMSProp.
    grad_clip : bool
        Set to true to clip gradients to [-5, 5].
    base_lr : float
        Base learning rate for optimizer.
    lr_decay : float
        Rate for decaying learning rate during training.
    dropout : float
        Probability to initialize dropout layer. Set to 0 for no dropout.
    random_seed : int
        Random seed to reproduce results.

    """

    def __init__(
        self,
        num_users,
        num_items,
        temporal_window_size: int = 10,
        hidden_neuron: int = 500,
        lambda_value: float = 1,
        train_epoch: int = 1000,
        batch_size: int = 1000,
        optimizer_method: str = "RMSProp",
        grad_clip: bool = False,
        base_lr: float = 1e-3,
        lr_decay: float = 1e-2,
        dropout: float = 0.05,
        random_seed: int = 0,
        recommender_mode: str = "ignore",
    ):
        """Create new Autorec recommender."""
        super().__init__(
            num_users,
            num_items,
            hidden_neuron,
            lambda_value,
            train_epoch,
            batch_size,
            optimizer_method,
            grad_clip,
            base_lr,
            lr_decay,
            dropout,
            random_seed,
            recommender_mode,
        )

        # Override the model with 'TemporalAutoRecLib'
        self.model = TemporalAutoRecLib(
            num_users,
            num_items,
            temporal_window_size,
            seen_users=set(),
            seen_items=set(),
            hidden_neuron=hidden_neuron,
            dropout=dropout,
            random_seed=random_seed,
        )

        # Init cyclic buffer
        self.ratings = torch.zeros(num_items, num_users, temporal_window_size)

    def train(self, data, optimizer, scheduler):
        """Train for a single epoch."""
        random_perm_doc_idx = np.random.permutation(self.num_items)
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size :]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i + 1) * self.batch_size]

            batch = data[batch_set_idx, :].to(self.device)
            output = self.model(batch)
            mask = self.mask_ratings[batch_set_idx, :].to(self.device)
            last_rating = batch[..., 0]
            loss = self.model.loss(output, last_rating, mask, lambda_value=self.lambda_value)

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super(recommender.ForeverPredictRecommender, self).update(users, items, ratings)
        self.model.prepare_model()
        self.model = self.model.train()
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        ratings = self._ratings.toarray()
        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        curr_ratings = torch.FloatTensor(ratings.T)

        self.ratings = torch.roll(self.ratings, 1, dims=-1)
        self.ratings[..., 0] = curr_ratings

        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(ratings.T).clamp(0, 1)

        self.train_model(self.ratings)
