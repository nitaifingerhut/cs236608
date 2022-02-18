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
        reg_value_time_enc = torch.mul(lambda_value / 2, list(self.W.parameters())[0].norm(p="fro") ** 2)
        # reg_value_time_enc = torch.mul(lambda_value / 2, self.W.norm(p="fro") ** 2)
        total_loss = autorec_loss + reg_value_time_enc
        return total_loss

    def prepare_model(self):
        # self.W = 0.9 ** torch.linspace(0, self.temporal_window_size, steps=self.temporal_window_size)
        # self.W = self.W.reshape((self.temporal_window_size, 1))

        self.W = torch.nn.Linear(self.temporal_window_size, 1, bias=False)
        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        # x = torch.einsum('bij,jk->bik', x, self.W).squeeze(dim=-1)
        x = self.W(x).squeeze(dim=-1)
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
        temporal_window_size: int = 1,
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
        losses = []
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
            losses.append(loss.item())

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()
        return losses

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super(recommender.ForeverPredictRecommender, self).update(users, items, ratings)
        self.model.prepare_model()
        self.model = self.model.train()
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        # Update the rating info.
        curr_rating = np.zeros(shape=(self.num_users, self.num_items), dtype=float)
        if ratings is not None:
            for (user_id, item_id), (rating, context) in ratings.items():
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                curr_rating[inner_uid, inner_iid] = rating
                self._rating_contexts[inner_uid, inner_iid].append(context)
                assert inner_uid < len(self._users)
                assert inner_iid < len(self._items)

        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        curr_ratings = torch.FloatTensor(curr_rating.T)

        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(curr_rating.T).clamp(0, 1)

        for item in range(self.ratings.shape[0]):
            for user in range(self.ratings.shape[1]):
                if self.mask_ratings[item, user]:
                    self.ratings[item, user, :] = torch.roll(self.ratings[item, user, :], 1, dims=-1)
                    self.ratings[item, user, 0] = curr_ratings[item, user]

        self.train_model(self.ratings)
