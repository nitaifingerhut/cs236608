"""Pytorch implementation of AutoRec recommender."""

import math
import numpy as np
import torch

from . import recommender


class AutoRecLib(torch.nn.Module):
    def __init__(self, num_users, num_items, seen_users, seen_items, hidden_neuron, dropout=0.05, random_seed=0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.seen_users = seen_users
        self.seen_items = seen_items

        self.hidden_neuron = hidden_neuron
        self.random_seed = random_seed
        self.dropout_p = dropout
        self.sigmoid = torch.nn.Sigmoid()

    def loss(self, pred, test, mask, lambda_value=1):
        mse = (((pred * mask) - test) ** 2).sum()
        reg_value_enc = torch.mul(lambda_value / 2, list(self.encoder.parameters())[0].norm(p="fro") ** 2)
        reg_value_dec = torch.mul(lambda_value / 2, list(self.decoder.parameters())[0].norm(p="fro") ** 2)
        return torch.add(mse, torch.add(reg_value_enc, reg_value_dec))

    def prepare_model(self):
        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x

    def predict(self, user_item, test_data):
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]

        user_item = zip(users, items)
        user_idx = set(users)
        item_idx = set(items)
        Estimated_R = self.forward(test_data)
        for item in range(test_data.shape[0]):
            for user in range(test_data.shape[1]):
                if user not in self.seen_users and item not in self.seen_items:
                    Estimated_R[item, user] = 3
        idx = [tuple(users), tuple(items)]
        Estimated_R = Estimated_R.clamp(1, 5)
        return Estimated_R.T[idx].cpu().detach().numpy()


class Autorec(recommender.ForeverPredictRecommender):
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
        hidden_neuron=500,
        lambda_value=1,
        train_epoch=1000,
        batch_size=1000,
        optimizer_method="RMSProp",
        grad_clip=False,
        base_lr=1e-3,
        lr_decay=1e-2,
        dropout=0.05,
        random_seed=0,
        recommender_mode: str = "ignore",
    ):
        """Create new Autorec recommender."""
        super().__init__(mode=recommender_mode)

        # We only want the function arguments so remove class related objects.
        self._hyperparameters.update(locals())
        del self._hyperparameters["self"]
        del self._hyperparameters["__class__"]

        self.model = AutoRecLib(
            num_users,
            num_items,
            seen_users=set(),
            seen_items=set(),
            hidden_neuron=hidden_neuron,
            dropout=dropout,
            random_seed=random_seed,
        )
        self.lambda_value = lambda_value
        self.num_users = num_users
        self.num_items = num_items
        self.train_epoch = train_epoch
        self.batch_size = num_users  # batch_size
        self.num_batch = int(math.ceil(self.num_items / float(self.batch_size)))
        self.base_lr = base_lr
        self.optimizer_method = optimizer_method
        self.random_seed = random_seed

        self.lr_decay = lr_decay
        self.grad_clip = grad_clip
        np.random.seed(self.random_seed)
        # pylint: disable=no-member
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train_model(self, data):
        """Train for all epochs in train_epoch."""
        self.model.train()
        if self.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        elif self.optimizer_method == "RMSProp":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.base_lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=self.lr_decay)

        self.model.to(self.device)
        self.losses = []
        for epoch in range(self.train_epoch):
            x = self.train(data, optimizer, scheduler)
            self.losses.extend(x)

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
            loss = self.model.loss(output, batch, mask, lambda_value=self.lambda_value)
            losses.append(loss.item())

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()
        return losses

    @property
    def name(self):  # noqa: D102
        return "autorec"

    def _predict(self, user_item):
        self.model = self.model.eval()
        return self.model.predict(user_item, self.ratings.to(self.device))

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        self.model.prepare_model()  # TODO: check if necessary since AutoRec calls this in his update method 5 lines below. (manorz, 02/16/22)
        super().reset(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super().update(users, items, ratings)
        self.model.prepare_model()
        self.model = self.model.train()  # TODO: check if necessary since AutoRec calls this in his train_model method. (manorz, 02/16/22)
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        ratings = self._ratings.toarray()
        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        self.ratings = torch.FloatTensor(ratings.T)
        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(ratings.T).clamp(0, 1)

        self.train_model(self.ratings)
