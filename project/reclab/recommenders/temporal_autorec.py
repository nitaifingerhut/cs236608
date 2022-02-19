"""Pytorch implementation of AutoRec recommender."""

import collections
import numpy as np
import scipy
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
        return torch.add(autorec_loss, reg_value_time_enc)

    def prepare_model(self):
        self.time_encoder = torch.nn.Linear(self.temporal_window_size, 1, bias=True)
        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        x = self.time_encoder(x).squeeze(dim=-1)
        # TODO: consider to add here non-linearity (such as sigmoid) and/or regularization (such as dropout) like the vanilla AutoRec. (manorz, 02/17/22)  
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
        rec_mode: str = "ignore",
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
            rec_mode,
        )

        # Override the sparse matrix of current numrial ratings with sparse matrix of <temporal_window_size> last numerical ratings (per user per item)
        # self._ratings = scipy.sparse.csr_matrix((temporal_window_size, 0, 0))  # this one ois not working :-(
        self._ratings = [scipy.sparse.csr_matrix((0, 0)) for _ in range(temporal_window_size)]

        self._temporal_window_size = temporal_window_size
        self._curr_temporal_ind = 0

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
    
    def reset(self, users=None, items=None, ratings=None):
        """Reset the recommender with optional starting user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The starting users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The starting items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The starting ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        self._users = []
        self._items = []
        # self._ratings = scipy.sparse.dok_matrix((self._temporal_window_size, 0, 0))
        self._ratings = [scipy.sparse.csr_matrix((0, 0)) for _ in range(self._temporal_window_size)]
        self._rating_contexts = collections.defaultdict(list)
        self._outer_to_inner_uid = {}
        self._inner_to_outer_uid = []
        self._outer_to_inner_iid = {}
        self._inner_to_outer_iid = []
        self._dense_predictions = None  # TODO: check if necessary since PredictRecommender calls this in his update method few lines below. (manorz, 02/16/22)
        self.update(users, items, ratings)

    def update(self, users=None, items=None, ratings=None):
        """Update the recommender with new user, item, and rating data.

        Parameters
        ----------
        users : dict, optional
            The new users where the key is the user id while the value is the
            user features.
        items : dict, optional
            The new items where the key is the user id while the value is the
            item features.
        ratings : dict, optional
            The new ratings where the key is a double whose first index is the
            id of the user making the rating and the second index is the id of the item being
            rated. The value is a double whose first index is the rating value and the second
            index is a numpy array that represents the context in which the rating was made.

        """
        self._dense_predictions = None

        # Update the user info.
        if users is not None:
            for user_id, features in users.items():
                if user_id not in self._outer_to_inner_uid:
                    self._outer_to_inner_uid[user_id] = len(self._users)
                    self._inner_to_outer_uid.append(user_id)
                    self._ratings[self._curr_temporal_ind].resize((self._ratings.shape[0] + 1, self._ratings.shape[1]))
                    self._users.append(features)
                else:
                    inner_id = self._outer_to_inner_uid[user_id]
                    self._users[inner_id] = features

        # Update the item info.
        if items is not None:
            for item_id, features in items.items():
                if item_id not in self._outer_to_inner_iid:
                    self._outer_to_inner_iid[item_id] = len(self._items)
                    self._inner_to_outer_iid.append(item_id)
                    self._ratings[self._curr_temporal_ind].resize((self._ratings.shape[0], self._ratings.shape[1] + 1))
                    self._items.append(features)
                else:
                    inner_id = self._outer_to_inner_iid[item_id]
                    self._items[inner_id] = features
        
        # Update the rating info.
        if ratings is not None:
            for (user_id, item_id), (rating, context) in ratings.items():
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                self._ratings[self.curr_temporal_ind][inner_uid, inner_iid] = rating
                self._rating_contexts[self.curr_temporal_ind, inner_uid, inner_iid].append(context)
                assert inner_uid < len(self._users)
                assert inner_iid < len(self._items)
        
        # increament the ind by 1 mod window size, so new ratings will override old ratings
        self.curr_temporal_ind += 1 
        self._curr_temporal_ind %= self.temporal_window_size  

        # Initialize the exclude dict.
        self._exclude_dict = collections.defaultdict(list)
        if self._exclude is not None:
            for user_id, item_id in self._exclude:
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                self._exclude_dict[inner_uid].append(inner_iid)
        
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
