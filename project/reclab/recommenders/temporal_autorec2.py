import imp
from matplotlib import collections
import numpy as np
import collections
import torch

from .autorec import AutoRecLib
from .temporal_autorec import TemporalAutorec, TemporalAutoRecLib
from . import recommender


class TemporalAutoRecLib2(TemporalAutoRecLib):
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
        super().__init__(num_users, num_items, temporal_window_size, seen_users, seen_items, hidden_neuron, dropout, random_seed)

    def prepare_model(self):

        self.temp_rat_encoder = torch.nn.Linear(self.temporal_window_size, 1, bias=True)
        self.temp_rec_encoder = torch.nn.Linear(self.temporal_window_size, 1, bias=True)

        self.encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.decoder = torch.nn.Linear(self.hidden_neuron, self.num_users, bias=True)

    def forward(self, x):
        temp_rat_embd = self.temp_rat_encoder(x[0]).squeeze(dim=-1)
        temp_rec_embd = self.temp_rec_encoder(x[1]).squeeze(dim=-1)
        temp_all_embd = temp_rat_embd + temp_rec_embd[None, ...]
        return super(TemporalAutoRecLib, self).forward(temp_all_embd)
    
    def predict(self, user_item, test_data):
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]

        user_item = zip(users, items)
        user_idx = set(users)
        item_idx = set(items)
        Estimated_R = self.forward(test_data)
        for item in range(test_data[0].shape[0]):
            for user in range(test_data[0].shape[1]):
                if user not in self.seen_users and item not in self.seen_items:
                    Estimated_R[item, user] = 3
        idx = [tuple(users), tuple(items)]
        Estimated_R = Estimated_R.clamp(1, 5)
        return Estimated_R.T[idx].cpu().detach().numpy()


class TemporalAutorec2(TemporalAutorec):
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
        rats_init_mode: str = "zeros",
        recs_init_mode: str = "zeros",
    ):

        super().__init__(
                num_users,
                num_items,
                temporal_window_size,
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
                rats_init_mode,
        )

        # Override the TemporalAutorecLib with 'TemporalAutoRecLib2'
        self.model = TemporalAutoRecLib2(
            num_users,
            num_items,
            temporal_window_size,
            seen_users=set(),
            seen_items=set(),
            hidden_neuron=hidden_neuron,
            dropout=dropout,
            random_seed=random_seed,
        )

        self.first = True
        # Init cyclic buffer for the recommendations history
        if recs_init_mode == "randint":
            self.recommendations = torch.randint(0, num_items, size=(num_users, temporal_window_size), dtype=torch.float32)
        else:
            self.recommendations = torch.zeros(num_users, temporal_window_size, dtype=torch.float32)

    def reset(self, users=None, items=None, ratings=None):  # noqa: D102
        super().reset(users, items, ratings)
        self.first = True

    # override TemporalAutorec update method (and not calling it) to avoid calling train_model before prepering the recs
    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        super(recommender.ForeverPredictRecommender, self).update(users, items, ratings)

        # Recommendations  supposed to be None only at reset. 
        # In contrast to Autorec & TemporalAutorec, we don't train the model on reset, only on iteration, since on reset we don't have recommendations. 
        # On reset, the ratings we have are originaed from the env, but the model can't correlate them with any past recommendations.
        if self.first:
            self.first = False
            return
        
        self.model.prepare_model()
        self.model = self.model.train()
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        # Update the rating info.
        curr_rating = np.zeros(shape=(self.num_users, self.num_items), dtype=float)
        curr_recommendation = np.zeros(shape=(self.num_users), dtype=float)

        if ratings is not None:
            for (user_id, item_id), (rating, context) in ratings.items():
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                curr_rating[inner_uid, inner_iid] = rating
                curr_recommendation[inner_uid] = item_id
                self._rating_contexts[inner_uid, inner_iid].append(context)
                assert inner_uid < len(self._users)
                assert inner_iid < len(self._items)
        
        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        curr_ratings = torch.FloatTensor(curr_rating.T)
        curr_recommendation = torch.FloatTensor(curr_recommendation.T)

        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(curr_rating.T).clamp(0, 1)
        self.mask_recommendation = torch.FloatTensor(curr_recommendation.T).clamp(0, 1)  # NOTE: we could use the same mask as mask_ratings. (manorz, 02/19/22)

        for item in range(self.ratings.shape[0]):
            for user in range(self.ratings.shape[1]):
                if self.mask_ratings[item, user]:
                    self.ratings[item, user, :] = torch.roll(self.ratings[item, user, :], 1, dims=-1)
                    self.ratings[item, user, 0] = curr_ratings[item, user]
        
        for user in range(self.recommendations.shape[1]):
            if self.mask_recommendation[user]:
                    self.recommendations[user, :] = torch.roll(self.recommendations[user, :], 1, dims=-1)
                    self.recommendations[user, 0] = curr_recommendation[user]

        self.train_model((self.ratings, self.recommendations))
    
    def train(self, data, optimizer, scheduler):
        """Train for a single epoch."""
        
        ratings = data[0]
        recommendations = data[1]

        random_perm_doc_idx = np.random.permutation(self.num_items)
        losses = []
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size :]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i + 1) * self.batch_size]

            ratings_batch = ratings[batch_set_idx, :].to(self.device)
            recommendations_batch = recommendations.to(self.device)

            output = self.model((ratings_batch, recommendations_batch))
            mask = self.mask_ratings[batch_set_idx, :].to(self.device)
            last_rating = ratings_batch[..., 0]
            loss = self.model.loss(output, last_rating, mask, lambda_value=self.lambda_value)
            losses.append(loss.item())

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()
        return losses
    
    def _predict(self, user_item):
        self.model = self.model.eval()
        return self.model.predict(user_item, (self.ratings.to(self.device), self.recommendations.to(self.device)))
