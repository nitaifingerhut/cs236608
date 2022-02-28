import numpy as np
import torch
import scipy
import collections
from .autorec import Autorec, AutoRecLib
from .recommender import PredictRecommender

class AutoRecLib_W_Topics(AutoRecLib):
    def prepare_model(self, item_topics):
        self.item_topics = item_topics
        self.topics_encoder = torch.nn.Linear(self.num_users, self.hidden_neuron, bias=True)
        super().prepare_model()
    
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.topics_encoder(x2)
        for i in range(x1.shape[0]):
            x1[i] += x2[self.item_topics[i]]
        x1 = self.sigmoid(x1)
        x1 = self.dropout(x1)
        x1 = self.decoder(x1)
        return x1
    
    def loss(self, pred, ratings, ratings_per_topic, mask, mask_ratings_per_topic, lambda_value=1):
        mse = (((pred - ratings) * mask) ** 2).mean()
        # reg_value_enc = torch.mul(lambda_value / 2, list(self.encoder.parameters())[0].norm(p="fro") ** 2)
        # reg_value_dec = torch.mul(lambda_value / 2, list(self.decoder.parameters())[0].norm(p="fro") ** 2)
        return mse # torch.add(mse, torch.add(reg_value_enc, reg_value_dec))
    
    def predict(self, user_item, test_data):
        ratings = test_data[0]
        ratings_per_topic = test_data[1]
        users = [triple[0] for triple in user_item]
        items = [triple[1] for triple in user_item]

        user_item = zip(users, items)
        user_idx = set(users)
        item_idx = set(items)
        Estimated_R = self.forward(ratings, ratings_per_topic)
        for item in range(ratings.shape[0]):
            for user in range(ratings.shape[1]):
                if user not in self.seen_users and item not in self.seen_items:
                    Estimated_R[item, user] = 3
        idx = [tuple(users), tuple(items)]
        Estimated_R = Estimated_R.clamp(1, 5)
        return Estimated_R.T[idx].cpu().detach().numpy()

class Autorec_W_Topics(Autorec):

    def __init__(
        self,
        num_users,
        num_items,
        num_topics,
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
            recommender_mode
        )

        self.model = AutoRecLib_W_Topics(
            num_users,
            num_items,
            seen_users=set(),
            seen_items=set(),
            hidden_neuron=hidden_neuron,
            dropout=dropout,
            random_seed=random_seed,
        )

        self.num_topics = num_topics
        self._ratings_per_topic = np.zeros((num_users, num_topics))

    def set_item_topics(self, item_topics):
        self._item_topics = item_topics

    def reset(self, users=None, items=None, ratings=None):
            # super().reset(users, items, ratings)
            self.model.prepare_model(self._item_topics)
            super(Autorec, self).reset(users, items, ratings)
            self._ratings_per_topic = np.zeros((self.num_users, self.num_topics))

    def update_w_topics(self, users=None, items=None, ratings=None):

        self._dense_predictions = None

        # Update the user info.
        if users is not None:
            for user_id, features in users.items():
                if user_id not in self._outer_to_inner_uid:
                    self._outer_to_inner_uid[user_id] = len(self._users)
                    self._inner_to_outer_uid.append(user_id)
                    self._ratings.resize((self._ratings.shape[0] + 1, self._ratings.shape[1]))
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
                    self._ratings.resize((self._ratings.shape[0], self._ratings.shape[1] + 1))
                    self._items.append(features)
                else:
                    inner_id = self._outer_to_inner_iid[item_id]
                    self._items[inner_id] = features

        # Update the rating info.
        if ratings is not None:
            for (user_id, item_id), (rating, context, topic_id) in ratings.items():
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                self._ratings[inner_uid, inner_iid] = rating
                self._ratings_per_topic[inner_uid, topic_id] = rating
                self._rating_contexts[inner_uid, inner_iid].append(context)
                assert inner_uid < len(self._users)
                assert inner_iid < len(self._items)

        # Initialize the exclude dict.
        self._exclude_dict = collections.defaultdict(list)
        if self._exclude is not None:
            for user_id, item_id in self._exclude:
                inner_uid = self._outer_to_inner_uid[user_id]
                inner_iid = self._outer_to_inner_iid[item_id]
                self._exclude_dict[inner_uid].append(inner_iid)

    def update(self, users=None, items=None, ratings=None):  # noqa: D102
        # super().update(users, items, ratings)
        self.update_w_topics(users, items, ratings)
        self.model.prepare_model(self._item_topics)
        self.model = self.model.train()  # TODO: check if necessary since AutoRec calls this in his train_model method. (manorz, 02/16/22)
        for user_item in ratings:
            self.model.seen_users.add(user_item[0])
            self.model.seen_items.add(user_item[1])

        ratings = self._ratings.toarray()
        ratings_per_topic = self._ratings_per_topic
        # Item-based autorec expects rows that represent items
        # pylint: disable=no-member
        self.ratings = torch.FloatTensor(ratings.T)
        self.ratings_per_topic = torch.FloatTensor(ratings_per_topic.T)
        # pylint: disable=no-member
        self.mask_ratings = torch.FloatTensor(ratings.T).clamp(0, 1)
        self.mask_ratings_per_topic = torch.FloatTensor(ratings_per_topic.T).clamp(0, 1)

        self.train_model((self.ratings, self.ratings_per_topic))
    
    def train(self, data, optimizer, scheduler):
        """Train for a single epoch."""
        ratings = data[0]
        ratings_per_topic = data[1]
        # random_perm_doc_idx = np.random.permutation(self.num_items)
        losses = []
        for i in range(self.num_batch):
            # if i == self.num_batch - 1:
                # batch_set_idx = random_perm_doc_idx[i * self.batch_size :]
            # elif i < self.num_batch - 1:
                # batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i + 1) * self.batch_size]

            # batch = data[batch_set_idx, :].to(self.device)
            # output = self.model(batch)
            output = self.model(ratings, ratings_per_topic)
            # mask = self.mask_ratings[batch_set_idx, :].to(self.device)
            mask_ratings = self.mask_ratings.to(self.device)
            mask_ratings_per_topic = self.mask_ratings_per_topic.to(self.device)
            loss = self.model.loss(output, ratings, ratings_per_topic, mask_ratings, mask_ratings_per_topic, lambda_value=self.lambda_value)
            losses.append(loss.item())

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            scheduler.step()
        return losses
    
    def _predict(self, user_item):
        self.model = self.model.eval()
        return self.model.predict(user_item, (self.ratings.to(self.device), self.ratings_per_topic.to(self.device)))