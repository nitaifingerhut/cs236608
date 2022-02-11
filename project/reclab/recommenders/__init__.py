from .autorec import Autorec
from .libfm import LibFM_MLHB
from .knn import KNNRecommender

RECOMMENDERS = dict(autorec=Autorec, libfm=LibFM_MLHB, knn=KNNRecommender)