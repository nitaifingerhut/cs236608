from reclab.recommenders.autorec_w_topics import Autorec_W_Topics
from .autorec import Autorec
from .libfm import LibFM_MLHB
from .knn import KNNRecommender
from .temporal_autorec import TemporalAutorec
from .temporal_autorec2 import TemporalAutorec2

RECOMMENDERS = dict(
    autorec=Autorec,
    autorec_w_topics=Autorec_W_Topics,
    libfm=LibFM_MLHB,
    knn=KNNRecommender,
    temporal_autorec=TemporalAutorec,
    temporal_autorec2=TemporalAutorec2
)
