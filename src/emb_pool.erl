-module(emb_pool).
-export([apply/4, l2_normalize/1]).

apply(_Pooling, Floats, _SeqLen, _) -> Floats.

l2_normalize(Vec) -> Vec.
