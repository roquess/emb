-module(emb).
-export([load/1, unload/1, encode/2, encode_batch/2, dim/1,
         cosine/2, dot/2,
         new_index/1, index/4, index_batch/3, search/4]).

-opaque encoder() :: map().
-export_type([encoder/0]).

load(_Opts)                  -> error(not_implemented).
unload(_E)                   -> ok.
encode(_E, _Text)            -> error(not_implemented).
encode_batch(_E, _Texts)     -> error(not_implemented).
dim(_E)                      -> error(not_implemented).
cosine(_A, _B)               -> error(not_implemented).
dot(_A, _B)                  -> error(not_implemented).
new_index(_E)                -> error(not_implemented).
index(_Ix, _E, _Id, _Text)   -> error(not_implemented).
index_batch(_Ix, _E, _Pairs) -> error(not_implemented).
search(_Ix, _E, _Text, _K)   -> error(not_implemented).
