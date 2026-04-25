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

-spec cosine(binary(), binary()) -> float().
cosine(A, B) ->
    VA = f32_bin_to_list(A),
    VB = f32_bin_to_list(B),
    Dot   = lists:sum([X * Y || {X, Y} <- lists:zip(VA, VB)]),
    NormA = math:sqrt(lists:sum([X * X || X <- VA])),
    NormB = math:sqrt(lists:sum([X * X || X <- VB])),
    case NormA * NormB < 1.0e-12 of
        true  -> 0.0;
        false -> Dot / (NormA * NormB)
    end.

-spec dot(binary(), binary()) -> float().
dot(A, B) ->
    VA = f32_bin_to_list(A),
    VB = f32_bin_to_list(B),
    lists:sum([X * Y || {X, Y} <- lists:zip(VA, VB)]).

%% Internal
f32_bin_to_list(<<>>) -> [];
f32_bin_to_list(<<F:32/float-little, Rest/binary>>) ->
    [F | f32_bin_to_list(Rest)].

new_index(_E)                -> error(not_implemented).
index(_Ix, _E, _Id, _Text)   -> error(not_implemented).
index_batch(_Ix, _E, _Pairs) -> error(not_implemented).
search(_Ix, _E, _Text, _K)   -> error(not_implemented).
