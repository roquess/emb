-module(emb_pool).
-export([apply/4, l2_normalize/1]).

-spec apply(mean | cls | none, [float()], pos_integer(), {pos_integer(), [0|1]}) -> [float()].
apply(mean, Floats, _SeqLen, {HidDim, Mask}) ->
    Rows     = chunk(Floats, HidDim),
    MaskSum  = lists:sum(Mask),
    Weighted = lists:foldl(
        fun({Row, M}, Acc) ->
            [A + V * M || {A, V} <- lists:zip(Acc, Row)]
        end,
        lists:duplicate(HidDim, 0.0),
        lists:zip(Rows, Mask)),
    [X / MaskSum || X <- Weighted];
apply(cls, Floats, _SeqLen, {HidDim, _Mask}) ->
    lists:sublist(Floats, HidDim);
apply(none, Floats, _SeqLen, _) ->
    Floats.

-spec l2_normalize([float()]) -> [float()].
l2_normalize(Vec) ->
    Norm = math:sqrt(lists:sum([X * X || X <- Vec])),
    case Norm < 1.0e-12 of
        true  -> Vec;
        false -> [X / Norm || X <- Vec]
    end.

%% @doc chunk(List, N) splits List into sublists of size N.
chunk([], _) -> [];
chunk(List, N) ->
    {Head, Rest} = lists:split(N, List),
    [Head | chunk(Rest, N)].
