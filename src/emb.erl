-module(emb).
-export([load/1, unload/1, encode/2, encode_batch/2, dim/1,
         cosine/2, dot/2,
         new_index/1, index/4, index_batch/3, search/4]).

-opaque encoder() :: map().
-export_type([encoder/0]).

-spec load(#{tokenizer  := file:filename(),
             model      := file:filename(),
             pooling    => mean | cls | none,
             normalize  => boolean(),
             output_name => binary()}) -> {ok, encoder()} | {error, term()}.
load(Opts) ->
    case {maps:find(tokenizer, Opts), maps:find(model, Opts)} of
        {{ok, TokPath}, {ok, ModelPath}} ->
            do_load(TokPath, ModelPath, Opts);
        _ ->
            {error, {missing_required_opts, [tokenizer, model]}}
    end.

do_load(TokPath, ModelPath, Opts) ->
    case tok:load(TokPath) of
        {error, _} = Err -> Err;
        {ok, Tok}        ->
            case onyx:load(ModelPath) of
                {error, _} = Err -> Err;
                {ok, Session}    ->
                    try
                        {AutoOut, AutoPool} = detect_output(Session),
                        Pooling    = maps:get(pooling,      Opts, AutoPool),
                        Normalize  = maps:get(normalize,    Opts, true),
                        OutName    = maps:get(output_name,  Opts, AutoOut),
                        InDType    = detect_input_dtype(Session),
                        EmbDim     = detect_dim(Session, OutName),
                        {ok, #{tok         => Tok,
                               session     => Session,
                               pooling     => Pooling,
                               normalize   => Normalize,
                               output_name => OutName,
                               input_dtype => InDType,
                               dim         => EmbDim}}
                    catch
                        error:Reason -> {error, Reason}
                    end
            end
    end.

-spec unload(encoder()) -> ok.
unload(#{session := Session}) ->
    onyx:unload(Session).

-spec encode(encoder(), binary()) -> {ok, binary()} | {error, term()}.
encode(#{tok := Tok, session := Session, pooling := Pooling,
         normalize := Normalize, output_name := OutName,
         input_dtype := InDType}, Text) ->
    {IdsBin, MaskBin, TypeBin} = tok:encode(Tok, Text),
    MaxLen  = byte_size(IdsBin) div 4,
    Inputs  = build_inputs(Session, IdsBin, MaskBin, TypeBin, MaxLen, InDType),
    case map_size(Inputs) of
        0 ->
            {error, {no_matching_inputs, maps:get(inputs, Session)}};
        _ ->
            try
                case onyx:run(Session, Inputs) of
                    {error, _} = Err -> Err;
                    {ok, Outputs}    ->
                        Tensor   = maps:get(OutName, Outputs),
                        Floats   = onyx:to_list(Tensor),
                        %% MaskBin is always i32 from tok:encode regardless of model input dtype
                        MaskList = [M || <<M:32/signed-little>> <= MaskBin],
                        {_, Shape, _} = Tensor,
                        {SeqLen, HidDim} = case Shape of
                            [_, S, H] -> {S, H};
                            [_, H]    -> {1, H};
                            [H]       -> {1, H};
                            _         -> error({unsupported_tensor_shape, Shape})
                        end,
                        Pooled = emb_pool:apply(Pooling, Floats, SeqLen, {HidDim, MaskList}),
                        Final  = case Normalize of
                            true  -> emb_pool:l2_normalize(Pooled);
                            false -> Pooled
                        end,
                        {ok, << <<F:32/float-little>> || F <- Final >>}
                end
            catch
                error:Reason -> {error, Reason}
            end
    end.

-spec encode_batch(encoder(), [binary()]) -> {ok, [binary()]} | {error, term()}.
encode_batch(E, Texts) ->
    Result = lists:foldl(fun
        (_Text, {error, _} = Err) -> Err;
        (Text, {ok, Acc}) ->
            case encode(E, Text) of
                {error, _} = Err -> Err;
                {ok, Vec}        -> {ok, [Vec | Acc]}
            end
    end, {ok, []}, Texts),
    case Result of
        {error, _} = Err -> Err;
        {ok, RevVecs}    -> {ok, lists:reverse(RevVecs)}
    end.

build_inputs(Session, IdsBin, MaskBin, TypeBin, MaxLen, DType) ->
    InputNames = [Name || {Name, _, _} <- maps:get(inputs, Session)],
    Candidates = #{
        <<"input_ids">>      => i32_to_tensor(IdsBin,  [1, MaxLen], DType),
        <<"attention_mask">> => i32_to_tensor(MaskBin, [1, MaxLen], DType),
        <<"token_type_ids">> => i32_to_tensor(TypeBin, [1, MaxLen], DType)
    },
    maps:filter(fun(K, _) -> lists:member(K, InputNames) end, Candidates).

i32_to_tensor(I32Bin, Shape, i32) ->
    onyx:tensor(I32Bin, Shape, i32);
i32_to_tensor(I32Bin, Shape, i64) ->
    I64Bin = << <<V:64/signed-little>> || <<V:32/signed-little>> <= I32Bin >>,
    onyx:tensor(I64Bin, Shape, i64);
i32_to_tensor(_I32Bin, _Shape, DType) ->
    error({unsupported_input_dtype, DType}).

-spec dim(encoder()) -> pos_integer().
dim(#{dim := D}) -> D.

-spec cosine(binary(), binary()) -> float().
cosine(A, B) when byte_size(A) =:= byte_size(B) ->
    VA = f32_bin_to_list(A),
    VB = f32_bin_to_list(B),
    Dot   = lists:sum([X * Y || {X, Y} <- lists:zip(VA, VB)]),
    NormA = math:sqrt(lists:sum([X * X || X <- VA])),
    NormB = math:sqrt(lists:sum([X * X || X <- VB])),
    case NormA * NormB < 1.0e-12 of
        true  -> 0.0;
        false -> Dot / (NormA * NormB)
    end;
cosine(A, B) ->
    error({dimension_mismatch, byte_size(A) div 4, byte_size(B) div 4}).

-spec dot(binary(), binary()) -> float().
dot(A, B) when byte_size(A) =:= byte_size(B) ->
    VA = f32_bin_to_list(A),
    VB = f32_bin_to_list(B),
    lists:sum([X * Y || {X, Y} <- lists:zip(VA, VB)]);
dot(A, B) ->
    error({dimension_mismatch, byte_size(A) div 4, byte_size(B) div 4}).

%% Internal

%% Inspect model outputs: 2D [batch,dim] -> pooling=none; 3D [batch,seq,dim] -> pooling=mean.
detect_output(Session) ->
    case maps:get(outputs, Session) of
        [{Name, [_, _],    f32} | _] -> {Name, none};
        [{Name, [_, _, _], f32} | _] -> {Name, mean};
        [{Name, _,         _}   | _] -> {Name, mean};
        []                           -> error(no_model_outputs)
    end.

detect_input_dtype(Session) ->
    Inputs = maps:get(inputs, Session),
    case lists:keyfind(<<"input_ids">>, 1, Inputs) of
        {_, _, DType} -> DType;
        false         -> i64
    end.

detect_dim(Session, OutName) ->
    Outputs = maps:get(outputs, Session),
    case lists:keyfind(OutName, 1, Outputs) of
        {_, Shape, _} when Shape =/= [] -> lists:last(Shape);
        {_, [], _}                      -> error({invalid_output_shape, OutName, empty});
        false                           -> error({unknown_output, OutName})
    end.

f32_bin_to_list(<<>>) -> [];
f32_bin_to_list(<<F:32/float-little, Rest/binary>>) ->
    [F | f32_bin_to_list(Rest)].

-spec new_index(encoder()) -> {ok, kvex:index()}.
new_index(#{dim := Dim}) ->
    kvex:new(Dim).

-spec index(kvex:index(), encoder(), kvex:id(), binary()) -> ok | {error, term()}.
index(Ix, E, Id, Text) ->
    case encode(E, Text) of
        {error, _} = Err -> Err;
        {ok, Vec}        -> kvex:add(Ix, Id, Vec)
    end.

-spec index_batch(kvex:index(), encoder(), [{kvex:id(), binary()}]) -> ok | {error, term()}.
index_batch(Ix, E, Pairs) ->
    Result = lists:foldl(fun
        (_, {error, _} = Err) -> Err;
        ({Id, Text}, {ok, Acc}) ->
            case encode(E, Text) of
                {error, _} = Err -> Err;
                {ok, Vec}        -> {ok, [{Id, Vec} | Acc]}
            end
    end, {ok, []}, Pairs),
    case Result of
        {error, _} = Err -> Err;
        {ok, RevPairs}   ->
            case kvex:add_batch(Ix, lists:reverse(RevPairs)) of
                ok              -> ok;
                {error, Reason} -> {error, {batch_insert_failed, Reason}}
            end
    end.

%% Note: encode/2 returns a normalized vector; kvex:cosine_search normalizes again — idempotent for unit vectors.
-spec search(kvex:index(), encoder(), binary(), pos_integer()) ->
    {ok, [{kvex:id(), float()}]} | {error, term()}.
search(Ix, E, Query, K) ->
    case encode(E, Query) of
        {error, _} = Err -> Err;
        {ok, Vec}        -> kvex:cosine_search(Ix, Vec, K)
    end.
