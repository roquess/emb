-module(emb_SUITE).
-include_lib("common_test/include/ct.hrl").
-export([all/0, suite/0,
         cosine_identical/1, cosine_orthogonal/1, cosine_opposite/1, dot_basic/1,
         load_missing_tokenizer/1, load_missing_model/1,
         encode_returns_f32_binary/1, encode_normalized_unit_length/1]).

suite() -> [{timetrap, {seconds, 30}}].

all() -> [cosine_identical, cosine_orthogonal, cosine_opposite, dot_basic,
          load_missing_tokenizer, load_missing_model,
          encode_returns_f32_binary, encode_normalized_unit_length].

f32_bin(Floats) ->
    << <<F:32/float-little>> || F <- Floats >>.

cosine_identical(_Config) ->
    V = f32_bin([1.0, 0.0, 0.0]),
    Sim = emb:cosine(V, V),
    true = abs(Sim - 1.0) < 1.0e-5.

cosine_orthogonal(_Config) ->
    A = f32_bin([1.0, 0.0]),
    B = f32_bin([0.0, 1.0]),
    Sim = emb:cosine(A, B),
    true = abs(Sim) < 1.0e-5.

cosine_opposite(_Config) ->
    A = f32_bin([1.0, 0.0]),
    B = f32_bin([-1.0, 0.0]),
    Sim = emb:cosine(A, B),
    true = abs(Sim - (-1.0)) < 1.0e-5.

dot_basic(_Config) ->
    A = f32_bin([1.0, 2.0, 3.0]),
    B = f32_bin([4.0, 5.0, 6.0]),
    D = emb:dot(A, B),
    true = abs(D - 32.0) < 1.0e-5.   %% 1*4 + 2*5 + 3*6 = 32

load_missing_tokenizer(_Config) ->
    {error, _} = emb:load(#{tokenizer => "/nonexistent/tok.json",
                             model     => "/nonexistent/model.onnx"}).

load_missing_model(Config) ->
    DataDir = ?config(data_dir, Config),
    TokPath = filename:join(DataDir, "tokenizer.json"),
    {error, _} = emb:load(#{tokenizer => TokPath,
                             model     => "/nonexistent/model.onnx"}).

model_path(Config) ->
    DataDir = ?config(data_dir, Config),
    {filename:join(DataDir, "model/tokenizer.json"),
     filename:join(DataDir, "model/model.onnx")}.

encode_returns_f32_binary(Config) ->
    {TokPath, ModelPath} = model_path(Config),
    case filelib:is_regular(ModelPath) of
        false -> {skip, "embedding model not present"};
        true  ->
            {ok, E}   = emb:load(#{tokenizer => TokPath, model => ModelPath}),
            {ok, Vec} = emb:encode(E, <<"Hello world">>),
            Dim       = emb:dim(E),
            true      = (Dim * 4 =:= byte_size(Vec)),
            emb:unload(E)
    end.

encode_normalized_unit_length(Config) ->
    {TokPath, ModelPath} = model_path(Config),
    case filelib:is_regular(ModelPath) of
        false -> {skip, "embedding model not present"};
        true  ->
            {ok, E}   = emb:load(#{tokenizer => TokPath, model => ModelPath,
                                    normalize => true}),
            {ok, Vec} = emb:encode(E, <<"Hello world">>),
            Floats    = [F || <<F:32/float-little>> <= Vec],
            Norm      = math:sqrt(lists:sum([X * X || X <- Floats])),
            true      = abs(Norm - 1.0) < 1.0e-4,
            emb:unload(E)
    end.
