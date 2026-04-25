-module(emb_SUITE).
-include_lib("common_test/include/ct.hrl").
-export([all/0, suite/0,
         cosine_identical/1, cosine_orthogonal/1, cosine_opposite/1, dot_basic/1,
         load_missing_tokenizer/1, load_missing_model/1]).

suite() -> [{timetrap, {seconds, 30}}].

all() -> [cosine_identical, cosine_orthogonal, cosine_opposite, dot_basic,
          load_missing_tokenizer, load_missing_model].

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
