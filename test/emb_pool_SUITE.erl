-module(emb_pool_SUITE).
-include_lib("common_test/include/ct.hrl").
-export([all/0,
         l2_normalize_unit_vec/1,
         l2_normalize_zero/1,
         l2_normalize_scale/1,
         mean_pool_uniform_mask/1,
         mean_pool_with_padding/1,
         mean_pool_all_padding_crashes/1,
         cls_pool/1,
         none_pool/1]).

all() -> [l2_normalize_unit_vec, l2_normalize_zero, l2_normalize_scale,
          mean_pool_uniform_mask, mean_pool_with_padding, mean_pool_all_padding_crashes, cls_pool, none_pool].

l2_normalize_unit_vec(_Config) ->
    [1.0, +0.0] = emb_pool:l2_normalize([1.0, 0.0]).

l2_normalize_zero(_Config) ->
    %% zero vector stays zero
    [+0.0, +0.0] = emb_pool:l2_normalize([0.0, 0.0]).

l2_normalize_scale(_Config) ->
    %% [3.0, 4.0] has norm 5.0 → [0.6, 0.8]
    [N1, N2] = emb_pool:l2_normalize([3.0, 4.0]),
    true = abs(N1 - 0.6) < 1.0e-6,
    true = abs(N2 - 0.8) < 1.0e-6.

mean_pool_uniform_mask(_Config) ->
    %% 2 tokens, 2 dims, mask=[1,1]
    %% Token0=[1.0,2.0], Token1=[3.0,4.0] → mean=[2.0,3.0]
    Floats = [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0] = emb_pool:apply(mean, Floats, 2, {2, [1, 1]}).

mean_pool_with_padding(_Config) ->
    %% 3 tokens, 2 dims, mask=[1,1,0] — third token is padding
    %% Token0=[1.0,0.0], Token1=[3.0,4.0], Token2=[99.0,99.0](padding)
    %% mean of real tokens = [(1+3)/2, (0+4)/2] = [2.0, 2.0]
    Floats = [1.0, 0.0, 3.0, 4.0, 99.0, 99.0],
    [R1, R2] = emb_pool:apply(mean, Floats, 3, {2, [1, 1, 0]}),
    true = abs(R1 - 2.0) < 1.0e-6,
    true = abs(R2 - 2.0) < 1.0e-6.

mean_pool_all_padding_crashes(_Config) ->
    Floats = [1.0, 0.0, 3.0, 4.0],
    try emb_pool:apply(mean, Floats, 2, {2, [0, 0]}) of
        _ -> ct:fail("expected error")
    catch
        error:{invalid_mask, all_padding} -> ok
    end.

cls_pool(_Config) ->
    %% CLS = first HidDim elements of the flat list
    Floats = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [1.0, 2.0] = emb_pool:apply(cls, Floats, 3, {2, [1, 1, 1]}).

none_pool(_Config) ->
    Floats = [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0] = emb_pool:apply(none, Floats, 1, {3, [1]}).
