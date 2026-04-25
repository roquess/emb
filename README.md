# emb

Pure Erlang embedding pipeline for HuggingFace ONNX models.
Combines [tok](https://hex.pm/packages/tok) (tokenization) + [onyx](https://hex.pm/packages/onyx) (ONNX inference) + [kvex](https://hex.pm/packages/kvex) (vector search) into a single high-level API.
No Python. No NIFs beyond onyx and kvex's existing ones.

## Installation

```erlang
%% rebar.config
{deps, [{emb, "0.1.0"}]}.
```

## Quick start

```erlang
{ok, E} = emb:load(#{
    tokenizer => "path/to/tokenizer.json",
    model     => "path/to/model.onnx"
}),

%% Encode text → normalized f32 binary (dim*4 bytes)
{ok, Vec} = emb:encode(E, <<"Hello world">>),

%% Similarity
Sim = emb:cosine(Vec1, Vec2),

%% Index + search via kvex
{ok, Ix} = emb:new_index(E),
ok       = emb:index(Ix, E, <<"doc1">>, <<"some text">>),
{ok, Rs} = emb:search(Ix, E, <<"query">>, 5),

%% Clean up
emb:unload(E),
kvex:delete(Ix).
```

## Getting a model

Any ONNX sentence-transformer from HuggingFace works:

```bash
pip install optimum
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./model/
```

`emb` auto-detects the pooling strategy and embedding dimension from the model's ONNX output spec — no manual configuration needed for standard sentence-transformer models.

## API

```erlang
%% Load a tokenizer + ONNX model as an encoder.
-spec load(#{tokenizer   := file:filename(),
             model       := file:filename(),
             pooling     => mean | cls | none,
             normalize   => boolean(),
             output_name => binary()}) -> {ok, encoder()} | {error, term()}.

%% Encode text → f32 little-endian flat binary (dim(E)*4 bytes).
-spec encode(encoder(), binary()) -> {ok, binary()} | {error, term()}.

%% Encode a list of texts.
-spec encode_batch(encoder(), [binary()]) -> {ok, [binary()]} | {error, term()}.

%% Free the ONNX session.
-spec unload(encoder()) -> ok.

%% Return the embedding dimension.
-spec dim(encoder()) -> pos_integer().

%% Cosine similarity between two f32 flat binaries. Range [-1.0, 1.0].
-spec cosine(binary(), binary()) -> float().

%% Dot product between two f32 flat binaries.
-spec dot(binary(), binary()) -> float().

%% Create a kvex index sized for this encoder's dimension.
-spec new_index(encoder()) -> {ok, kvex:index()}.

%% Encode text and insert into index.
-spec index(kvex:index(), encoder(), kvex:id(), binary()) -> ok | {error, term()}.

%% Encode and insert a batch of {id, text} pairs.
-spec index_batch(kvex:index(), encoder(), [{kvex:id(), binary()}]) -> ok | {error, term()}.

%% Encode query and return top-K results as [{id, score}].
-spec search(kvex:index(), encoder(), binary(), pos_integer()) ->
    {ok, [{kvex:id(), float()}]} | {error, term()}.
```

## Notes

- **Auto-detection**: pooling and dim are inferred from the model's ONNX output. 2D `[batch, dim]` → `pooling=none`; 3D `[batch, seq, dim]` → `pooling=mean`.
- **Input dtype**: inferred from `input_ids` spec in the ONNX session (usually `i64` for BERT-family).
- **`token_type_ids`**: only included in model inputs if the model declares it.
- **normalize**: defaults to `true` — L2-normalized output enables cosine similarity via simple dot product, which kvex uses internally via sied SIMD.

## License

Apache 2.0
