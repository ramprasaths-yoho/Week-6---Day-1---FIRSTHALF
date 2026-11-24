"""
Week 6 - Day 1 | Task 1: Creating Embeddings
Local (all-MiniLM-L6-v2) vs API (text-embedding-3-small)

Setup:
    pip install sentence-transformers openai python-dotenv
    export OPENAI_API_KEY="sk-..."      (or put it in a .env file)
"""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
API_MODEL_NAME = "text-embedding-3-small"
# OpenAI price for text-embedding-3-small: USD per 1M tokens (verify on the pricing page)
API_PRICE_PER_1M_TOKENS = 0.02

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the leave policy?",
    "How many casual leaves do I get?",
    "The weather in Chennai is hot",
    "Office timings are 9 to 6",
    "Our team meets every Monday morning",
    "The cafeteria serves lunch at one o'clock",
]

# 100 sentences for the time/cost comparison (cycled from the 8 above, made unique)
sentences_100 = [f"{sentences[i % len(sentences)]} (variation {i})" for i in range(100)]

client = OpenAI()  # reads OPENAI_API_KEY from the environment


def api_embed(texts):
    """Return (embeddings, total_tokens_used)."""
    response = client.embeddings.create(model=API_MODEL_NAME, input=texts)
    vectors = [item.embedding for item in response.data]
    return vectors, response.usage.total_tokens


# ---------------------------------------------------------------- 1. Local, 8 sentences
print("=" * 60)
print("1. LOCAL EMBEDDINGS (8 sentences)")
print("=" * 60)
local_model = SentenceTransformer(LOCAL_MODEL_NAME)
local_model.encode(["warm-up"])  # first call is slower (lazy init), so warm up before timing

start = time.perf_counter()
local_vecs = local_model.encode(sentences)
local_time_8 = time.perf_counter() - start
print(f"Local: shape {local_vecs.shape}, {local_time_8:.2f} s")

# ---------------------------------------------------------------- 2. API, 8 sentences
print("\n" + "=" * 60)
print("2. API EMBEDDINGS (8 sentences)")
print("=" * 60)
start = time.perf_counter()
api_vecs, tokens_8 = api_embed(sentences)
api_time_8 = time.perf_counter() - start
print(f"API: dimension {len(api_vecs[0])}, {api_time_8:.2f} s  ({len(api_vecs)} vectors)")

print(f"\nExpected-output style -> Local: shape {local_vecs.shape}, {local_time_8:.1f} s "
      f"| API: dimension {len(api_vecs[0])}, {api_time_8:.1f} s")

# ---------------------------------------------------------------- 3. 100 sentences: time + cost
print("\n" + "=" * 60)
print("3. LOCAL vs API: 100 sentences")
print("=" * 60)
start = time.perf_counter()
local_model.encode(sentences_100)
local_time_100 = time.perf_counter() - start

start = time.perf_counter()
_, tokens_100 = api_embed(sentences_100)
api_time_100 = time.perf_counter() - start
api_cost_100 = tokens_100 / 1_000_000 * API_PRICE_PER_1M_TOKENS

print(f"{'':<12}{'Time (s)':>10}{'Cost (USD)':>16}{'Dimensions':>12}")
print(f"{'Local':<12}{local_time_100:>10.2f}{'0.00 (free)':>16}{local_vecs.shape[1]:>12}")
print(f"{'API':<12}{api_time_100:>10.2f}{api_cost_100:>16.7f}{len(api_vecs[0]):>12}")
print(f"(API used {tokens_100} tokens at ${API_PRICE_PER_1M_TOKENS}/1M tokens)")
print("Note: local cost is $0 per call, but you pay for your own CPU/GPU and setup; "
      "API time includes network latency.")

# ---------------------------------------------------------------- 4. First 5 numbers of a vector
print("\n" + "=" * 60)
print("4. FIRST 5 NUMBERS OF ONE VECTOR")
print("=" * 60)
print(f'Sentence: "{sentences[0]}"')
print("First 5 numbers (local):", [round(float(x), 4) for x in local_vecs[0][:5]])
print("First 5 numbers (API)  :", [round(x, 4) for x in api_vecs[0][:5]])
print("""
What they represent:
  Each number is the sentence's coordinate along ONE dimension of the model's
  meaning-space (384 dims for MiniLM, 1536 for text-embedding-3-small).
  Individual dimensions are not human-readable (no 'dimension 1 = password');
  the meaning lives in the whole pattern of numbers. Sentences with similar
  meaning end up with vectors pointing in similar directions, which we
  measure with cosine similarity.
""")

# ---------------------------------------------------------------- 5. Local vs API: how to choose
print("=" * 60)
print("5. HOW TO CHOOSE: LOCAL vs API EMBEDDING MODEL")
print("=" * 60)
print("""1. Choose LOCAL (e.g. all-MiniLM-L6-v2) when data is private/sensitive, you need offline use, or volume is high, since there is no per-token cost.
2. Choose API (e.g. text-embedding-3-small) when you want higher quality and bigger vectors with no GPU, no model hosting and no setup.
3. Local latency is predictable and has no network or rate-limit issues; API adds network delay, rate limits and a vendor dependency.
4. Never mix models in one vector database: vectors from different models have different dimensions/spaces, so re-embed everything if you switch.""")
