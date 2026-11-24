"""
Week 6 - Day 1 | Bonus: Tamil / Hindi with a multilingual embedding model

Setup:
    pip install sentence-transformers
"""
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

english = [
    "How do I reset my password?",
    "What is the leave policy?",
    "The weather in Chennai is hot",
]

queries = {
    "Tamil (how do I change my password?)": "என் கடவுச்சொல்லை எப்படி மாற்றுவது?",
    "Hindi (how do I reset my password?)": "मैं अपना पासवर्ड कैसे रीसेट करूं?",
    "Tamil (how many leaves do I get?)": "எனக்கு எத்தனை விடுப்பு நாட்கள் கிடைக்கும்?",
    "Hindi (the weather in Chennai is hot)": "चेन्नई में मौसम गर्म है",
}

english_emb = model.encode(english, convert_to_tensor=True)

for name, text in queries.items():
    q_emb = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, english_emb)[0]
    best = int(scores.argmax())
    print(f"\n{name}\n  {text}")
    for sent, score in zip(english, scores):
        print(f"    {float(score):.3f}  {sent}")
    print(f"  -> closest English sentence: \"{english[best]}\"")

print("\nObservation: a multilingual model maps different languages into ONE shared "
      "vector space, so a Tamil/Hindi sentence lands near its English equivalent. "
      "all-MiniLM-L6-v2 is English-focused and would do poorly here.")
