
from sentence_transformers import SentenceTransformer, util
from datetime import datetime


intents = []

# fallback intent
intent = {}
intent["name"] = "Fallback"
intent["questions"] = []
intent["answer"] = (lambda: """
I don't understand. May you repeat with different words?
""")
intents.append(intent)

# welcome intent
intent = {}
intent["name"] = "Welcome"
intent["questions"] = [
    "Hello", "Hi", "Hey", "Hola", "Good morning", "How are you?"
]
intent["answer"] = (lambda: "Hi! How can I help you?")
intents.append(intent)

# transformer
intent = {}
intent["name"] = "Transformer"
intent["questions"] = [
    "What is a Transformer?",
    "What's a transformer neural network?",
    "What are Transformers?",
    "How do transformers work?",
    "How are transformers made?",
    "Tell me what a transformer is"
]
intent["answer"] = (lambda: \
    "A Transformer is an encoder-decoder model that leverages attention " \
    "mechanisms to compute better embeddings and to better align output " \
    "to input."
)
intents.append(intent)

# date
intent = {}
intent["name"] = "Date"
intent["questions"] = [
    "What day is today?",
    "What date is today?",
    "What is today's date?",
    "Tell me the date of today"
]
intent["answer"] = (lambda: "Today is " + datetime.now().date().strftime("%B %d, %Y"))
intents.append(intent)


# help intent
intent = {}
intent["name"] = "Help"
intent["questions"] = [
    "What questions can I ask?",
    "What can I ask?",
    "How can you help me?",
    "What should I ask you?",
    "Help"
]
intent["answer"] = (lambda: \
    "You can say hi, ask what is a Transformer, or ask what date is today." \
)
intents.append(intent)


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# fill corpus and from_corpus_id_to_intent_id mapping
corpus = []
from_corpus_id_to_intent_id = {}
corpus_id = 0
for intent_id, intent in enumerate(intents):
    for question in intent["questions"]:
        corpus.append(question)
        from_corpus_id_to_intent_id[corpus_id] = intent_id
        corpus_id += 1
        
# embed corpus
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def get_matched_intent(query, corpus_embeddings, from_corpus_id_to_intent_id, intents,
                      min_score_threshold=0.6):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hit = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0][0]
    score = hit['score']
    closest_question = corpus[hit['corpus_id']]
    intent_id = from_corpus_id_to_intent_id[hit['corpus_id']]
    matched_intent = intents[intent_id]
    if score >= min_score_threshold:
        return matched_intent, score, closest_question
    # return fallback intent
    return intents[0], 0, ""
    
text = input("hello this is assistant shail, ask me anythinh: \n")

matched_intent, _, _ = get_matched_intent(text, corpus_embeddings, from_corpus_id_to_intent_id, intents)
print("Intent: " + matched_intent["name"])
print("Answer: " + matched_intent["answer"]())
# Intent: Date
# Answer: Today is July 18, 2022