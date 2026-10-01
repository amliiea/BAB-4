# Exercise 4.18, based on Example 4.22 NLTK text analysis
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

sentence = """
Ada Lovelace and Charles Babbage discussed the Analytical Engine in London
during 1843. Their work influenced the history of computing.
"""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)

# NLTK identifies London and Charles Babbage, but mislabels some proper names.
