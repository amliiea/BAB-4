# Exercise 4.19, based on Example 4.23 sentiment analysis
from transformers import pipeline

classifier = pipeline('sentiment-analysis', framework='pt')
sentences = [
	'The story was warm, clever, and wonderfully acted.',
	'The film was tedious, confusing, and poorly acted.',
]
results = classifier(sentences)
for sentence, result in zip(sentences, results):
	print(sentence)
	print(result)

# The first review is positive and the second review is negative.
