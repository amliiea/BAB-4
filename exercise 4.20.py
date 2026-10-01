# Exercise 4.20, based on Example 4.24 question answering
from transformers import pipeline

question_answerer = pipeline('question-answering', framework='pt')
context = (
	'The Eiffel Tower is in Paris, France. It opened to the public in 1889 '
	'and was designed by engineers from Gustave Eiffel\'s company.'
)
questions = [
	'Where is the Eiffel Tower?',
	'When did it open to the public?',
]
for question in questions:
	result = question_answerer({'question': question, 'context': context})
	print(question)
	print(result)

# The model should extract Paris, France and 1889 from the context.
