# Project: A Python A.I. chatbot, trained using deep learning

import nltk
# nltk.download('punkt')		# need to run only once, to download
from nltk.stem.lancaster import LancasterStemmer

# Use to get the root of a word
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle


with open("intents.json") as f:
	data = json.load(f)

# Check if data has already been pre-processed, so we don't need to waste time re-processing
try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

	# *** HOWEVER, if 'intents.json' changes, then delete the .pickle file,
	#	  so that this try statement won't skip over processing the NEW data

except:
	words = []					# bag-of-words criteria
	docs_x, docs_y = [], []		# two lists, to correlate patterns with their tags
	labels = []					# get all tags

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			tokenized_patterns = nltk.word_tokenize(pattern)
			words.extend(tokenized_patterns)
			docs_x.append(tokenized_patterns)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])
	labels = sorted(labels)

	# Stem the bag-of-words criteria
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	training = []
	output = []

	# create a default list of 0s, for convenient reference when doing one-hot-encoding
	out_empty = [0 for _ in range(len(labels))]

	for i, doc in enumerate(docs_x):
		bag = []
		wrds = [stemmer.stem(w.lower()) for w in doc]

		# One-hot-encode the training against the bag-of-words criteria
		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[i])]

		training.append(bag)
		output.append(output_row)

	# Turn into numpy arrays, for more advanced manipulation
	training = np.array(training)	# one-hot-encoding for bag-of-words inputs
	output = np.array(output)		# array of "tags" that correspond to each bag-of-words input

	# Save info to a pickle file
	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)


# ************************************************
# THE COMPLETE A.I. MODEL
# ************************************************

# Reset any previous underlying data-graphs
tensorflow.reset_default_graph()

# Initialize neural network, and create input layer
net = tflearn.input_data(shape=[None, len(training[0])])
	# Note: All training inputs are the same length, so len(training[0]) is fine

# Add two hidden layers to the neural network, with 8 neurons each
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# Add output layer, equal to number of "tags"
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
	# Note: "softmax" means we'll get PROBABILITIES for each neuron ("tag") in the output layer

net = tflearn.regression(net)

# Train the model
model = tflearn.DNN(net)
	# Note: DNN means "deep neural network"

# ************************************************
# END – COMPLETE A.I. MODEL – END
# ************************************************

# Check if model has alraedy been trained, so we don't need to waste time re-training
try:
	model.load("model.tflearn")
except:
	# Pass the model our training data
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
		# Note: "n_epoch" means the number of times the models sees the data

	model.save("model.tflearn")


# Function that checks the user input to a bag-of-words
def bag_of_words(sentence, words):
	bag = [0 for _ in range(len(words))]

	# Tokenize the "sentence" (think of sentence as the input)
	s_words = nltk.word_tokenize(sentence)
	s_words = [stemmer.stem(w.lower()) for w in s_words]

	for s in s_words:
		for i, w in enumerate(words):
			# Check if our current word in our bag-of-words matches
			# the current word in the sentence. If so, one-hot-encode a '1'.
			if w == s: 
				bag[i] = 1

	return np.array(bag)


# Function that chats with the user
	# should probably change the "words" call in this function 
	# to be an argument instead, rather than a global call
def chat():
	print("Start talking with the bot! (type quit to stop)")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		# Pass the user's input into the trained model
		results = model.predict([bag_of_words(inp, words)])
		
		# Find the most-likely tag, and print a random response for that tag
		results_index = np.argmax(results)
		tag = labels[results_index]
		for tg in data["intents"]:
			if tg["tag"] ==  tag:
				responses = tg["responses"]
		print(random.choice(responses))


chat()


print("***************************")
print("Done")
print("***************************")



# ************************************************
# MISC NOTES
# ************************************************
# 
# It would be a good idea to create an "I don't know" intent for the chatbot.
# aka: Just check the probability output at each neuron, and make sure that 
# the neuron selected has a probability above a certain threshold 
# (i.e. the chatbot has a threshold of confidence in interpreting the user's intent)
# 
# ************************************************
# END – MISC NOTES – END
# ************************************************
