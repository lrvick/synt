import nltk

negative_file = open('negative.txt','rw')

for line in negative_file:
    tokenized_text = nltk.word_tokenize(line)
    print nltk.pos_tag(tokenized_text)
