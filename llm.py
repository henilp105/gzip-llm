# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import gzip
import numpy as np
import scipy.special

data = open('input.txt','r').read()

alphabets = ''

for i in range(32,127):
  alphabets+=chr(i)

# print(alphabets)
gen = 'least'

for i in range(10):
  scores = np.array([len(gzip.compress("".join([data,gen,x]).encode())) for x in alphabets])
  # print(scores)
  probs = scipy.special.softmax(-scores*np.log(len(alphabets)))
  # print(probs)
  # print(probs.tolist().index(max(probs)),alphabets[probs.tolist().index(max(probs))])
  x = np.random.choice(range(len(alphabets)),p = probs)
  gen+=alphabets[x]

print(gen)