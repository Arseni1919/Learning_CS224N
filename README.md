# Stanford CS224N: NLP with Deep Learning | Spring 2024

## Lecture 1 - Intro and Word Vectors

- Language is a very deep consept that allows us to achieve a lot of things. It is complex and changing.

### P: (word2vec) Efficient Estimation of Word Representations in Vector Space

They introduced the idea of presenting the words as vectors:
- we have a large corpus ("body" (corpora in plural)) of text:a long list of words
- go through each position _t_ in the text, which has a center word _c_ and context ("outside") word _o_
- use the similarity of the word vectors for _c_ and _o_ to calc the prob of _o_ given _c_ (or vice versa)
- keep adjusting the vectors to maximize this prob

<img src="pics/w2v.png" width="700">
 
- In the class the lecturer shows a very nice example of how exactly the minimization process is done at the end of the lecture.

## Lecture 2 - Word Vectors and Language Models

### SGD

Gradient decent is the idea to take small steps towards minimizing values of the objective function. It is ideal but in reality we usually use so-called _stochastic gradient decent_ (SGD). We take some mini batch from the data and calc the gradient on that. Empirically it is much better to do it that way.
Because the SGD method use mini batches it is naturally has some noice within it. Apparently, that is even better! More exploration is done and SGD gets even better results that other classical approaches.

### Word vectors

We initialize with random numbers and not with zeros. Otherwise, nothing works.
Word2vec works as magic.
You can do simple arithmetic with the word vectors and can get a lot of cultural, semantic, logical insights - and was one of the magical moments of the algorithm. It is so simple and yet captured so many things. 

The idea of the _negative sampling_ is to avoid the huge sum operation in the denominator over all words in a vocabulary. Instead, we can sample some random words and say that the prob of them needs to be low. They use sigmoid function there instead of exponent.

### How to evaluate word vectors?

- intrinsic eval - small internal task to check the model, fast to do
- extrinsic eval - complex "real-life" task, slow to do

### Nemed Entity Recognition (NER)

There we use the word vectors and try to learn the classification problem, e.g. is "Paris" a name or a location. It is done with labeled dataset with supervised learning.



## Lecture 3 - Backpropagation, Neural Network




