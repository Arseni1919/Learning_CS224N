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

## Lecture 2 - 



