---
list_title:   Deep Learning | Sequence Models and Attention Mechanism
title: Sequence Models and Attention Mechanism
layout: post
mathjax: true
categories: ["AI", "Machine Learning", "Deep Learning"]
---

## Basic Models

 Let's say you want to input a French sentence like `Jane visite I'Afrique Septembre`, and you want to translate it to the English sentence, `Jane is visiting Africa in September`. How can you train a neural network to input the sequence `x` and output the sequence `y`?

 First, let's build a LSTM as an <mark>encoder</mark> network. We feed the network with one word a time from the original french sentence. After ingesting the input sequence the RNN then outputs a vector that represents the input sentence. After that, we can build a <mark>decoder</mark> network that takes the input from the encoder and outputs the translated English words.

 <img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/dl-nlp-w3-1.png">

 Similarly, we can use the same encoder and decoder architecture for image captioning. Our encoder will be a CNN model (e.g., AlexNet), and we replace the last layer (softmax) with an RNN decoder. This model works quite well especially for generating text that are not so long.

 <img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/dl-nlp-w3-2.png">

## Machine translation as building a conditional language model

Since the decoder in a machine translation model is an RNN, it generates text probabilistically, meaning it can produce multiple translations for the same English sentence:

 ```
 Jane visite I'Afrique Septembre 

-> Jane is visiting Africa in September.
-> Jane is going to be visiting Africa in September.
-> In September, Jane will visit Africa.
-> Her African friend welcomed Jane in September.
```

However, we don't want to output a random English translation, we want to output the best and the most likely English translation. In the example above, clearly, the first translation is the most accurate, while the others are suboptimal. 

But how can we control this generation process? We can frame it as a conditional probability problem: given an input $x$, we aim to compute the probability of a sequence $y^{t}$:

 $$
 P(y^{<1>}, \dots, y^{<T_y>} \mid x)
 $$

A naive approach is greedy search, where at each step, we select the word $y^{t}$ with the highest probability. However, this strategy often fails to produce the best overall translation because it optimizes for immediate word probabilities at each step rather than considering the global sequence structure. In this case, the probability of choosing `Jane is going to` is higher than `Jane is visiting`, as `going` is a more commonly seen word in English.

Instead, what we need to do, is to find a search algorithm that can find the following value for us:

$$
\arg\max_{y^{<1>}, \dots, y^{<T_y>}} P(y^{<1>}, \dots, y^{<T_y>} \mid x)
$$

### Beam Search

Let's explain the Beam Search algorithm using our running example above. Once our decoder outputs the probability of $P(y^{<1>} \mid x)$ (output of a `softmax` layer that contains the possibility over 1000 words), unlike greedy search, Beam will consider multiple candidates. The number of candidates is set by `beam width`. If `beam_width = 3`, then Beam will look at three candidates at a time.

Let's say when evaluating the first words, it finds that the choices `in`, `Jane` and `September` are the most likely three possibilities for English outputs. Then Beam search will save the words in memory that it wants to try all three of these words, 