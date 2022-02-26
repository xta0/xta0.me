---
layout: post
list_title: Compilers | Parser
title: Parser
mathjax: true
categories: [Compiler]
---


## Context-Free Grammars

- Not all strings of tokens are programs
- Parser must distinguish between valid and invalid strings of tokens
- We need
    - A language for describing valid strings of tokens
    - A method for distinguishing valid from invalid strings of tokens
- Programming languages have recursive structure
- An `EXPR` is 

```
if EXPR then EXPR else EXPR fi
whie EXPR loop EXPR pool
```

