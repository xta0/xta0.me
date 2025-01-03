---
layout: post
list_title: CS143 Compilers | Parser
title: Parser
mathjax: true
categories: [Compiler]
---

### Regular Language

- The weakest formal languages widely used
- Many applications are simply not regular languages
    - ${(^{i})^{i}} | i >= 0$
    - Regular language can never parse balance brackets
    - Nested `if ... fi` structures
    
    ```shell
    if ...
      if ...
      fi
    fi
    ```
### Parser

- Inputs: list of tokens
- Outputs: parse tree of the program (AST )

### Context-Free Grammars

- Not all strings of tokens are programs
- Parser must distinguish between valid and invalid strings of tokens
- We need
    - A language for describing valid strings of tokens
    - A method for distinguishing valid from invalid strings of tokens
- Programming languages have recursive structure
- An `EXPR` is 

```shell
if EXPR then EXPR else EXPR fi
whie EXPR loop EXPR pool
```
- Context-free grammars are a natural notation for this recursive structure
- A CFG consists of
    - A set of terminals
        - should be tokens of the language
    - A set of non-terminals
    - A start symbol
    - A set of productions (产生式)
        - `X -> Y1...Yn`

### Derivations

- A derivation is a sequence 