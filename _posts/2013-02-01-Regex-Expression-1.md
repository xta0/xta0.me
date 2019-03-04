---
layout: post
list_title: Regular Expression  | 基本语法 | Syntax 
title: 正则表达式的基本语法
categories: [Regular Expression]
---

## Overview

- 1956年 早起版本的正则语言首次出现，并运用到有限状态机理论中
- 1968年 正则表达式开始兴起，在文本编辑器和语义分析中
- 1980年 Perl中正则表达式提供了更多功能，并逐步标准化
- 1997年 Perl Compatible Regular Expressions被提出，统一了正则表达式的实现方式
- Regular Expressions Engines
    - C/C++
    - .NET
    - PHP
    - Java
    - JavasScript
    - Python
    - MySQL
    - UNIX Tools(POSIX,BRE,ERE)
    - Apache(POSIX,BRE,ERE)
    - Perl
- Online Tools
    - [Regex 101](https://regex101.com/)

- Basic Syntax
    - Enclose in Literals such as `/regex/`

- Matching Algorithm
    - Geedy 
    - Backtracking

- Modes
    - Standard Mode: `/regex/`
    - Global Mode: `/regex/g`
    - Single Line Mode: `/regex/s`
    - Case insensitive Mode: `/regex/i`
    - Multi Line Mode: `/regex/m`

- Meta Characters
    - [`*`,`+`,`-`,`!`,`=`,`()`,`{}`,`[]`,`^`,`$`,`|`,`?`,`:`,`\`]

## Syntax

### Matching Character

-  **WildCard**
    - Notation: `.`
    - 单个字符匹配所有除了换行以外的字符
        - `/.ohn/`-->`John`,`mohn`
        - `/3.14/` -->`3.14`,`3x14`,...        
    - 用`\`做转义字符
        - `user\.txt` ---> `user.txt`

- **Character Set**
    - Notion: `[]`
    - 如果`[]`前后没有字符，则每个字符都按上述规则匹配
        - `\[abcde]\` --> `dcbeas`
        - `[a-zA-Z_.]+@\w+.(com|cn)` --> `jayson.xu@foxmail.com`
    - 如果`[]`前后有字符，则只有对应位的单个字符按照规则匹配
        - `/[cd]ash/ ` -> `cash, dash`

- **Character Ranges**
    - Notion: `-`
    - `\[a-z]\`,`\[A-Z]\`,`\[a-zA-Z]\`, `[0-9]`
    - `/[a-z] team/` --> `a team , b team, c team ... z team`
    - `/5[0-5]/` --> `50, 51, 52, 52, 54, 55`
    ```
- **Not Symbol**
    - Notion `^`
    - `/[^abcdef]/` matches every character except abcdef
    - `/[^cd]ash/` matches 4 characters except cash, dash
    - `/[^vr]a[nd]ish/` matches 6 characters except vanish, radish

- **Escaping Meta Character**
    - Notion: `\`
        - `a\.txt` --> `a.txt`
        - `user\/local` -> `user/local`
        - `c:\\user\\local\\a.txt` --> `c:\user\local\a.txt`
    - 换行:`\n`, tab: `\t`
    - 在`[]`中只有四个字符需要被转义`[-/*]]`
        - `[\-]` --> `-`
        - `[\-\/]` --> `-/`
        - `[\-\/\*]` --> `-/*`
        - `[\-\/\*\]]` -> `-/*]`

- **Easy way to write Sets**
    - `\w`是`/[a-zA-Z0-9_]/`的简写，表示word和下划线集合
    - `\W`是`/[^a-zA-Z0-9_]/`的简写，是上面的补集，表示非word+非下划线
    - `\s` -> `[\t]` -> tab and white space
    - `\S` -> `[^\t]` -> No space, no tab
    - `\d` -> `\[0-9]\` -> digits only
    - `[\D]` -> no digit
    - `[^\d]` -> no digit

- **Quantified Repetition**
    - Notation: `*`
        - Match zero or more of previous
            - `/Buz*/` --> `Bu,Buz,Buzz,Buzzz,...`
    - Notation: `+`
        - Match one or more of previous
            - `[0-9]+` --> match all numbers
            - `[0-9][0-9][0-9]+` --> `[x]012, [x]2, [x]86`
    - Notation: `?`
        - Match one or zero previous
            - `/Flavou?r/` --> `Flavour, Flavor`
    
    - Notion: `{number}`,`{min, max}`,`{min,}`
        - Match exactly <em>number</em> of previous
            - `/[0-9]{3}/` --> `000,723, [x]45, [x]4544`
            - phone number: `/[0-9]{3}-[0-9]{4}-[0-9]{4}/`
        - Match as least <em>min</em> but not more than <em>max</em> of previous characters
            - `/[0-9]{3,5}/` --> `000,727, [x]45,4545,[x]454545`
        - Match at least <em>min</em> times
            - `/[0-9]{3,}/` --> `000,723, [x]45, 4545, 345435`

- **Lazy Expressions**
    - Greey mode
        - 如果当前Quantifier满足条件，则一直向后匹配，直到末尾，到达末尾后，如果发现还有未执行的正则符号，则进行backtrack，从后向前搜索。
        - `/".+"/` --> This is <mark>"Jason" from "Microsoft"</mark> developer team.
            1. 首先找到第一个引号位置
            2. `.+`匹配一个字符`J`后，由greedy的特性，`.+`会一直向后匹配到整个字符串结尾`.`
            3. `.+`发现后面还有一个`"`未使用，因此从字符串末尾开始向前回溯，直到遇到第一个`"`
            
    - Lazy mode
        -  "repeat minimum number of times."
        - 在Quantifier后`*,+,?`加上`?`表示lazy求值
        - `/".+?"/` --> This is <mark>"Jason"</mark> from <mark>"Microsoft"</mark> developer team. 分析下这个例子
            1. 首先找到第一个引号位置
            2. `.+`匹配一个字符`J`，由于`.+`被声明成了lazy(跟随`?`)，因此匹配一次后，会将匹配控制权交给`?`之后的字符，即第二个引号
            3. 第二个引号匹配`a`不满足，将控制权交还给`.+`，`.+`成功匹配`a`后继续将控制权交给第二个引号
            4. 重复过程3，直到遇到下一个`"`，则第一次匹配完成
            5. 接着从`f`开始，继续重复步骤1
            6. 上述正则表达式也等价于`/"[^"]"+/`
    - One more example
        - `\d+ \d+` ---> <mark>123 456</mark>
        - `\d+ \d+?` ---> <mark>123 4</mark>56
- **Group**
    - Notion：`()`
    - 增加可读性, 不能出现在`[]`中
    - `/(xyz)+/` ---> <mark>xyzxyz</mark>abcdef1234<mark>xyz</mark>
    - `/([a-z]+[0-9]{0,3}) team/` ---> `abcds812team`,`zero team`

- **Alternation/Choice**
    - Notion: `|`
    - 和括号搭配使用, 允许嵌套
    - `/boy|gird/` ---> `boy`, `girl`
    - `/I think (China|Brazil|England) will win the world cup in 2050./`
    - `/((Jane|John) likes blue|(Tom|Jim) likes green)/`

### Matching Positions

- **Anchors**
    - Notion: `^,%`
        - 如果`^`出现在正则式开头，表示匹配以`^`后面字符开头，以`$`前一个字符结尾的串。Anchor字符用来确定匹配位置。
        - `^a[a-z]+a` -->`america`
        - `^[^a-zA-Z]+` -->`123`,`$#%`
        - `^[A-Z][a-zA-Z, ]+\.` --> 匹配英文每段的第一句话
        - `[A-Z][a-zA-Z, ]+\.$` --> 匹配英文每段的最后一句话
    - Notion: `\b,\B`
        - 用来分割word, word由`[a-zA-Z0-9_]`构成
            - `\b(\w)+\b\`匹配所有word
            - `\b[a-z]+\b`匹配文本中所有小写单词
            - `\b[A-Z]+\b`匹配文本中所有大写单词

### Advacnced Topics

- **Group & Backreference**
    - Notion: `(ab)(cd)\1\2`
    - 用标号`\1,\2`指代前面前面的group
        - `/(ab)(cd)\1\2/` --> `abcdabcd`
        - `/(Bruce) Wayne \1/` --> `Bruce Wayne Bruce`
        - `<([a-z][a-z0-9]*)>.*<\/\1>`

    - A python example

    ```python
    import re

    regex = r'^(\d{2})-(\w{3})\s(\d{9})\s([A-Z][\w .]+)\s(\d{4})'
    str = "01-SSN 123324134 S.Neis Steve 1997"
    m = re.match(regex,str)

    print(m.group(0)) # 01-SSN 123324134 S.Neis Steve 1997
    print(m.group(1)) #01
    print(m.group(2)) #SSN
    print(m.group(3)) #123324134
    print(m.group(4)) #S.Neis Steve
    print(m.group(5)) #1997
    ```
- **Non-Capturing Groups**
    - 可以令正则式成组，但是不引用它们
    - Notion: `?:`
        - `?` --> Give this group a different meaning
        - `:` --> The meaning is non-capturing
    - Advantages
        - Increase Speed
        - Make a room to caputures necessary groups
    - `/(ab)(?:cd)\1/` ---> `abcdab, [x]abcdabcd`
    - `/(?:red) looks (white)\1 to me` ---> `\1`指向的是`white`

### Assertions

- Look Ahead Assertions
    - **Positive Look Ahead Assertions**
        - Notion: `?=`
            - `?` --> Give this group a different meaning
            - `=` --> The meaning is PLAA
        - 用来匹配出现在某个字符前面的字符
            - `/long(?=island)` --> 只匹配出现在`island`之前的long
                - <mark>long</mark>island, `[x]longbeach`
            - `/a(?=b)/` --> <mark>a</mark>b
            - `\b[a-z][A-Z]+\b(?=\.)`--->匹配所有句号前面的单词
        - 另一种写法是`(?=)`在前，表示过滤条件
            - `\b(?=\w*ce)(?=\w*O)[a-z][A-Z]+\b(?=\.)`--->匹配所有句号前面的单词，并且是以`ce`结尾，`O`开头的
            - 电话号码正则式为`\d{3}-\d{3}-\d{4}`
                - `(?=^[0-6\-]+$)\d{3}-\d{3}-\d{4}`--> 匹配电话号码数字在`0-6`之间的
                - `(?=^[0-6\-]+$)(?=.*321)\d{3}-\d{3}-\d{4}`--> 匹配电话号码数字在`0-6`之间的，且有321的号码            
    - **Negative Look Ahead Assertions**
        - Notion: `?!`
        - 用来匹配不出现在某个字符后面的字符串
            - `/long(?!island)` --> 只匹配`long`后面不是`island`的字符串
                - <mark>long</mark>beach,<mark>long</mark>drive, `[x]longisland`


- Look Behind Assertions
    - **Positive Look Behind Assertions**
        - Notion: `?<=`
        - 用来匹配出现在某个字符后面的字符
            - `(?<=long)island` --> long<mark>island</mark>
    - **Negative Look Behind Assertions**
        - Notion: `?<!`
        - 用来匹配前面不是某个字符的字符串
            - `(?<!long)island` --> xx<mark>island</mark>
    - 所有能用Looke Behind Assertions 表示的情况，也能用Looke Ahead Assertions表示，因此

## Resource

- [Learn regex the easy way](https://github.com/zeeshanu/learn-regex)
- [精通正则表达式](http://shop.oreilly.com/product/9780596528126.do)
- [Regex101](https://regex101.com/)