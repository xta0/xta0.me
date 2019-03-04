---
layout: post
list_title: Regular Expression | 案例分析 | Case Study
title: 正则表达式的应用案例
categories: [Regular Expression]
---

### Names

- Rules
    - include a-z, A-Z
    - Must start with A-Z
    - No numbers
    - No symbols
- Regex

    ```shell
    ^(?!.*\s\s)(?!.*\.\.)(?!.*,,)[A-Z][a-zA-Z .,]{2,30}$
    ```

- Explaination
    1. `^$` -> 开始于结束符号
    2. `[A-Z][a-zA-Z .,]{2,30}` -> 大写字母开头，后面包含大小写字母，`.`号，`,`号以及名字长度为`{2,30}`个字符
    3. 增加过滤条件
        - `(?!.*\s\s)`, 过滤掉有两个空格及以上的名字
        - `(?!.*\.\.)`, 过滤掉有两个`.`及以上的名字
        - `(?!.*,,)`, 过滤掉有两个`,`及以上的名字

### Emails

- Rules
    - include `a-z, A-Z`
    - include `@`
- Regex
    ```shell
    (?!.*\.\.)([\w.\-#!$%&'*+\/=?^_{}|~]{1,35})@([\w.\-]+)\.([a-zA-Z]{2,10})
    ```
- Explaination
    1. `([\w.\-#!$%&'*+\/=?^_{}|~]{1,35})` ,匹配`@`之前的可能字符，限定长度为`{1,35}`
    2. `@`符号
    3. `([\w.\-]+)`, 匹配`@`之后，`.`之前可能的字符，可能有`xxx@xx.xx`的情况
    4. `([a-zA-Z]{2,10})`, 匹配`.`之后的字符，限定长度为`{2,10}`
    5. `(?!.*\.\.)` , 过滤条件，过滤掉有`..`情况

### URLs

- Rules
    - Must be Valid
- Regex
    
    ```shell
    ^(?:http|https|ftp):\/\/[a-zA-Z0-9_~:\-\/?#[\]@!$&'()*+,;=`^.%]+\.[a-zA-Z0-9_~:\-\/?#[\]@!$&'()*+,;=`^.%]+$
    ```
- Explaination
    1. `(?:http|https|ftp)` ---> 匹配三种中 的一种
    2. `:`
    3. `\/\/[a-zA-Z0-9_~:\-\/?#[\]@!$&'()*+,;=^.%]+` --> 匹配域名
    4. `.` --> 至少有一个`.`
    5. `\/\/[a-zA-Z0-9_~:\-\/?#[\]@!$&'()*+,;=`^.%]+` --> 匹配子域名

### IP Address

- Rules
    1. 250 - 255 --> `25[0-5]`
    2. 200 - 249 --> `2[0-4][0-9]`
    3. 100 - 199 --> `1[0-9][0-9]`
    4. 000 - 099 --> `0?[0-9][0-9]?`

- Regex

    ```
    ^(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|0?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|0?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|0?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|0?[0-9][0-9]?)$
    ```

### Datas

- Rule
    - Month/Day/Year

- Regex

    ```
    ^(?:0?[1-9]|1[012])([\/\-])(?:0?[1-9]|1[0-9]|2[0-9]|3[01])\1(?:20[0-4][0-9]|2050)$
    ```
- Explaination
    1. Month -->  `(0?[1-9]|1[0-2]])`
    2. Day --> `(0?[1-9]|1[0-9])|2[0-9]|3[01])`
    3. Year --> `(20[0-4][0-9]|2050)` , from 2000-2050

### Time

- Rules
    - 12 Hour Format
        - Hour: 1-12
        - Minutes: 0-59
        - Seconds: 0-59
    - 24 Hour Format
        - Hour: 0-24
        - Minutes: 0-59
        - Seconds: 0-59
- Regex
    - 12-hour format
    ```
    ^0?(?:[1-9]|1[012])\:(?:[0-5]?[0-9])(?:\:[0-5]?[0-9])?(?: am| pm| AM | PM)?$
    ```
    - 24-hour format
    ```
    ^(?:0?[0-9]|1[0-9]|2[0-3]):(?:[0-5]?[0-9])(?::[0-5]?[0-9])?(?: GMT| EST)?$
    ```

### Post Code

- Rules
    - No Rule

- Regex
    - Pakistan
    ```
    ^\d{5}$
    ```
    - India 
    ```
    ^\d{6}$
    ```
    - CANADA
    ```
    ^[A-Z]\d[A-Z] \d[A-Z]\d$`
    ```
    - USA
    ```
    ^\d{5}(?:-\d{4})?$`
    ```
    - UK
    ```
    ^(?:[A-Z]{1,2}\d{1,2} \d[A-Z]{2})|(?:[A-Z]{1,2}\d[A-Z] \d[A-Z]{2})$`
    ```

### Credit Cards

- Rule
    1. Visa
        - 16 digit
        - Start with `4`
        - `4xxxxxxxxxxxxxxx`
        - `4xxx-xxxx-xxxx-xxxx`
    2. MasterCard
        - 16 digit
        - start with `51 to 55`
        - `51xxxxxxxxxxxxxx`
        - `52xx-xxxx-xxxx-xxxx`
        - `53xx-xxxx-xxxx-xxxx`
        - `54xx-xxxx-xxxx-xxxx`
        - `55xx-xxxx-xxxx-xxxx`
    3. Discover
        - 16 digit
        - Start with `6011`
        - `6011xxxxxxxxxxxxx`
        - `6011-xxxx-xxxx-xxxx`
    4. American Express
        - 15 digit
        - start with `34 or 37`
        - 34xxxxxxxxxxxxx
        - 37xx-xxxxxx-xxxxx
    5. China Union Pay
        - 16 digit
        - Start with `62`
        - `62xxxxxxxxxxxxxx`
        - `62xx-xxxx-xxxx-xxxx`

- Regex
    - Visia & MasterCard & Discover & Union Pay
    ```
    ^(4\d{3}|5[1-5]\d{2}|62\d{2}|6011)[ -]?(\d{4}[ -]?)(\d{4}[ -]?)(\d{4})$
    ```
    - American Express
    ```
    ^(34|37\d{2})[ -]?(\d{6}[ -]?)(\d{5
    ```
    
    - Combine tow

    ```
    ^(?:(?:4\d{3}|5[1-5]\d{2}|6011|62\d{2})([\- ]?)\d{4}\1\d{4}\1\d{4})|(?:(?:3[47])\d{2}([\- ]?)\d{6}\2\d{5})$
    ```
    
### Password

- Rule
    1. length 8 to 15
    2. At least 1 digit
    3. At least 1 a-z
    4. At least 1 A-Z
    5. At least 1 symbol

- Regex
    
    ```
    ^(?=.*\d)(?=.*[A-Z])(?=.*[a-z])(?=.*[~!@#$%^&*{}\-[\];:<>'"?|]).{8,15}$
    ```
- Explaination
    - `^.{8,15}$` --> 匹配任意字符，限定长度
    - `(?=.*\d)` --> 过滤器，匹配的字符中至少包含一个数字
    - `(?=.*[A-Z])(?=.*[a-z])` -->过滤器，匹配的字符中至少包含一个大小写字母
    - `(?=.*[~!@#$%^&*{}\-[\];:<>'"?|])` -->过滤器，匹配字符中至少包含一个符号

## Resource

- [Learn regex the easy way](https://github.com/zeeshanu/learn-regex)
- [精通正则表达式](http://shop.oreilly.com/product/9780596528126.do)
- [Regex101](https://regex101.com/)