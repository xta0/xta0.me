---
layout: post
list_title: Database System Part 2 | SQL Syntax
title: SQL语法 | SQL Syntax
mathjax: true
---


### SELECT

- **SELECT**

```SQL
SELECT column1, column2,...FROM table_name;
```

查询包含某列所有的行数据（包含重复信息）

```SQL
SELECT * FROM actors;
SELECT actor_id, first_name FROM actors;
SELECT actor_id FROM actors;
```

- **SELECT DISTINCT**

```SQL
SELECT DISTINCT column1, column2,...FROM table_name;
```
查询包含某列的所有行数据（去重）

```SQL
<!-- 查询所有电影发布年限的取值 -->
SELECT DISTINCT release_year FROM film;
<!-- 查询所有电影分级的取值 -->
SELECT DISTINCT rating FROM film;  
```

- **SELECT WHERE**

```SQL
SELECT column_1, column_2,...column_n FROM table_name WHERE conditions
```

根据条件查询某条行数据，过滤条件不一定要包含查询的属性。PostgreSQL提供一系列的条件判断操作符：

|--|--|
|OPERATOR | DESCRIPTION|
| = | Equal|
| > | Greater than|
| < | Less than | 
| >= | Greater than or equal | 
| <= | Less than or equal | 
| <> or != | Not equal |
|AND|Logical operator AND|
|OR| logicl operator OR | 

```SQL
SELECT last_name, first_name 
FROM customer 
WHERE first_name = 'Jamie';

SELECT last_name, first_name 
FROM customer 
WHERE first_name = 'Jamie' AND last_name = 'Rice';

SELECT customer_id, amount,payment_date 
FROM payment
WHERE amount<=1 OR amount>=8;

<!-- 过滤条件不一定包含查询的信息 -->
SELECT email 
FROM customer
WHERE first_name = 'Jared' AND last_name='Thomas';
```

- **COUNT**

```SQL
SELECT COUNT(*) FROM table;
SELECt COUNT(colunm) FROM table;
SELECT COUNT(DISTINCT colunmn) FROM table;
```
COUNT返回SELECT查询结果的数量

```SQL
SELECT COUNT(*) FROM payment;
SELECT COUNT(DISTINCT amount) FROM payment;
```
- **LIMIT**

查询包含某列的有限条的行数据（包含重复信息），顺序从上到下

```SQL
SELECT * FROM customer LIMIT 5;
```

- **ORDER BY**

```SQL
SELECT column_1, column_2 FROM table_name
ORDER BY column_3 ASC/DESC, column4 ASC/DESC;
```

对查询得到的行数据进行某个属性进行升序或者降序的排序，默认为升序排列

```SQL
<!-- 对first_name升序排列 -->
SELECT first_name, last_name FROM customer 
ORDER BY first_name ASC;
<!-- 对last_name进行降序排列后，再对first_name进行升序排雷 -->
SELECT first_name, last_name FROM customer 
ORDER BY last_name DESC, first_name ASC;

<!-- 对所有first_name='Kelly'的last_name降序 -->
SELECT first_name, last_name FROM customer 
WHERE first_name = 'Kelly'
ORDER BY last_name DESC

<!--  -->
SELECT customer_id,amount FROM payment
ORDER BY amount DESC
LIMIT 10;
```
> PostgreSQL允许ORDER BY的字段和SELECT的字段不相同，例如 SELECT first_name FROM user ORDER BY last_name；其它数据库不允许这种情况


- **BETWEEN**

```SQL
value BETWEEN A AND B
value NOT BETWEEN A AND B
```

通常和WHERE一起使用，用来做过滤条件，查询具备某个列属性的<mark>闭区间</mark>值的所有行数据,

```SQL
SELECT customer_id, amount FROM payment 
WHERE amount BETWEEN 8 AND 9;

SELECT customer_id, amount FROM payment 
WHERE amount NOT BETWEEN 8 AND 9;

SELECT amount,payment_date FROM payment 
WHERE payment_date BETWEEN '2007-02-07' AND '2007-02-15';
```

- **IN**

```SQL
value IN (value1, value2, ...)
value NOT IN (value1, value2, ...)
value IN (SELECT value FROM table_name)
value NOT IN (SELECT value FROM table_name)
```

通常和WHERE一起使用，用来做过滤条件，相当于OR，对于过滤值较多的情况，使用IN写法更简洁

```SQL
SELECT customer_id,rental_id,return_date
FROM rental
WHERE customer_id IN (1,2)
ORDER BY return_date DESC;

SELECT customer_id,rental_id,return_date
FROM rental
WHERE customer_id NOT IN (7,12,19)
ORDER BY return_date DESC;
```

- **LIKE**

```SQL
SELECT column1, column2 FROM table_name
WHERE column1 LIKE 'str%';
```
通常和WHERE一起使用，用来模糊查询，对查询条件进行模式匹配

1. `%`用来匹配任意序列
    - `Jen%`开头为Jen的任意序列
    - `y%`以y为结尾的任意序列
    - `%er%`以er为中间字符的任意序列
2. `_`用来匹配单个字符
    - `%_en%%`以任意字符开头+`en`+任意字符

```SQL
<!-- 返回Jen开头的名字 -->
SELECT first_name, last_name FROM customer
WHERE first_name LIKE 'Jen%';
```

> postgreSQL提供ILIKE，可以忽略字符大小写

### GOURP BUY

- **AVG/MAX/MIN/SUM**

```SQL
SELECT ROUND(AVG(amount),5) FROM payment;
SELECT MIN(amount) FROM payment;
SELECT MAX(amount) FROM payment;
SELECT SUM(amount) FROM payment;
```

- **GROUP BY**

GROUP BY将查询结果根据某种规则分成多个group，每个group可以使用上面提到的函数来进行计算。

1. 单独使用GROUP BY相当于DISTINCT

```SQL
SELECT customer_id
FROM payment
GROUP BY customer_id;
```

上述例子相当于按照`customer_id`查询所有记录，然后按照`customer_id`分组，因此得到的是`customer_id`去重后的记录

2. 和Aggregate函数一起使用

```SQL
SELECT customer_id, SUM(amount)
FROM payment
GROUP BY customer_id
ORDER BY SUM(amount);
```
将所有行数据按照`customer_id`进行分组，同一个`customer_id`可能有多条记录，相同`customer_id`的记录分在同一组，分组后对每组数据的`amount`字段进行求和。上述SQL的实际意义是统计每个用户消费的总金额，从高到低排列。

```SQL
SELECT staff_id, COUNT(payment_id),SUM(amount)
FROM payment
GROUP BY staff_id
ORDER BY COUNT(payment_id) DESC;
```
上述例子对所有员工中，处理交易次数的统计以及处理交易金额的统计，并按照交易次数从高到低排列

- **HAVING**

```SQL
SELECT column1, aggregate_function(column2)
FROM table_name
GROUP BY colunm1
HAVING condition;
```

HAVING通常和GROUP一起使用，来过滤不满足条件的结果，类似于WHERE。和WHERE的区别在于，HAVING是在GROUP BY之后对分组后的每组数据进行过滤，WHERE实在GROUP_BY之前，对分组前的每组数据进行过滤。

```SQL
SELECT customer_id, SUM(amount)
FROM payment
GROUP BY customer_id
HAVING SUM(amount) > 200;
```
在统计用户消费的基础上增加消费金额大于200的条件

```SQL
SELECT rating,AVG(rental_rate)
FROM film
WHERE rating IN('R','G','PG')
GROUP BY rating
HAVING AVG(rental_rate)<3;
```
GROUP之前应用WHERE



### Schema
![](/assets/images/2011/03/sql-sample-schema.png)

### Cheat Sheet

![](/assets/images/2011/03/sql-cheatsheet.png)