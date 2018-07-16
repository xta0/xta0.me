---
layout: post
list_title: Database System Part 1 
title: 数据库概述 | Core Database Concept
mathjax: true
---

### Relational Algebra

Relational Algebra是一种formal language，是SQL底层实现。对一个关系型结构的查询结果仍旧是一个关系型结构

假设有一个大学数据库管理系统，其三个实体为

```
college(cName, state, enrollment)
student(sID sName, GPA, sizeHS)
Apply(sID cName, major, description)
```

<div style=" content:''; display: table; clear:both; height=0">
    <div style="float:left">
        <table>
                <caption> college</caption>
                <thead>
                    <tr><th>cName</th><th>state</th><th>enr</th></tr>
                </thead>
                <tbody>
                    <tr> <td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td></tr>
                </tbody>
        </table> 
    </div>
    <div style="float:left;margin-left:40px;">
        <table>
                <caption> student</caption>
                <thead>
                    <tr><th>sID</th><th>sName</th><th>GPA</th><th>HS</th></tr>
                </thead>
                <tbody>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                </tbody>
        </table>
    </div>
    <div style="float:left;margin-left:40px;">
        <table>
                <caption> Apply</caption>
                <thead>
                    <tr><th>sID</th><th>cName</th><th>major</th><th>dec</th></tr>
                </thead>
                <tbody>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                    <tr> <td></td><td></td><td></td><td></td></tr>
                </tbody>
        </table>
    </div>
</div>

### Operators

- Select

Select用于获取一系列行数据，使用 $\sigma$符号表示，通用形式为

$$
\sigma_{cond} \quad Ralation \thinspace Name
$$

1. Students with GPA>3.7
    - $ \sigma_{GPA>3.7} \quad student $
2. Students with GPA>3.7 and HS < 1000
    - $ \sigma_{GPA>3.7 \^ \thinspace HS \thinspace < 1000} \quad student $
3. Applications to Stanford CS major
    - $ \sigma_{cname ='stanford' \thinspace ^ \thinspace major='cs'} \quad student $

- Project

Project用来获取列数据，使用$\pi$表示

1. filter

2. slice

3. combine




### 附录

- Setup PostgreSQL + admin on MacOS

1. [install postgreSQL](https://www.postgresql.org/download/macosx/)
2. [install postgreSQL admin](https://www.pgadmin.org/download/)

- Setup PostgreSQL.app on MacOS

1. Download PostgreSQL from the official website: http://postgresapp.com/
2. Double-click on the .dmg file and copy it to the applications folder
3. Go to the launchpad and you should see the postgresql icon
4. Click on the PostgreSQL application and click initialize to start the SQL server
5. Open the terminal and execute the following line to configure the path

```shell
sudo mkdir -p /etc/paths.d && echo /Applications/Postgres.app/Contents/Versions/latest/bin | sudo tee /etc/paths.d/postgresapp
```

6. Now, close the terminal and restart it again
7. Enter the following command to enter the interactive shell for executing postgres queries: 

```shell
➜  ~ psql -U postgres
psql (10.4)
Type "help" for help.

# list all databases
postgres=# \list 

                                  List of databases
   Name    |  Owner   | Encoding |   Collate   |    Ctype    |   Access privileges
-----------+----------+----------+-------------+-------------+-----------------------
 postgres  | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 template0 | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | =c/postgres          +
           |          |          |             |             | postgres=CTc/postgres
 template1 | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | =c/postgres          +
           |          |          |             |             | postgres=CTc/postgres
```


