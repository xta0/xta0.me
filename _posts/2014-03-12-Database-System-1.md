---
layout: post
list_title: Database System Part 1 
title: 数据库概述 | Core Dataabse Concept
---



### 附录

Setup PostgreSQL on mac

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