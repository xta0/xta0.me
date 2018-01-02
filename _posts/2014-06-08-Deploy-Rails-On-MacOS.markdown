---
layout: post
title: MacOS部署Rails
tag: Ruby
categories: 随笔
---

<em>所有文章均为作者原创，转载请注明出处</em>


##Install Rails

关于Rails在MacOS上的安装，可以参考以前的[文章](http://akadealloc.github.io/blog/2010/08/07/Deploy-Rails.html),或者Ruby China中的[文章](https://ruby-china.org/wiki/install_ruby_guide)


##Install Nginx and Passenger

在Mac下这部选择用gem安装：

- install passenger：

	`gem install passenger`

- install nginx:
	
	`passenger-install-nginx-module`
	
- 配置nginx:

	- nginx的配置文件在 `/usr/local/nginx/conf/nginx.conf`

	- link shell命令
	
		- `sudo ln -s /usr/local/nginx/sbin/nginx /usr/sbin/`
		
- nginx.conf:
	
```
worker_processes  1;

events
{
    worker_connections  1024;
}

http 
{
    passenger_root /Users/admin/.rvm/gems/ruby-2.1.1@global/gems/passenger-4.0.55;
    passenger_ruby /Users/admin/.rvm/gems/ruby-2.1.1/wrappers/ruby;

    include       mime.types;
    default_type  application/octet-stream;

    sendfile        on;
    keepalive_timeout  65;

    server 
    {
        listen       2000;
        server_name  localhost;

        root /Users/admin/rails/vizline/vizline/public;
        passenger_enabled on;
    }
}

```

- 启动nginx：

	- `sudo nginx`
	
- 停止nginx:

	- `sudo nginx -s stop`
	
- 修改nginx配置



##Install MySQL

- 删除[原来mysql文件](http://akadealloc.github.io/blog/2011/10/10/Tips-And-Tricks.html)

- `brew install mysql`

- 启动mysql:

	- `mysql.server start`

- 停止mysql:

	- `mysql.server stop`
	
- install mysql2

	- `gem install mysql2`
	
- 用root登录

	- `mysql -u root -p`

	

##配置Rail环境

- gem中引入mysql:

	- `gem mysql2`
	
- 修改database.yml:
	
```
default: &default
  adapter: mysql2
  encoding: utf8
  host: localhost
  username: root
  password: 

development:
  <<: *default
  database: db/development.sqlite3

test:
  <<: *default
  database: vizline_test


production:
  <<: *default
  database: vizline_production
  
```

##运行Rails


- 开发环境:

	- 使用WebBrick + Sqlite3

	- `rake db:create`
	
	- `rake db:migrate`
	
	- `rails s`
	
- 生产环境:
	
	- 使用Nginx+Passenger+MySQL 
	
	- `bundle exec rake db:create RAILS_ENV=production`
	
	- `bundle exec rake db:migrate RAILS_ENV=production`
	
	- 验证mysql数据库是否创建成功:
	
		- `mysql -u root -p`
		
		- `show databases;`
		
		- `use vizline_production;`
		
		- `show tables;`
		
	- 生成secret
	
		- `bundle exec rake secret RAILS_ENV=production`
		
		- 替换secret.yml中的加密串
		
	- 预编译asset
	
		- 在production.rb中：`config.assets.compile = true`
		
	- `sudo nginx -s stop`
	
	- `sudo nginx`

	- `localhost:2000`


##Further Reading:

- [Rails 实战圣经：进阶安装](http://ihower.tw/rails4/advanced-installation.html)
- [Rails 实战圣经：网站部署](https://ihower.tw/rails4/deployment.html)
- [Ruby China](https://ruby-china.org/wiki)

	