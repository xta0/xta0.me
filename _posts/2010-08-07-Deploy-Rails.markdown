---
layout: post
title: Ubuntu上部署ROR 
---

<em>所有文章均为作者原创，转载请注明出处</em>


>更新于 2014/05/02

下面所有操作以非root账户登录,账户名为 : admin

##Install RVM

- 安装rvm: 
	- `curl -L https://get.rvm.io | bash -s stable`
	
- 载入rvm环境:
	- `source ~/.rvm/scripts/rvm`
	
- 验证rvm是否安装成功:
	- `rvm -v`
	
##Install Ruby

- 安装ruby:
	- `rvm install 2.1.1`
	
- 指定默认ruby版本:
	- `rvm 2.1.1 --default`
	
- 验证ruby是否安装成功
	- `ruby -v`
	- `gem -v`
	
- 查看rvm默认的gemset:
	- `rvm gemset list`
		 	
##Install Rails

- 安装rails:
	- `gem install rails` 

- 安装sqlite3:
	- `sudo apt-get install sqlite3`
	- `sudo apt-get install libsqlite3-dev`

##Install MySQL:

- 检查是否安装过sql: 	
	- `netstat -tap |grep mysql`
- 安装mysql: 
	- `sudo apt-get install mysql-server mysql-client` 
	- `sudo /etc/init.d/mysql start|stop|restart|reload|force-reload|status`

##Install Passenger

- 安装passenger：
	- 导入 Passenger 的密钥: 
		- `sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 561F9B9CAC40B2F7` 
	
	- 安装 apt 插件以支持 https :
		- `sudo apt-get install apt-transport-https ca-certificates`
	
	- 进入apt配置列表：
		- `/etc/apt/sources.list.d`
	
	- 创建passenger.list
		- 创建：`vim passenger.list`			
		- Ubuntu:12.04: deb https://oss-binaries.phusionpassenger.com/apt/passenger precise main 
		- Ubuntu:14.04: deb https://oss-binaries.phusionpassenger.com/apt/passenger trusty main		
	
	- 改写权限:
		- `sudo chown root: /etc/apt/sources.list.d/passenger.list` 
    		- `sudo chmod 600 /etc/apt/sources.list.d/passenger.list`
    	
	- 更新apt源:
		- `sudo apt-get update`
		
- 安装nginx
	- `sudo apt-get install nginx-extras passenger`
	
##Config nginx
	
- 编辑nginx的配置文件:
	- `sudo vim /etc/nginx/nginx.conf`	
	- 打开关于passenger的两段注释
	- 修改ruby路径:
		- 查看当前的ruby路径:`% which ruby` 
		- 替换nginx.conf的ruby路径: `passenger_ruby /home/admin/.rvm/wrappers/default/ruby;`
	
- 配置rails server:
	- `cd /etc/nginx/sites-enabled`
	- 干掉默认的配置文件: `sudo rm -rf default`
	- 创建新的配置文件:`sudo vim vizline`
	
```
http {

   server {
       listen       2000;
       server_name  123.456.78.9;

       root /home/admin/rails_app/vizline/public;
       passenger_enabled on;
       rails_env development;
   }
    
``` 

- link 到 site-available中:
	- ` ln -s /etc/nginx/sites-enabled/vizline /etc/nginx/sites-available/vizline`
- 更新nginx配置:
	- `sudo nginx -s reload`
- 启动nginx:
	- `sudo service nginx restart` 
- 停止nginx:	
	- `sudo service nginx stop`	
- 访问:
	- `http://123.456.78.9:2000`



## MacOS

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

	


