---
layout: post
title: Ubuntu上部署Rails 
categories: 随笔
tag: Ruby

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






