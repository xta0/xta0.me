---
layout: post
title: Nginx基本配置
list_title: 配置Nginx（一） | Nginx Basic Configuration
categories: [Linux, Nginx,Backend]
---

### About Nginx

- created in 2004
	- High performance, High Concurrency, Low Memory
	- webserver
    - load balancer / reverse proxy

![](/assets/images/2011/04/nginx-1.png)

### Nginx vs Apache  
	
- Basic Architecture
	- Apache多个进程，每个进程起一个处理一个请求，
	- Nginx多个进程，每个进程可以实现并发处理多个请求，反向代理
- Resource Usage
	- Apache每个进程都及时处理静态资源的请求也需要加载php等语言环境，有一定overhead的损耗
	- Nginx对静态资源不需要加载语言环境 
- Performance
- Configuration
	- Nginx使用URI定位资源
	- Apache使用文件路径定位资源

### Install Nginx

- 使用Package Manger
    - `apt-get install nginx`
- 配置文件路径 
    - `/etc/nginx`
- 日志路径
    - `/var/log/nginx/error.log`

- Check Nginx Status
	- `ps aux | grep nginx`

- pid路径
    - `/var/run/nginx.pid`
    - `/run/ngix.pid`
- 启动/结束
    - `%sudo nginx` 
    - `%sudo service nginx start` 
    - `%sudo service nginx restart`
    - `%sudo nginx -s stop`

- 使用Systemd标准化Linux任务
    - 路径： `/lib/systemd/system/nginx.service`
    - 修改`nginx.service`为：[Nginx Systemd Service Config](https://www.nginx.com/resources/wiki/start/topics/examples/systemd/)
    - 修改完成后reload配置文件 
        - `%sudo systemctl daemon-reload`
- 后序所有操作均使用systemd
    - `%sudo systemctl status nginx`
    - `%sudo systemctl start nginx`
    - `%sudo systemctl stop nginx`

- 配置nginx自动启动
    - `%sudo systemctl enable nginx`

### 配置文件基本结构

- `directive`, `ngix.conf`中的键值对
    - 同名的子directive可以覆盖上一级的directive
    ```
    //注意分号
    sendfile on;
    ```
- `context`,`nginx.conf`中的section
     - context类似scope可被嵌套和继承父类的配置
     - global context用来配置所有的master

    ```yaml
    #global context
    user www www;
    error_log /var/log/nginx/error.log
    pid /run/niginx.pid

    events{
        worker_connection 4096;
    }

    #http context
    http {
        index.html index index.htm

        #server #1
        server{
            listen 80;
            server_name: domain.com;
            access_log /var/log/domain.access.log.main;
            root html;

            //location用来路由路径
            location / some_path{
                add_header header_name header_value;
            }
        }
        #server #2
        server{
            listen 455;
            
        }
    }
    ```




- 配置一个基本的Static Website

    ```yaml
    events{}
    http{
        #inlcude mime types for front-end
        include /etc/nginx/mime.types;
        #每个server host用一个server来表示
        server{
            listen 80;
            domain abc.com www.abc.com
            root /usr/home/xx/site/
        }
    }
    ```

### Location Blocks

```yaml
server{

    #prefix match
    #match: domain/greet/, domain/greeting/, domain/greet/more 
    location /greet{
        //处理个别路径的请求
        return 200 "Hello From Nginx From /Greet"
    }

    #Exact match
    #match: domain/greet/
    location =/greet{
        //处理个别路径的请求
        return 200 "Hello From Nginx From /Greet"
    }

    #Regular Expression match, case sensitave
    #match: domain/greet0/, domain/greet2/, ...domain/greet9/
    location ~/greet[0-9]{
        //处理个别路径的请求
        return 200 "Hello From Nginx From /Greet"
    }

    #Regular Expression match, case insensitave
    #match: domain/greet0/, domain/greet2/, ...domain/greet9/
    location ~*/greet[0-9]{
        //处理个别路径的请求
        return 200 "Hello From Nginx From /Greet"
    }
}
```

Nginx对location的匹配规则为，优先级由低到高如下

1. Exact Match `= uri`
2. Preferential Prefix Match `^~uri`
3. REGEX Match `~* uri`
4. Prefix Match `uri`

对所有的子路径，可以配置

```yaml
location / {
    #如果当前路径不存在，指向404
    try_files $uri $uri/ =404
}
```

### 模板语言

Nginx配置中可以使用两类变量

- Configuration Variables
    - `set $var 'somethig'`
- NGINX Module Variables
    - `$http, $uri, $args`
    - [Built-in variables](http://nginx.org/en/docs/varindex.html)
    
- 常用的变量
    - `$host`: `domain.com`
    - `$uri` : `/inspect`

- 逻辑控制  

```yaml
if ($arg_apikey != 1234 ){
    return 401 "Incorrect API key"
}
#使用自定义变量
set $weekend 'No';
#正则匹配
if( $date_local ~ 'Saturday|Sunday'){
    set $weekend 'Yes';
}
location /is_weekend{
    return 200 $weekend;
}
```

### Redirect

- `rewrite pattern URI`

rewrite url进行重新路由

```yaml
rewrite ^/usr/w+ /greet;  #将^/usr/w+ 路由到/greet
```

- `return status URI`
    - 成功返回字符串：`return 200 some_string`
    - 重定向返回路径：`return 301 https://$host$request_uri`
	
    ```yaml
    #HTTPs 重定向
    server{
		listen 80;
		server_name xta0.me www.xta0.me;
			return 301 https://$host$request_uri;
	}
    ```

### Logs

查看log路径`ls -al /var/log/nginx`，可以根据不同的server配置不同的log

```yaml
location /secure {
    access_log /var/log/nginx/secure.access.log;
    #关闭log
    access_log off;
}
```

### Worker Process

Nginx启动后Master Process会启动一个worker process来处理HTTP请求

```
//master process
//pid #30704
root 30704 0.0 0.1 125108 1492 ? Ss 10:59 0:00 nginx: master process 

//worker process
//pid #30706
www-data 30706 0.0 0.3 125464  3288 ? S 10:59 0:00 nginx: worker process
```
我们可以配置worker process的数量, 在配置文件的global context中指定

```ymal
work_processers: 2 #产生2个子进程
```
Nginx的设计是多进程，每个CPU一个进程，在一个进程内增加多个worker_processer并不能提高效率，因为并未实现真正的并发，只是CPU轮训。可通过下面命令查看CPU状态

```
% nproc
% lscpu
```

Nginx提供了一种自动配置worker processers的directive，当CPU个数增加时，Nginx的master会自动增加worker_processers的个数

```ymal
worker_processers: auto;
```

对于每个worker_processer，可以配置其最大并发连接数，该数值和系统能力相关，可使用`%ulimit -n`查看

```ymal
events{
    worker_connections: 1024;
}
```

### Buffers & Timeouts

Buffer是Nginx用来缓存Response或者Request的内存区，配置如下:

```ymal
# Buffer size for POST submissions
client_body_buffer_size 10K;
client_max_body_size 8m;

# Buffer size for Headers
client_header_buffer_size 1k;

# Max time to receive client headers/body
client_body_timeout 12;
client_header_timeout 12;

# Max time to keep a connection open for
keepalive_timeout 15;

# Max time for the client accept/receive a response
send_timeout 10;

# Skip buffering for static files
sendfile on;

# Optimise sendfile packets
tcp_nopush on;
```

### Headers & Epires

Nginx可以通过配置文件向HTTP response的Header中插入字段，一个常用的配置是对静态资源做浏览器级别的缓存，减少对server的频繁调用

```yaml
#正则匹配图片请求
location  ~* \.(jpg|png|jpeg){
    access_log off;
    add_haeder Cache-Control public;
    add_header Pragma public;
    add_header Vary Accept-Encoding;
    expires 60m; #60 mins
}
```

### Gzip

打开HTTP gzip module

```yaml
gzip on;
#set to 3 or 4
gzip_comp_level 3; 
gzip_types text/css
gzip_types text/javascript
```

### HTTP2

HTTP2支持

1. Binary Protol 传输二进制而不是plain/text
2. Compressed Header 头部压缩，省空间
3. Persistent Connections 短链改长链，减少频繁建立短连接的开销
4. Multiplex Streaming 多路复用，合并资源请求
5. Server Push 支持push

## Resources

- [Nginx Doc](https://nginx.org/en/docs/)
- [Nginx Configuration Example](https://www.nginx.com/resources/wiki/start/topics/examples/full/)
- [Nginx Config Pitfalls](https://www.nginx.com/resources/wiki/start/topics/tutorials/config_pitfalls/)
- [Nginx Tutorial DigistalOCean](https://www.digitalocean.com/community/tutorials/understanding-the-nginx-configuration-file-structure-and-configuration-contexts)
- [Nginx Resources Github](https://github.com/fcambus/nginx-resources)

