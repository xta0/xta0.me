---
updated: '2016-08-04'
layout: post
title: Nginx配置反向代理和负载均衡
list_title: 配置Nginx（二）| Security & Reverse Proxy
categories: [Linux, Nginx]
---

### HTTPs

- 登录[certbot](https://certbot.eff.org/)选择系统版本，安装certbot

```shell
sudo apt-get update
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:certbot/certbot
sudo apt-get update
sudo apt-get install python-certbot-nginx 
```

- 使用certbot安装SSL证书

```shell
% certbot --nginx
```
按提示修改nginx 配置文件

```ymal
#SSL configuration
listen 443 ssl default_server;
listen [::]:443 ssl default_server;
server_name xta0.me www.xta0.me;
ssl_certificate /etc/letsencrypt/live/xta0.me/fullchain.pem; # managed by Certbot
ssl_certificate_key /etc/letsencrypt/live/xta0.me/privkey.pem; # managed by Certbot
include /etc/letsencrypt/options-ssl-nginx.conf; #managed by Certbot
ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; #managed by Certbot
```

- 自动更新证书

```shell
%crontab -e
#编辑task，每日更新证书
@daily certbot renew

#确认修改
%crontab -l
```

### Basic Auth

如果某个URL只允许管理员访问，可以在Nginx中配置用户名和密码认证

```yaml
location / {
    auth_basic "Secure Area";
    auth_basic_user_file /etc/nginx/.htpasswd;
    try_files $uri $uri/ =404;
}
```
用户名和密码可以使用`apach2-utils`：

```shell
% sudo apt-get install apache2-utils
% htpasswd -c /etc/nginx/.htpasswd user_name
% cat /etc/nginx/.htpasswd
```
### Other Security Options

其它一些常用的安全配置

```yaml
http{
    #hide nginx version from HTTP header
    server_tokens off;
    
    server{
        #放置页面被其它网站用iframe嵌入
        add_header X-Frame-Options "SAMEORIGIN";
        #Cross-site scripting protection
        add_header X-XSS-Protection "1; mode-block";
    }
}
```

### Reverse Proxy

所谓Reverse Proxy是指对内部服务的代理，client的请求先到达Nginx，Nginx会根据配置文件来分发请求到内部的server上

```
client              server
                
    | domain           |
    | ------------>  nginx                     
    |                  | ----> index.html
    | domain/path1     |
    | ------------>    | -----> localhost: 9999 (nodejs server)   
    |                  |
    | domain/path2     |
    | ------------>    | -----> localhost: 8888 (python server)

```

例如，配置文件如下

```yaml
events{}
http{
    server{
        listen 8888;
        location /php{
            proxy_pass 'http://localhost:9999'
        }
        location /nginxorg{
            proxy_pass 'https://nginx.org'
        }
    }
}
```

另一个使用Ngix做反向代理的好处是，可以在request的header中插入字段传递给内部的server，也可以在response的header中插入字段，传递给client

```yaml
server{
    listen 8888;
    location /php{
        #set header on proxy request
        proxy_set_header proxied nginx;
        proxy_pass 'http://localhost:9999'
    }
    location /nginxorg{
        proxy_pass 'https://nginx.org'
    }
}
```





## Resources

- [Nginx Doc](https://nginx.org/en/docs/)
- [Nginx Configuration Example](https://www.nginx.com/resources/wiki/start/topics/examples/full/)
- [Nginx Config Pitfalls](https://www.nginx.com/resources/wiki/start/topics/tutorials/config_pitfalls/)
- [Nginx Tutorial DigistalOCean](https://www.digitalocean.com/community/tutorials/understanding-the-nginx-configuration-file-structure-and-configuration-contexts)
- [Nginx Resources Github](https://github.com/fcambus/nginx-resources)

