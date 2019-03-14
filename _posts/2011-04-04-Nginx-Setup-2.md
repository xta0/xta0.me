---
updated: '2016-08-04'
layout: post
title: Nginx配置反向代理
list_title: 配置Nginx（二）| Nginx as Reverse Proxy 
categories: [Linux, Nginx,Backend]
---

## Reverse Proxy

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

