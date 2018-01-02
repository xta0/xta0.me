## Nginx Fundamentals

###About Nginx

- created in 2004
	- High performance, High Concurrency, Low Memory
	- webserver, reverse proxy

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

### Install nginx from source code


### 配置Nginx

- 两个名词
	- `context`：`nginx.conf`中的section，类似scope:
	```
	events {
		worker_connections 768;
		# multi_accept on;
	}
	```
	- `directive`：`ngix.conf`中的键值对，例如：`sendfile on;`
	
- 配置Virtual Host