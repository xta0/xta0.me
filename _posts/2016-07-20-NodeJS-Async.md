---
layout: post
updated: "2017-10-13"
list_title: 理解Node.js | Asynchronous I/O
title: Asynchronous
categories: [Javascript，nodejs]
---

异步通信是作为Server最基本的能力，Node中的异步通信主要是通过libuv完成的，libuv是一个开源的，基于C的，跨平台的，事件驱动的异步通信引擎。简单的说，它是对各个操作系统底层异步IO通信API的一个封装，它在UNIX系统上使用的是libev（0.9.0版本改成自己实现`epoll`），在Windows上使用IOCP。除了node.js以外，包括Mozilla的Rust编程语言，和许多的语言底层都也使用了libuv。

### 异步模型

Node中的异步模型，主要由下面这三部分构成

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2016/07/node-1.png">

V8自身不具备异步能力，它单独跑在一个线程中执行JS代码，libuv除了和OS对接异步IO的API之外，内部还有一个event loop，用来监听事件。翻看起源码，在`unix/core.c`中可看到`event loop`的实现

```c
int uv_run(uv_loop_t* loop, uv_run_mode mode) {
    //...
    uv__io_poll(loop, timeout);
    //...
}
void uv__io_poll(uv_loop_t* loop, int timeout) {
    //...
    for (;;) {
        //...
        nfds = poll(loop->poll_fds, (nfds_t)loop->poll_fds_used, timeout);
        //...
    }
}
```

上述代码中可以看到，在UNIX下，libuv使用的还是UNX API`poll`来监听事件


### Buffers & Stream

Node使用`Buffer`类来做读写缓冲，`Buffer`是一个全局类，使用的时候不需`require`，它的底层是使用C++编写的，JS封装了调用的API。使用`Buffer`可以帮助我们做二进制数据的序列化和反序列化。

```javascript
var buf = new Buffer('Hello','utf8')
console.log(buf) //<Buffer 48 65 6c 6c 6f>
```
上述代码将字符串序列化成二进制，Node中的字符集使用Unicode，编码标准为UTF-8。也可以将二进制数据反序列化成string和JSON对象

```javascript
buf.toString(); //Hello
buf.toJSON(); //{ type: 'Buffer', data: [ 72, 101, 108, 108, 111 ] }
```
JavaScript中没有专门的类处理二进制数据，ES6中提供了`TypedArray`

```javascript
var buffer = new ArrayBuffer(8); //创建一个8x8=64bit的缓存
var view = new Int32Array(buffer); //创建一个TypedArray，每个元素长度为32bit
```
上述代码创建了一个`TypedArray`并为其分配了`buffer`大小的内存空间，由于一个元素是32bit，该数组只能存放2个元素

```javascript
view[0] = 10;
view[1] = 20;
console.log(view) //Int32Array [ 10, 20 ]
```

`Stream`用`Buffer`做为缓冲器，提供异步读写数据流的API，`Stream`继承了`EventEmiiter`的API，可以向使用Event意向使用Stream，例如异步读文件

```javascript
var readable = fs.createReadStream(__dirname+'/20k.txt',{encoding: 'utf8', highWaterMark: 10*1024})

readable.on('data',(chunk)=>{
	console.log(chunk.length) //10240,10240
})
```
`ReadStream`可以配置编码方式和每次load的chunk大小，上面例子中读取一个20k大小的文件，每次读取1k并回调，因此回调函数会触发两次

`Stream`的另一个用法是用`Pipe`串联多个stream，这种方式以数据流为思考模型，简化了很多冗余的细节，例如我们压缩数据为例

```javascript
var readable = fs.createReadStream(__dirname+'/20k.txt',{encoding: 'utf8', highWaterMark: 10*1024})
var writable = fs.createWriteStream(__dirname+'/20k.txt.gz')
var zlib = require('zlib')
var gzip = zlib.createGzip()
readable.pipe(gzip).pipe(writable)
```
上面例子中一共有三段数据流，分别是`readable`,`gzip`和`writable`，通过`pipe`将三段数据流串联起来，首先读取文件，然后进行gzip压缩，最后写入文件。

### A Simple WebServer

```javascript
var http = require('http')

http.createServer(function(req,res){
    res.writeHead(200,{'Content-Type':'text/plain'});
    res.end('Hello world\n');
}).listen(1337,'127.0.0.1');
```


### Resource

- [libuv](http://libuv.org/)
- [libev](https://github.com/enki/libev)