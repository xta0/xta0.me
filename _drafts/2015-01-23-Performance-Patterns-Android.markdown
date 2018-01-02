---

title: Rendering Technique

layout: post

---


##Before Reading

[UIView的渲染过程](http://akadealloc.github.io/blog/2012/10/22/UIView-Rendering.html)

##60 FPS

Android和iOS帧率都为60

##Both Needs GPU and CPU


##Understanding Overdraw

对于用户不可见的区域进行绘制会浪费GPU资源:

- 无用的背景

##Understanding VSYNC 

- 屏幕的刷新率：60HZ

- 帧率：GPU每秒能处理的帧数：60fps

- 这两者通常情况下是一致的，如果不一致，GPU会通过双缓冲来保证显示到屏幕上的图片是完整的：

![Alt text](/blog/images/2015/01/double_buffer.png)

- VSYNC(Vertical Synchronization):  用来同步Back buffer的数据到Frame buffer

![Alt text](/blog/images/2015/01/vsync.png)

- 如果GPU的帧率大于屏幕的刷新率，没有问题 

![Alt text](/blog/images/2015/01/vsync01.png)

- 如果GPU的帧率低于屏幕的刷新率，则会出问题




##Tools

- iOS: 需要使用Instrument : 

	- CPU: Time Profiler
		
	- GPU: Core Animation 
	
- Android:

	- CPU：TraceView
	
	![Alt text](/blog/images/2015/01/tools2.png)
	
	- GPU：机器自带On-Device Tools:
	
	![Alt text](/blog/images/2015/01/tools1.png)
	
