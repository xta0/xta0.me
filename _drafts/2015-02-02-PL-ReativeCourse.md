---
layout: post
title: Reactive Programming
categories: PL
tag: Reactive
---


##Chap1: What is Reactive Programming


###Why Reactive Programming?

- Popular: web service/mobile app

- Reactive Programming依赖Functional Programming

- Eric Meijer等大神会来


###Changing Requirements...

- 机器的硬件升级

- 新的硬件应该有新的Architechture:

	- event-driven
	
	- scalable
	
	- resilient
	
	- responsive
	
###Reactive
	
- **Reactive**: *"readly responsive to stimulus"*

	- event-driven: React to events
	
	- scalable: React to load
	
	- resilient: React to failures
	
	- responsive: React to users
	
	
- **Event-Driven**:

	- *Traditionally*: System are composed of multiple threads. which communicate with shared, sychronized state.
	
		- Strong coupling. hare to compose
		
	- *Now*: System are composed from loosely coupled event handlers
	
		- Event can be handled asynchronously, without blocking.
		
- **Scalable**: An application is *scalable* if it is able to be expaned according to its usage

	- scale up : make use of parallelism in multi-core systems
	
	- scale out: make use of multiple server nodes
	
	- Important for scalability: Minimize shared mutable state.
	
	- Important for scale out: Location transparency, resilience.  
	
- **Resilient**: An application is *resilient* if it can recover quickly from failures.

	- Failures can be:
		
		- software failures
		
		- hardware failures
		
		- connection failures
		
	Typically, resilience cannot be added as an afterthought; it needs to be part of the design from the beginning.
	
	Needed:
	
	- loos coupling
	
	- strong encapsulation of state
	
	- pervasive supervisor hierarhies


- **Responsive** : An application is responsive if it provides rich, real-time interaction with its users event under load and in the presence of failures.

Responsive app can be built on event-driven, scalable, resilience architechture. Still need careful attention to algorithms, system designs, back-pressure and many other details.


###Callbacks

Handling events is often done using callbacks. E.g: using Java observers:

```

class Counter extends ActionListener
{
	private var count = 0;
	
	button.addActionListener(this);
	
	def actionPerformed(e:ActionEvent): Unit = {count += 1;};


}

```

上面代码的问题：

- needs shared mutable state : 修改`count`值，产生side effect。而这种listener + callback的设计本身就很容易产生shared mutable state。

- cannot be composed

- leads quickly to "callback hell": 代码中大部分是callback代码，很难去track和理解


###How To Do Better

Use fundamental constructions from *functional programming*... to get ***composable*** event abstractions.

- Events are first class

- Events are often represented as messages

- Handles of events are also first-class

- Complex handlers can be composed from primitive ones.



###Contents of This Course

- Review of functional programming language

- An important class of functional patterns:*monads*

- Functional programs in a stateful world

- Abstracting over events:*futures*

- Abstracting over event streams: *observables*

- Message passing architechture: *actors*

- Scaling out: *distributed actors*


###Prerequisites

- Need a solid programming in functional programming

- Ideally, the "Principles of Functional Programming in Scala" class.

- If you know some other functional language, the switch should be easy.


##Chap2: Recap: Functions and Pattern Matching

 
###Recap: Case Classes

Case classes are Scala's preferred way to define complex data

**Example**: Representing JSON:

```
{
	"name" : "John",
	
	"address" : { "state":"NY", "code":10021 },
	
	"phoneNums":[ {"type":"home","num":"1234"}, {"type":"fax","num":"7890"}]

}
```
