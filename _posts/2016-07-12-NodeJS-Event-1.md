---
layout: post
list_title: Node.js中的Event(一)| Event in Node.js
title: Event
categories: [Javascript，nodejs]
---

### 两种Event

- System Events
    - C++ Core
        - libuv
    - File Operation
    - Network Opertaion

- Custom Events
    - Javascript Core
        - Event Emitter
    - Self-define event

JS层的event实际上是底层event的封装，JS自身没有Event这个类