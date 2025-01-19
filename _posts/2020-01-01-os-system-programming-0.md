---
layout: post
list_title: CS162 | Operating System | Introduction
title: Introduction
categories: [System Programming, Operating System]
---

## Operating System

- Provide consistent abstractions to applications, even on different hardware
- Manage sharing of resources among multiple applications
- The key building blocks:
    - Processes
    - Threads, Concurrency, Scheduling, Coordination
    - Address Space
    - Protection, Isolation, Sharing, Security
    - Communication, Protocols
    - Persistent storage, transactions, consistency, resilience
    - Interfaces to all devices

- Illusionist
    -  Provide clean, easy to use abstractions of physical resources
        - Infinite memory, dedicated machine
        - Higher level objects: files, users, messages
        - Masking limitations, virtualization
- Referee
    – Manage sharing of resources, Protection, Isolation
        - Resource allocation, isolation, communication

### Syllabus

- OS Concepts: How to Navigate as a Systems Programmer!
    - Process, I/O, Networks and Virtual Machines
- Concurrency
    - Threads, scheduling, locks, deadlock, scalability, fairness
- Address Space
    - Virtual memory, address translation, protection, sharing
- File Systems
    - I/O devices, file objects, storage, naming, caching, performance, paging, transactions, databases
- Distributed Systems
    - Protocols, N-Tiers, RPC, NFS, DHTs, Consistency, Scalability, multicast
- Reliability & Security
    – Fault tolerance, protection, security
- Cloud Infrastructure

### Virtualizing the Machine

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-00-01.png">

- ISA
    - Application's "machine" is the process abstraction provided by the OS
    - Each running programing runs in its own process
    - Processes provide nicer interfaces than raw hardware
- Process
    - Address Space
    - One or more threads of control executing in that address space
    - Additional system state associated with it
        - Open files
        - Open sockets (network connection)
        - ...
- Threads
    – locus of control (PC)
    – Its registers (processor state when running)
    – And its “stack” (SP)
        - As required by programming language runtime


### Switching Processes

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-00-03.png">

Here we have two compiled programs running at the same time. They are isolated to each other. But we only have one processor. How can these two processes appear to be running at the same time? 

Each process has its own process descriptor in memory, and the OS will protect the memory. The CPU core will switch between these two processes by loading their process descriptors into the registers. Since the switching happens so quickly, it gives the delusion that multiple processes are running at the same time.

## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides-1](https://sharif.edu/~kharrazi/courses/40424-012/)
- [slides-2](https://github.com/Leo-Adventure/Berkeley-CS162-Operating-System/tree/main/Lecture/Slides)