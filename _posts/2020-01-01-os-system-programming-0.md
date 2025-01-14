---
layout: post
list_title: CS 162  | Operating and System Programming | Introduction
title: Introduction
categories: [System Programming, Operating System]
---

## Operating System

- Provide consistent abstractions to applications, even on different hardware
- Manage sharing of resources amoung multiple applications
- The key building blocks:
    - Processes
    - Threads, Concurrency, Scheduling, Coordination
    - Address Spacee
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

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-00-02.png">



## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)