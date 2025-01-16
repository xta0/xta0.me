---
layout: post
list_title: CS162 | Operating System and System Programming | Thread and Processes
title: Thread and Processes
categories: [System Programming, Operating System]
---

## Goals

- What threads are and what they are not
- Why threads are useful
- How to write a program using threads
- Alternatives to using threads

### Recap: Threads and Process

- Thread: <mark>Execution Context</mark>
    - Fully describes program state
    - Program Counter, Registers, Execution Flags, Stack
- Address space (with or w/o <mark>translation</mark>)
    - Set of memory addresses accessible to program (for read or write)
    - May be distinct from memory space of the physical machine(in which case programs operate in a virtual address space)
- Process: an instance of a running program
    - An execution environment with restricted rights
    - One or more threads executing in a (protected) Address Space
        - Encapsulates one or more threads sharing process resource
    - Owns memory (mapped pages), file descriptors, network connections, file system context...
    - <mark>In modern OS, anything that runs outside the kernel runs in a process</mark>
- Dual mode operation / Protection
    - Only the “system” has the ability to access certain resource
    - Combined with translation, isolates programs from each other and the OS from programs

### Recap: Illusion of Multiple Processors

- Threads are <mark>virtual cores</mark>
- Multiple threads - <mark>Multiplex</mark> hardware in time
- A thread is <mark>executing</mark> on a processor when it is resident in that processor's registers
- Each virtual core(thread) has:
    - Program counter (PC), stack pointer(SP)
    - Registers - both integer and floating point
- Where is the thread?
    - On the real(physical) core or
    - Saved in chunk of memory - called the <mark>Thread Control Block (TCB)</mark>

### Recap: Memory Address Translation through Page Table

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-01.png">

Note that the translation map guarantees that each process maps their address spaces to different locations in the physical memory, preventing one process accessing the memory from the other process.

### Motivation for Threads

- Operating System must handle multiple things at once (`MTAO`)
 - Processes, interrupts, background system maintenance
- Networked servers must handle MTAO
    - Multiple connections handled simultaneously
- Parallel programs must handle MTAO
    - To achieve better performance



## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides](https://sharif.edu/~kharrazi/courses/40424-012/)