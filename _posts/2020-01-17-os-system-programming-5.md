---
layout: post
list_title: CS162 | Operating System | Concurrency and Mutual Exclusion | Part 1
title:  Concurrency and Mutual Exclusion (Part 1)
categories: [System Programming, Operating System]
---

## Agenda

- How does the OS provide concurrency through threads
    - Brief discussion of process/thread states and scheduling
    - High-level discussion of how stacks contribute to concurrency
- Introduce needs for synchronization
- Discussion of Locks and Semaphores

## Processes Multiplexing

### The Process Control Block

- A chunk of memory in the Kernel represents each process as a process control block (PCB)
    - Status (running, ready, blocked, …)
    - Register state (when not running)
    - Process ID (PID), User, Executable, Priority, …
    - Execution time, …
    - Memory space, translation, …
- Kernel <mark>Scheduler</mark> maintains a data structure containing the PCBs
    - Give out CPU to different processes (who gets the CPU to run themselves!)
    - This is a Policy Decision
- Give out non-CPU resources
    - Memory/IO
    - Another policy decision

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-01.png">

### Context Switch

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-02.png">

## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides-1](https://sharif.edu/~kharrazi/courses/40424-012/)
- [slides-2](https://github.com/Leo-Adventure/Berkeley-CS162-Operating-System/tree/main/Lecture/Slides)
