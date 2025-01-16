---
layout: post
list_title: CS162 | Operating System | Thread and Processes
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

## Threads

### Motivation for Threads

- Operating System must handle multiple things at once (`MTAO`)
 - Processes, interrupts, background system maintenance
- Networked servers must handle MTAO
    - Multiple connections handled simultaneously
- Parallel programs must handle MTAO
    - To achieve better performance
- Programs with user interface often must handle MTAO
    - To achieve user responsiveness while doing computation
- Network and disk bound programs must handle MTAO
    - To hide network/disk latency
    - Sequence steps in access or communication

### Concurrency is not Parallelism

- Concurrency is about handling multiple things at once
- Parallelism is about doing multiple things simultaneously (usually with multiple CPU cores)
- Two threads on a single-core system
    - execute concurrently, but not in parallel

### Threads Mask I/O Latency

A thread is in one of the following three states:

- `RUNNING`
- `READY` - eligible to run, but not currently running
- `BLOCKED` - ineligible to run

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-02.png">

If a thread is waiting for an I/O to finish, the OS marks it as `BLOCKED`. Once the I/O finally finishes, the OS marks it as `READY`.

The OS(scheduler) maintains a ready queue for context switching. If there are no threads perform I/O, threads are put into the ready queue for execution:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-03.png">

If there is a thread performing a blocking I/O operation, the schedule will put that thread into a wait queue associated with I/O, once it finishes, the scheduler will move it back to the wait queue.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-04.png">

The blue thread in the above picture runs without interruption because the pink thread is doing the I/O operation and is removed from the READY queue, so there is no context switching during that time.

### `pthreads`

```c
int pthread_create(
    pthread_t* tidp,
    const pthread_attr_t* attr,
    void *(*start_routine)(void *), 
    void * arg);
```
- thread is created executing `start_routine` with `arg` as its sole argument
- return is implicit call to `pthread_exit`

```c
void pthread_exit(void *value_ptr);
```
- terminates the thread and makes `value_ptr` available to any successful join

```c
int pthread_join(pthread_t thread, void **value_ptr);
```
- Suspends execution of the calling thread until the target `thread` terminates
- On return with a non-NULL `value_ptr` the value passed to `pthread_exist()` by the terminating thread is made available in the location referenced by `value_ptr`.


What happens when `pthread_create(...)` is called in a process?

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-05.png">

`pthread_create` is just a C function that has special assembly code in it. The assembly code helps set up the registers in a way the kernel is going to recognize. And then it executes a special `trap` instruction, which is a way to jump into the kernel (think of it as an error). This will let us transition out of user mode into kernel mode due to the exception (`trap`).

Once we jump into the kernel, the kernel knows that this is the system call for creating a thread. It then gets the arguments, does the creation of the thread and returns the pointer (there is a special register to store return value).

Now we are back to the user mode. We grab the return value from the registers and do the rest of the work. This `pthread_create` function is not a normal function. It wraps the system call internally, but from users perspective, it just looks like a library C function.

It's worth noting that the system call can take thousands of cycles. The OS has to store a bunch of registers when going to the kernel and come out again. The translation from user mode to kernel mode is not cheap.

### fork join

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-06.png">

- Main thread creates(forks) collection of sub-threads passing them args to work on
- ... and then joins with them, collection results

## Thread states

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-07.png">

- State shared by all threads in process/address space
    - Content of memory (global variables, heap)
    - I/O state(file descriptors, network connections, etc)
- State "private" to each thread
    - Kept in TCB (Thread Control Block)
    - CPU registers(including, program counter)
    - Execution stack
        - Parameters, temporary vars
        - Return PCs are kept while called procedures are executing

### Execution Stacks

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-08.png">

- Two sets of CPU registers
- Two sets of stacks
- Issues:
    - How do we position stacks relative to each other?
    - What maximum size should we choose for the stacks?
    - What happens if threads violate this?
    - How might you catch violations?

### Thread Execution

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-09.png">

- Illusion: Infinite number of processors
- Reality: Threads execute with variable "speed"
    - Programs must be designed to work with any schedule




## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides](https://sharif.edu/~kharrazi/courses/40424-012/)