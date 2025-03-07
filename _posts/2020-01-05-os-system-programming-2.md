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

Note that the translation map guarantees that each process maps their address spaces to different locations in the physical memory, <mark>preventing one process accessing the memory from the other process</mark>.

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

### Thread Execution and Race Condition

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-09.png">

- Illusion: Infinite number of processors
- Reality: Threads execute with variable "speed"
    - Programs must be designed to work with any schedule

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-10.png">

- Non-determinism:
    - Scheduler can run threads in <mark>any order</mark>
    - Scheduler can switch threads <mark>at any time</mark>
    - This can make testing very difficult
- Independent threads
    - No state shared with other threads
    - Deterministic, reproducible conditions

- Synchronization:
    - Coordination among threads, usually regarding shared data
- Mutual Exclusion:
    - Ensuring only one thread does a particular thing at a time (one thread excludes the others)
    - A type of synchronization
- Critical Section: 
    - Code exactly one thread can execute at once
    - Result of mutual exclusion
- Lock
    - An object only one thread can hold at a time
    - Provides mutual exclusion

- Locks provide two <mark>atomic</mark> operations
    - `lock.lock()` - wait until lock is free; then mark it as busy. If the lock is being hold by other threads, the current thread that tries to acquire it will be put to `sleep`.
    - `lock.unlock()` - mark lock as free
        - should only be called by a thread that currently holds the lock
- Semaphore:
    - A kind of <mark>generalized lock</mark>
    - First defined by Dijkstra in late 60s
    - Main synchronization primitive used in original UNIX
    - A Semaphore has a non-negative integer value and supports the following two operations:
        - `P() or down()`: atomic operation that waits for semaphore to become positive, then decrements it by 1
        - `V() or up()`: an atomic operation that increments the semaphore by 1, waiting up a waiting `P`, if any

### Two patterns of using Semaphores

 - mutual exclusion(like lock), also called a "binary semaphore" or "mutex"

 ```c    
    // the initial value of semaphore = 1;

    // all the subsequence threads that try to access the code (decrement the semaphore)
    // will be put at `sleep()`
    semaphore.down();
    // critical section goes here
    semaphore.up();
```

- Signaling other threads, e.g. ThreadJoin

```c
    // the initial value of semaphore = 0;

    // in the main thread
    ThreadJoin {
        semaphore.down(); // <----
    }                     //      |
                          //      |  
    // in another thread  //      |
    ThreadFinish {        //      |
        semaphore.up();   //-------
    }
```


## Processes

- How to manage process state?
    - How to create a process?
    - How to exit from a process?

- If processes are created by other processes, how does the first process start?
    - First process is started by the kernel
        - Often configured as an argument to the kernel before the kernel boots
        - Often called the `"init"` process
    - After this, all processes on the system are created by other processes

### Kernel APIs

Every process react to a bunch of signals

- `exit` - terminate a process
    - The `exist(0)` function
- `fork` - copy the current process
     - State of the original process duplicated in the child process
        - Address Space (memory), File descriptors, etc,...
- `exec` - change the program being run by the current process
- `wait` - wait for a process to finish

### `fork`

- `pid_t fork()` - copy the current process
    - new process has different pid
    - new process contains a single thread
- Return value from `fork()`
    - when `>0`:
        - running in parent process
        - return value is pid of new child process
    - when `=0`:
         - running in new <mark>child process</mark>
    - when `<0`:
        - Error, must handle
        - Running in the original process

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-11.png">

After `fork()` is called. The code after that will be executed by two processes at the same time - parent and child. This is because child inherits all the information from the parent, including the executing context of the current thread that is calling the `fork()`. Depending on the return value (`cpid`), we know if the current process is parent or child. In the above example, the red arrow points to the parent process. The green arrow points to the child process.

Don't call `fork()` in a multithreaded process. The other threads(the ones that didn't call `fork()`) just vanish.
- What if one of these threads was holding a lock
- What if one of these threads was in the middle of modifying a data structure
- No cleanup happens!

### `exec`

If we want the child process to execute something different, we can use the `exec`function

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-12.png">

In this case, the child process will immediately execute `ls -al` once it's created. It is safe to call `exec()` in the child process. It just replaces the entire address space.

### `wait`

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-13.png">

The `wait` function waits for the child process to finish. In the above example, once the child process exits with the status code `42`, the parent process will continue to get the pid from the process and continue the execution.

### POSIX Signals

- `kill` - send a signal (interrupt-like notification) to another process
- `sigaction` - set handlers for signals

The `sigaction` function allows you to add a handler in the user level space to capture signals thrown from the OS level. For each signal, there is a default handler defined by the system.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-03-14.png">

In the code above, we have an infinite `while` loop. The process won't exit until receives a `SIGINT` signal. The `SIGINT` can be triggered using `ctrl-c` from keyboard, which is going to terminate the process.

- Common POSIX Signals
    - SIGINT: ctrl-c
    - SIGTERM: default for the `kill` shell command
    - SIGSTP: ctr-z (default action: stop process)
    - SIGKILL/SIGSTOP: – terminate/stop process
        - Can't be changed or disabled with `sigaction`

## Summary

- Threads are the OS unit concurrency
     - Abstraction of a virtual CPU core
     - Can use `pthread_create`, etc, to manage threads within a process
     - They share data -> needs synchronization to avoid data races
- Processes consist of one or more threads in an address space
    - Abstraction of the machine: execution environment for a program
    - Can use `fork, exec`, etc to manage threads within a process


## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides-1](https://sharif.edu/~kharrazi/courses/40424-012/)
- [slides-2](https://github.com/Leo-Adventure/Berkeley-CS162-Operating-System/tree/main/Lecture/Slides)