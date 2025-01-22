---
layout: post
list_title: CS162 | Operating System | Concurrency and Mutual Exclusion | Part 1
title:  Concurrency and Mutual Exclusion (Part 1)
categories: [System Programming, Operating System]
---

## Agenda

- Processes Multiplexing
- How does the OS provide concurrency through threads
    - Brief discussion of process/thread states and scheduling
    - High-level discussion of how stacks contribute to concurrency
- Introduce needs for synchronization
- Discussion of Locks and Semaphores

## Processes Multiplexing

### The Process Control Block(PCB)

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

<!-- <img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-01.png"> -->

### Context Switch

We use a single thread process as an illustration of how context switching work between two processes

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-02.png">

### Scheduling: All About Queues

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-03.png">

If we look inside the kernels, we have a <mark>ready queue</mark> for processes that are pending for execution. We also have a <mark>run queue</mark> for CPU. There are many other queues as well. What happens here is the PCB works their way from the ready queue to CPU. Scheduling is about deciding which process in the ready queue gets the CPU next. For example, when `fork` happens, the child process will be put into the ready queue waiting for CPU cycles.

### Ready Queue And Various I/O Device Queues

We have a lot of queues in the system. For each queue, it is structured as a linked list. They all have suspended processes in them. The scheduler is only interacting with the ready queue. <mark>The rest of the queue are interacted with the device drivers</mark>. For example, in the disk queue, once a process finishes the operation, it'll remove its PCB from the queue, and put it back to the ready queue.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-04.png">

### Scheduler

The scheduler is a simple loop

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-05.png">

- Scheduling: Mechanism for deciding which processes/threads receive the CPU
- Lots of different scheduling policies provide
    – Fairness or Real-time guarantees or Latency optimization or...

## The core of Concurrency: the Dispatch Loop

Conceptually, the scheduling loop of the operating system looks as follows

```c
// an inifinte loop
Loop {
    RunThread();
    ChooseNextThread();
    SaveStateOfCPU(curTCB); // TCB: thread control block
    LoadStateofCPU(newTCB);
}
```
### Running a thread

- How do I run a thread?
    - Load its state (registers, PC, stack pointer) into CPU
    - Load environment (virtual memory space, etc.)
    - Jump to the PC and start running

One thing to note is that both OS (which is managing threads) and the threads themselves run on the same CPU, so when the OS is running, the thread isn't and vice versa. We need to make sure we can transition properly between those. So once the OS loads the states and jumps to the PC, this essentially means the OS gives up the control of the CPU to a user level program. In old days, one user-level application crashes may cause the whole OS freeze. Fortunately, In modern operating system, we have memory protections, and we also have preemption possibilities through interrupts.

- How does the dispatcher get control back?    
    - Internal events: thread returns control voluntarily
    - External events: thread gets preempted

## Internal Events

Internal events are times when every thread is cooperating and voluntarily gives up its CPU. 

A good example is blocking I/O. When we make a system call, and we ask the operating system to do a `read()`, we give up the CPU (implicitly yielding), so the OS can work on your task by reading data from disk.

- Blocking on I/O
    - The act of requesting I/O implicitly yields the CPU
- Waiting on a "signal" from other thread
    - Thread asks to wait for thus yards the CPU
- Thread executes a yield()
    - Thread volunteers to give up CPU

```c
computePI() {
    while(TRUE) {
        ComputeNextDigit();
        yield(); //sleep is type of yield
    }
}
```

### Stack For Yielding Thread

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-06.png">

In the diagram above, when we call `yield`, we transition from user space to the kernel space. We also enter a kernel-level stack (the red ones). For every user-level thread(stack), there is a corresponding kernel-level stack, which remembers the address of `yield`, so when the kernel-level stack exists, it knows where to find the `yield`.

- How do we `run_new_thread`?

```c
run_new_thread() {
    nextThread = PickNextThread();
    switch(curThread, nextThread);
    ThreadHouseKeeping(); /* Do any cleanup */
}
```

- How does dispatcher switch to a new thread?
    - Save anything next thread may trash: PC, regs, stack pointer (the blue areas)
    - Maintain isolation for each thread

Consider the following code blocks:

```c
proc A() {
    B();
}

proc B() {
    while(TRUE) {
        yield();
    }
}
```

Suppose we have 2 threads: Threads `S` and `T`, and they are both running the same code above:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-07.png">

In this particular example, where there's only two threads in the system, and they both running the same code. What's going to happen is:

1. We are going down the stack for thread `S`
2. When hitting `yield`, we switch to `T`
3. We then go down the stack for `T` and come back for `S`

### Saving/Restoring state (often called “Context Switch)

The `switch` routine is quite simple. We just save all the registers states of the current thread to a TCB, and load the register states from TCB for the next thread.

```c
Switch(tCur,tNew) {
    /* Unload old thread */
    TCB[tCur].regs.r7 = CPU.r7;
    …
    TCB[tCur].regs.r0 = CPU.r0;
    TCB[tCur].regs.sp = CPU.sp; // sve the stack pointer
    TCB[tCur].regs.retpc = CPU.retpc; /*return addr*/
    /* Load and execute new thread */
    CPU.r7 = TCB[tNew].regs.r7;
    …
    CPU.r0 = TCB[tNew].regs.r0;
    CPU.sp = TCB[tNew].regs.sp; // load the stack pointer
    CPU.retpc = TCB[tNew].regs.retpc;
    return; /* Return to CPU.retpc */
}
```
Now with this in mind, let breakdown the above diagram and figure out the context switch in detail:

1. Let's say the CPU is running thread `S`. In the `switch` function, we update the CPU registers with the thread `T`'s state, meaning <mark>before we hit `return` in the current `switch` function(Thread `S`), we are already in thread `T`'s stack</mark>.
2. Now in thread `T`'s stack, the CPU reads the <mark>stack pointer, which points to `return` </mark>at the bottom of the `switch` function. Since we are already in thread `T`'s stack. So the `return` will exit the `switch` function, and we will pop up the stack for `switch` and `run_new_thread`.
3. Now we are transitioning back to the user-level stack(still in thread `T`'s stack). We pop up `yield` and hit `B(while)` again.
4. Next, thread `T` hits `yield` again due to the while loop, we then enter the kernel space and call `run_new_thread` and then `switch`. We are back to step #1

### More on `switch`

- TCB + stacks(user/kernel) contains complete restartable state of a thread. We can put it on any queue for later revival!
- Switching threads is much cheaper than switching processes
    - No need to change address space
- Some numbers from Linux:
    - Frequency of context switch: `10-100ms`
    - Switching between processes: `3-4 micro sec`.
    - Switching between threads: `100 ns`

### What happens when thread blocks on I/O

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-06-08.png">

- What happens when a thread requests a block of data from the file system?
    - User code invokes a system call
    - Read operation is initiated
    - Run new thread/switch
- Thread communication similar
    - Wait for Signal/Join
    - Networking

## External Events


## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides-1](https://sharif.edu/~kharrazi/courses/40424-012/)
- [slides-2](https://github.com/Leo-Adventure/Berkeley-CS162-Operating-System/tree/main/Lecture/Slides)
