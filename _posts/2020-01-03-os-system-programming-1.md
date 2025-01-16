---
layout: post
list_title: CS162 | Operating System and System Programming | Four Fundamental OS Concepts
title: Four Fundamental OS Concepts
categories: [System Programming, Operating System]
---

## Four Fundamental OS Concepts

- Thread: Execution context
    - Full describes program state
    - Program Counter, Registers, Execution Flags, Stack
- Address space (with or w/o translation)
    - Set of memory addresses accessible to program(for read or write)
    - May be distance from memory space of the physical machine (in which case programs operate in a virtual address space)
- Process: an instance of a running program
    - Protected Address Space + One or more threads
- Dual mode operation / protection
    - Only the "system" has the ability to access certain resources
    - Combined with translation, isolates programs from each other and OS from programs

## Thread of Control

- Thread: A single unique execution context
    - Program Counter, Registers, Execution Flags, Stack, Memory State
- A thread is <mark>executing</mark> on a processor(core) when it is <mark>resident in the processor registers
- Resident means: Registers hold the root state (context) of the thread:
    - Including program counter (PC), register & currently executing instruction
        - PC points at the next instruction in the memory
        - instructions are stored in the memory
    - Including intermediate values for ongoing computations
        - Can include actual values or pointers to values <mark>in memory</mark>
    - Stack pointer holds the address of the top of stack (which is in memory)
    - The rest is in memory
- A thread is <mark>suspended</mark>(not running) when its state <mark>is not</mark> loaded into the processor state pointing at some other thread
    - Program counter register is not pointing at the next instruction from this thread
    - Often: a copy of the last value for each register stored in memory

### What happens during program execution

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-01.png">

Execution sequence:
– Fetch Instruction at PC
– Decode
– Execute (possibly using registers)
– Write results to registers/memory
– PC = Next Instruction(PC)
– Repeat

### Illusion of Multiple Processors

- Assume a single processor(core). How do we provide the illusion of multiple processors?
    - Multiplex in time!
- Threads are <mark>virtual cores</mark>
- Contents of virtual core (thread):
    - program counter, stack pointer
    - Registers
- Where is "it" (the thread)?
    - On the real physical core (when running)
    - Saved in chunk of memory (when not running) - called the Thread Control Block(TCB)

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-02.png">

Consider:

- At T1: vCPU1 on real core, vCPU2 in memory
- At T2: vCPU2 on real core, vCPU1 in memory

During the context switch, what happened?

- OS saved PC, SP, ... in vCPU1's thread control block (memory)
- OS loaded PC, SP, ... from vCPU2's TCB, jumped to PC

- How long does the context switch take?

The order of a few microseconds. If the switching time is too long or too frequently, this will cause a thrashing situation.

- What triggered this switch?
    - Timer, voluntary yield, I/O, other things we will discuss

- What happens during the context switch

- There is a centralized cache (TCB) per core to store the thread context. The cache itself is typically in a physical space.
- Thread Control Block (TCB)
    - Holds contents of registers when thread not running
- Where are TCBs stored?
    - For now, in the kernel
    
## Address Space

The set of accessible addresses + state associated with them

- For 32-bit processor: `2^32 = 4 billion` addresses
- For 64-bit processor: `2^64 quadrillion` addresses

What happens when you read or write to an address?
- Perhaps acts like regular memory
- Perhaps ignores writes
- Perhaps causes I/O operation (memory-mapped I/O)
- Perhaps causes exception (fault)
- Communicates with another program
- ...

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-03.png">


## Paged Virtual Address Space

- What if we break the entire virtual address space into equal size chunks (i.e., pages) have a base for each?
- All pages same size, so easy to place each page in memory
- Hardware translates address using a <mark>page table</mark>
    - Each page has a separate base
    - The "bound" is the page size
    - Special hardware register stores pointer to page table
    - Treat memory as page size frames and put any page into any frame

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-04.png">


– Instruction address, load/store data address
    - Translated to a physical address (or Page Fault) through a Page Table by the hardware
    - Any Page of address space can be in any (page sized) frame in
memory
    – Or not-present (access generates a page fault)
    - Special register holds page table base address (of the process)

## Process

- Definition: execution environment with Restricted Rights
    - (Protected) Address Space with <makr>One or More Threads</makr>
    - Owns memory (address space)
    - Owns file descriptors, file system context, ...
    - Encapsulate one or more threads sharing process resources
- Application program executes as a process
    - Complex applications can fork/exec child processes
- Why processes
    - Protected applications from each other
    - OS protected from them
    - Processes provides memory protection
- Fundamental trade off between protection and efficiency
    - Communication easier within a process
    - Communication harder between processes

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-05.png">

- Threads encapsulate <mark>concurrency</mark> 
    - "Active" component
- Address spaces encapsulate protection: “Passive” part
    – Keeps buggy program from trashing the system
- Why have multiple threads per address space?
    - Parallelism: take advantage of actual hardware parallelism (e.g. multicore)
    - Concurrency: ease of handling I/O and other simultaneous events

### Protection and Isolation

- Why do need processes?
    - Reliability: Bugs can only overwrite memory of process they are in
    - Security and privacy: malicious or compromised process can't read or write other process' data
- Mechanisms:
    - Address translation: address space only contains its own data

## Dual Mode Operation

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2020/01/os-02-06.png"></div>
<div><img src="{{site.baseurl}}/assets/images/2020/01/os-02-07.png"></div>
</div>

- Hardware provides at least two modes
    - Kernel mode (or "supervisor" mode)
    - User mode
- Processes execute in <mark>user mode</mark>
    - To perform privileged actions, processes request services from the OS kernel
    - Carefully controlled transition from user to kernel mode
- Kernel executes in <mark>kernel mode</mark>
    - Performs privileged actions to support running processes
    - ...and configures hardware to properly protect them (e.g., address translation)
- Certain operations are prohibited when running in user mode
    - Changing the page table pointer, disabling interrupts, interacting directly w/ hardware, writing to kernel memory
- Carefully controlled transitions between user mode and kernel mode
    - System calls, interrupts, exceptions
    - User to Kernel transition sets system mode AND saves the user's PC (program counter)
    - Kernel to User transition clears system mode AND restores appropriate user PC

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-08.png">

## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides](https://sharif.edu/~kharrazi/courses/40424-012/)