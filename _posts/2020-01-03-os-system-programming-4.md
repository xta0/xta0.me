---
layout: post
list_title: CS 162  | Operating and System Programming | Four Fundamental OS Concepts
title: Four Fundamental OS Concepts
categories: [System Programming, Operating System]
---

## Four Fundamental OS Concepts

- Thread: Execution context
    - Full describes program state
    - Program Counter, Registers, Execution Flags, Stack
- Address space (with or w/o translation)
    - Set of memory addresses accessible to program(for read or write)
    - May be distince from memory sapce of the physical machine (in which case programs operate in a virtual address space)
- Process: an instance of a running program
    - Protected Address Space + One or more threads
- Dual mode operation / protection
    - Only the "system" has the ability to access certain resources
    - Combined with translation, isolates programs from each other and OS from programs

## Thread of Control

- Thread: A single unique execution context
    - Program Counter, Registers, Execution Flags, Stack, Memory State
- A thread is <mark>exexcuting</mark> on a processor(core) when it is <mark>resident in the processor registers
- Resident means: Registers hold the root state (context) of the thread:
    - Including program counter (PC), register & currently executing instruction
        - PC points at the next instruction in the memory
        - instructures are stored in the memory
    - Including intermediate values for ongoing computations
        - Can include acutal values or pointers to values <mark>in memory</mark>
    - Stack pointer holds the address of the top of stack (which is in memory)
    - The rest is in memory
- A thread is <mark>suspended</mark>(not running) when its state <mark>is not</mark> loaded into the processor
    - Processor state pointing at some other thread
    - Program counter register is not pointing at the next instructure from this thread
    - Often: a copy of the last value for each register stored in memory

### What happens during program execution

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-01.png">

Execution sequence:
– Fetch Instruction at PC
– Decode
– Execute (possibly using registers)
– Write results to registers/mem
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
    - Holds contents of regiters when thread not running
- Where are TCBs stored?
    - For now, in the kernel
    
## Address Space

The set of accessible addresses + state associated with them

- For 32-bit processor: 2^32 = 4 billion addresses
- For 64-bit processor: 2^64 quardrillion addresses

What happens when you read or write to an address?
- Perhaps acts like regular memory
- Perhaps ignores writes
- Perhaps causes I/O opertion (memory-mapped I/O)
- Perhaps causes exception (fault)
- Communicates with another program
- ...

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-02-03.png">