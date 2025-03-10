---
layout: post
list_title: CS162 | Operating System | Semaphores, Lock Implementation, Atomic Instructions
title:  Semaphores, Lock Implementation, Atomic Instructions
categories: [System Programming, Operating System]
---

## Semaphores

- Semaphores are a kind of generalized lock
    - First defined by Dijkstra in late 60s
    - Main synchronization primitive used in original UNIX
- Definition: a Semaphore has <mark>a non-negative integer value</mark> and supports the following operations:
    - Set value when you initialize
    - `Down()` or `P()`: an atomic operation that waits for semaphore to become positive, then decrements it by `1`
        - Think of this as the `wait()` operation
- `Up()` or `V()`: an atomic operation that increments the semaphore by 1, waking up a waiting P, if any
    - This of this as the `signal()` operation
- Semaphores are like integers, except
    - No negative values
    - Only operations allowed are `P `and `V` – can’t read or write value, except initially
- Operations must be atomic
    - Two P’s together can’t decrement value below zero
    - Thread going to sleep in `P` won’t miss wake up from `V` – even if both happen at same time
- POSIX adds ability to read value, but technically not part of proper interface!


### Two Uses of Semaphores

- Mutual Exclusion (initial value = 1)
    - Also called "Binary Semaphore" or "mutex".
    - Can be used for mutual exclusion, just like a lock:

    ```c
    semaP(&mysem);
    // Critical section goes here
    semaV(&mysem);
    ```
- Scheduling Constraints(initial value = 0)
    - Allow thread `1` to wait for a signal from thread `2`
        - thread 2 waits for thread 1 to finish

        ```c
        // thread1
        ThreadJoin {
            semaP(&mysem); //wait
         }

        // thread2
        ThreadFinish {
            semaV(&mysem); // signal
        }
        ```

### Revisit Bounded Buffer: Correctness constraints for solution

- Correctness Constraints:
    - Consumer must wait for producer to fill buffers, if none full (scheduling constraint)
    - Producer must wait for consumer to empty buffers, if all full (scheduling constraint)
    - Only one thread can manipulate buffer queue at a time (mutual exclusion)
- Remember why we need mutual exclusion
    - Because computers are stupid
    - Imagine if in real life: the delivery person is filling the machine and somebody comes up and tries to stick their money into the machine
- General rule of thumb: <mark>Use a separate semaphore for each constraint</mark>
    - Semaphore fullBuffers; // consumer’s constraint
    - Semaphore emptyBuffers;// producer’s constraint
    - Semaphore mutex; // mutual exclusion

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-07-02.png">

- Why asymmetry?
    - Producer does: `semaP(&emptyBuffer)`, `semaV(&fullBuffer)`
    - Consumer does: `semaP(&fullBuffer)`, `semaV(&emptyBuffer)`


## Resources

- [Berkeley CS162: Operating Systems and System Programming](https://www.youtube.com/watch?v=4FpG1DcvHzc&list=PLF2K2xZjNEf97A_uBCwEl61sdxWVP7VWC)
- [slides-1](https://sharif.edu/~kharrazi/courses/40424-012/)
- [slides-2](https://github.com/Leo-Adventure/Berkeley-CS162-Operating-System/tree/main/Lecture/Slides)
