---
title: Concurrency with asyncio
list_title: Python Deep Dive | Concurrency with asyncio
layout: post
categories: [Python]
---

## `asyncio`

There are several ways to achieve this concurrency in Python. One of the most recent additions to the Python ecosystem is the `asyncio` library. `asyncio` is a concurrency framework built around three important concepts:

- **Coroutines**: Functions that can suspend execution using await.
- **Event loop**: A scheduler that manages coroutine execution.
- **Non-blocking I/O**: Allows a thread to keep processing other work while waiting for I/O readiness.

While the name of `asyncio` may make us think that this library is only good for I/O operations, it has functionality to handle other types of operations as well by interoperating with multithreading and multiprocessing. With this interoperability, we can use `async` and `await` syntax with threads and processes making these workflows easier to understand.

### Non-blocking I/O vs Blocking I/O

Let's take a look at the following two code examples:

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
async def fetch(name):
    print(f"{name}: start")

    await asyncio.sleep(2)

    print(f"{name}: done")

async def main():
    await asyncio.gather(
        fetch("A"),
        fetch("B"),
        fetch("C"),
    )

asyncio.run(main())


A: start
B: start
C: start
A: done
B: done
C: done
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
import asyncio
import time

async def task(name):
    print(f"{name}: start")

    time.sleep(2)  # Blocking!

    print(f"{name}: done")

async def main():
    await asyncio.gather(
        task("A"),
        task("B"),
        task("C"),
    )

asyncio.run(main())

A: start
A: done
B: start
B: done
C: start
C: done
</code>
</pre>
</div>
</div>

In the first example, all three tasks finish in approximately 2 seconds, rather than 6 seconds. `asyncio.sleep(2)` simulates waiting without blocking the event-loop thread. When Task A encounters await asyncio.sleep(2), the event loop can execute Task B and then Task C. The event loop doesn't have to wait for Task A to finish. 

Conceptually, when dealing with non-blocking I/O:

- The coroutine attempts to read from a non-blocking object.
- If data isn't available, the coroutine suspends.
- The event loop monitors the object through an OS readiness mechanism.
- Meanwhile, the event loop runs other ready tasks.

Once the object becomes readable, the event loop schedules the suspended coroutine to resume.

In the second example, all three tasks finish in approximately 6 seconds. This is because `time.sleep()` blocks the event-loop thread. The coroutine cannot suspend cooperatively while that call executes.

`async` does not automatically make I/O non-blocking. The distinction becomes especially important with CPU-intensive code:

```
async def calculate():
    # Long CPU-bound loop with no await
    for i in range(100_000_000):
        do_some_math(i)
```
This coroutine can block the event loop for its entire computation. Adding async does not make it parallel or preemptible.

| Operation | Blocks event-loop thread? |
|---|---|
| `time.sleep(2)` | Yes |
| `await asyncio.sleep(2)` | No |
| `requests.get(url)` | Yes |
| `await aiohttp_session.get(url)` | No, for normal async network waits |
| `open(path).read()` | Typically yes |
| `await asyncio.to_thread(blocking_func)` | No, moves blocking work to a worker thread |

`await` means the coroutine may suspend and allow other tasks to run. It does not automatically mean the function is non-blocking, nor does it guarantee that the underlying operation is performed asynchronously.

### Preemptive multitasking vs Cooperative multitasking

- **Preempting**: We let the operating system decide how to switch between which work. How this mechanism works under the hood is up to the operating system itself. It is primarily achieved through using either multiple threads or multiple processes.

- **Cooperative**: Instead of relying on the operating system to decide when to switch between which work is currently being executed, we explicitly code points in our application where we can let other tasks run. The tasks in our application operate in a model where they cooperate, explicitly saying, “I’m pausing my task for a while; go ahead and run other tasks.”



