---
layout: post
list_title: CS162 | Operating System | Files and I/O
title: Files and I/O
categories: [System Programming, Operating System]
---

## Goals

- UNIX I/O Design Concepts
- High-level File I/O: Streams
- Low-level File I/O: File Descriptors
- How and Why of High-level File I/O
- Process State for File Descriptors
- Common Pitfalls with OS Abstractions

## UNIX I/O Design Concepts

- Uniformity: Everything is a "File"
    - file operations, device I/O, and interprocess communication through open, read/write, close
- Identical interface for:
    - Files on the disk
    - Devices (terminals, printers, etc.)
    - Regular files on the disk
    - Networking (sockets)
    - Local interprocess communication (pipes, sockets)
- Based on the system calls
    - `open()`
    - `read()`
    - `write()`
    - `close()`
- Additional: `ioctl()` for custom configuration that doesn't quite fit
- Open before use
     - Provide opportunity for access control and arbitration
     - Sets up the underlying machinery, i.e., data structures
- Explicitly close
- Byte-oriented
    - Even if blocks are transferred, addressing is in bytes
    - OS responsible for hiding the fact that real devices may not work this way (e.g. hard drive stores data in blocks)
- Kernel buffered reads
    - Part of making everything byte-oriented
    - Process is <mark>blocked</mark> while waiting for device
    - Let other processes run while gathering result
- Kernel buffered writes
    - Complete in background
    - Return to user when data is "handed off" to kernel

### The File System Abstraction

- File
    - Named collection of data in a file system
    - POSIX File data: <mark>sequence of bytes</mark>
        - Could be text, binary, serialized objects, ...
    - File Metadata: information about the file
        - size, modification time, owner, security info, etc.
- Directory
    - "Folder" containing files & directories
    - Hierarchical (graphical) naming
        - Path through the directory graph
        - Uniquely identifies a file or directory
            -  `/home/ff/cs162/public_html/fa18/index.html`
    - Links and Volumes

## I/O and Storage Layers

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-01.png">

## High-Level File API – Streams

Operate on “streams” - sequence of bytes, whether text or data, with a position

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-03.png">

The `fopen` function returns a pointer to a `FILE` data structure. A null pointer will be returned if there is an error.

### Standard Streams and C APIs

Three <mark>predefined streams are opened</mark> implicitly when a program/process is executed 

- `FILE *stdin`: normal source of input, can be redirected
- `FILE *stdout`: normal source of output, can be redirected
- `FILE *stderr`: diagnostics and errors, can be redirected

The `STDIN / STDOUT` enables composition in UNIX. All can be redirected, for instance, using pipe symbol: `|`:

```shell
# `cat`'s `stdout` goes to `grep`'s `stdin`
cat hello.txt | grep "World"
```

- A file copy example:

```c
#include <stdio.h>

#define BUFFER_SIZE = 1024

int main(void) {
    FILE* input = fopen("input.txt", "r");
    FILE* output = fopen("output.txt", "w");
    char buffer[BUFFER_SIZE];
    size_t length;
    // read the whole file and store the length
    length = fread(buffer, BUFFER_SIZE, sizeof(char), input);
    while(length > 0) {
        fwrite(buffer, length, sizeof(char), output);
        // update the length, util reaching the end of the file
        length = fread(buffer, BUFFER_SIZE, sizeof(char), intput);
    }
    fclose(input);
    fclose(output);
    return 0;
}
```

- C API for positioning the file pointer:

```c
int fseek(FILE* stream, long int offset, int whence);
long int ftell(FILE* stream);
void rewind(FILE* stream);
```

For `fseek()` the offset is interpreted based on the `whence` argument:

- `SEEK_SET`: Then `offset` interpreted from beginning (position 0)
- `SEEK_END`: Then `offset` interpreted backwards from end of file 
- `SEEK_CUR`: Then `offset` interpreted from the current position

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-05.png">

### What's in a `FILE`?

- File descriptor (from call to the low-level `open` API)
- An array buffer
- Lock (In case multiple threads use the FILE concurrently)
- some other stuff...

When you call `fwrite`, what happens to the data you provided?

- It gets written to `FILE`'s <mark>buffer</mark>
- If the `FILE`'s buffer is full, then it is flushed, meaning it's written to the underlying file descriptor
- The C standard library may flush the FILE more frequently
    - e.g., if it sees a certain character in the stream
- When you write code, make the weakest possible assumptions about how data is flushed from FILE buffers

```c
char x = 'c';
FILE* f1 = fopen("file.txt", "w");
fwrite("b", sizeof(char), 1, f1);
FILE* f2 = fopen("file.txt", "r");
fread(&x, sizeof(char), 1, f2);
```
The call to `fread` might see the latest write `'b'`, or it might miss it and see end of file, in which case `x` will remain `'c'`.

The first `fwrite` might not have gotten into the kernel depending on whether it got flushed or not. We need to explicitly flush the buffer after `fwrite`

```c
fwrite("b", sizeof(char), 1, f1);
fflush(f1);
```

## Low-Level File API: File Descriptors

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-06.png">

- The integer return from `open()` is a <mark>file descriptor</mark>. This is how OS object
representing the state of a file. User can use that as the "handle" to the file.
- Operations on file descriptors
    - Open system call created an open file description entry in system-wide table of open files
    - Open file description object in the kernel represents an instance of an open file
- System default file descriptors for `stdin`, `stdout`, `stderr`

```c
#include <unistd.h>
STDIN_FILENO ‐ macro has value 0
STDOUT_FILENO ‐ macro has value 1
STDERR_FILENO ‐ macro has value 2
```
- Get file descriptor inside `FILE *`

```c
int fileno(FILE *stream);
```

- Make `FILE*` from descriptor

```c
FILE *fdopen(int filedges, const char* opentype);
```

- Read data from open file using file descriptor:

```c
// Reads up to maxsize bytes
// returns bytes read, 0 => EOF, ‐1 => error
ssize_t read (int filedes, void *buffer, size_t maxsize)
```
- Write data to open file using file descriptor

```c
// returns bytes written 
ssize_t write (int filedes, const void *buffer, size_t size)
```
- Reposition file offset with kernel(this is independent of any position held by high-level FILE descriptor for this file!)

```c
off_t lseek (int filedes, off_t offset, int whence)
```

-  Wait for i/o to finish

```c
int fsync (int fildes)

// wait for ALL to finish
void sync (void)
```

- A little example

```c
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
int main() {
    char buf[1000];
    int      fd = open("lowio.c", O_RDONLY, S_IRUSR | S_IWUSR);
    ssize_t  rd = read(fd, buf, sizeof(buf));
    int     err = close(fd);
    ssize_t  wr = write(STDOUT_FILENO, buf, rd);
}
```

### Other Low-Level APIs

- Operations specific to terminals, device, networking
    - e.g., `ioctl`
- Duplicating descriptors
    - `int dup2 (int old, int new)`
    - `int dup (int old)`
- Pipes - channel
    - `int pipe(int pipefd[2])`
    - `writes to pipefd[1] can be read from pipefd[0]`
- Memory mapped files
- File Locking
- Asynchronous I/O
- Generic I/O Control Operations

## High-Level vs. Low-Level File APIs

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-07.png">

As you can see, `fread()` does more work before going into the kernel. Internally, the `fread()` maintains a buffer. When reading from the kernel, `fread()` put it into a local memory buffer, and all the subsequent `freads()` you do for a while just look in that buffer and grab the next `BUFFER_SIZE` without having to go into the kernel, as kernel processing is expensive and slower.

## Streams vs. File Descriptors

Streams are buffered in user memory

```c
printf("Beginning of line ");
sleep(10); // sleep for 10 seconds
printf("and end of line\n");
```
The `printf` function goes to the buffered version of `stdout`. When hitting the new line character, the buffer will be flushed out, so we print everything all at once.

However, if we use low-level C API calls, operations on file descriptors are visible immediately

```c
write(STDOUT_FILENO, "Beginning of line ", 18);
sleep(10);
write("and end of line \n", 16);
``` 
This outputs "Beginning of line" 10 seconds earlier than `and end of line`. There is no buffering in this path at the bottom, but there is buffering in the path at top. However, the system/kernel level buffering is completely transparent to users. You won't feel it.

###  Why Buffering in Userspace?

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-07.png">

- Avoid <mark>system call overhead</mark>
    - Time to copy registers, transition to kernel mode, jump to system call handler, etc.
- Minimum syscall time: 
    - syscalls are <mark>25x more expensive than function calls(~100 ns)</mark>
    - The blue bars are user level function calls
    - The green bars are all system calls for getting `getpid()`!
    - Not to make syscall if we can avoid them
- Read/write a file byte by byte?
    - With the syscall APIs, the max throughput is ~10MB/second
    - With `fgetc`, the speed can keep up with your SSD
-  System call operations less capable
    - Simplifies operating system
    - Example: No "read until new line" operation
        - Solution: Make a big read syscall, find first new line in userspace

