---
layout: post
list_title: CS162 | Operating System | Files and I/O
title: Files and I/O
categories: [System Programming, Operating System]
---

## Goals

- High-level File I/O: Streams
- Low-level File I/O: File Descriptors
- How and Why of High-level File I/O
- Process State for File Descriptors
- Common Pitfalls with OS Abstractions

### POSIX I/O: Everything is a "File"

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
- Note that the "Everything is a File" idea was a radical idea when proposed
    - Dennis Ritchie and Ken Thompson described this idea in their seminal paper on UNIX called "The UNIX Time-Sharing System" from 1974

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

### High-Level File API – Streams

Operate on “streams” - sequence of bytes, whether text or data, with a position

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-03.png">

The `fopen` function returns a pointer to a `FILE` data structure. A null pointer will be returned if there is an error.

### Standard Streams 

<makr>Three predefined streams are opened implicitly when a program/process is executed </mark>

- `FILE *stdin`: normal source of input, can be redirected
- `FILE *stdout`: normal source of output, can be redirected
- `FILE *stderr`: diagnostics and errors, can be redirected

`STDIN / STDOUT` enables composition in UNIX

- All can be redirected (for instance, using pipe symbol: `|`):
    – `cat hello.txt | grep "World"`
        - `cat`'s `stdout` goes to `grep`'s `stdin`

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2020/01/os-04-04.png">