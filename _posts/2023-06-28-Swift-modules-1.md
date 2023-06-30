---
list_title: Understand Swift module | Part 1
title: Swift module 101
layout: post
categories: ["Swift", "Compiler", "iOS", "Apple"]
---

## Modules in Objective-C

If you’ve been doing iOS development long enough, the term module probably sounds familiar to you even before Swift came out. The idea of Objective-C modules(Clang modules) was introduced to LLVM in [WWDC 2013 (session 404)](https://devstreaming-cdn.apple.com/videos/wwdc/2013/404xbx2xvp1eaaqonr8zokm/404/404.pdf). Basically, it allows developers to use a new syntax to import frameworks, for instance,  `@import UIKit`, which is equivalent to `#import <UIKit/UIKit.h>`. The main selling point of using modules is that it can significantly reduce the compilation time, especially for large applications. This is because it saves the time for compiler to parse and preprocess headers (textural inclusion).

Since then, modules have crept their way into the Xcode developer ecosystem, becoming even more prevalent with the introduction of Swift. There is a fantastic document on the [rationale behind modules](https://clang.llvm.org/docs/Modules.html#introduction) in the Clang documentation — give that a read if you are interested in learning more.

## Modules in Swift

In Swift, you can only import  modules. A Swift program is composed of a number of modules. Each module is a binary representation that is imported by the source files. There are four types of modules that can be imported:

- **A binary `.swiftmodule` file**: binary `.swiftmodule` files are created by the Swift compiler. It provides the interface by which other Swift modules can access. Binary `.swiftmodule` files are <mark>tied to a specific compiler version</mark>.
- **A binary `.pcm` file**: this is created by the Swift compiler's embedded Clang compiler when it builds an Objective-C/C module. It provides the interface by which Swift modules can access to the binary. `.pcm` files are <mark>tied to a specific compiler version</mark>.
- **A textual `.swiftinterface` file**: textual `.swiftinterface` files are a superset of Swift source code that can be distributed along with binary libraries. They are compatible with multiple versions of the Swift compiler.
- **Objective-C modules (Clang module)**: a set of Objective-C headers, described by a module map that can be imported into Swift.

There are two different ways to import these modules: implicitly and explicitly. Implicit importing is the default, and requires passing `-I/some/path` for every folder that contains a module. Each search path is walked to find the available modules, which are then loaded during the import stage.

Explicit module importing passes no search paths, instead, it passes the paths of the imported modules directly using `-explicit-swift-module-map-file (JSON)` and `-fmodule-file=/some/pcm` for every clang modules.

## Implicit Module

- When the Swift compiler sees a module import such as import `MyModule`, it looks through the module search path to find a module with the corresponding name.
- When the compiler finds a binary `.swiftmodule` or `.pcm` file, it loads it directly. However, when it finds a `.swiftinterface` file or an Objective-C module (Clang module), the Swift compiler will implicitly spawn a thread with another compiler instance to compile each textual module into binary. Once complete, the Swift compiler thread will load it into its thread.
- The compiled module binaries can be reused during the compilation. They are cached in the module cache.  The module cache is a shared directory on the system (within DerivedData in Xcode), which can be overridden via the `-module-cache-path` command line parameter.
- The Swift compiler will look for an up-to-date binary module in the module cache before initiating the compilation of a textual module; when it does compile the textual module into a binary module, it will be recorded in the cache for other Swift compiler instances to bind. Multiple Swift compiler instances will access the module cache at the same time, so the compiler put locks on reading/writing to the cache.
- Using shared module cache can lead to performance issues:
    - During compilation, there are likely to be many Swift compiler instances sharing the same module cache, and those instances could compete for accessing the same binary modules. The concurrent situation could cause threads to be suspended and waiting for locks to be released.
    - Compiler instances/threads will only realize which binary modules are necessary during their compilation, which means they need to duplicate work (each compiler instance compiles a copy of the binary module)
    - Additionally, every compiler instance is doing redundant work to validate each binary module in the module cache, e.g., running stat for every header file in an Objective-C module.

## Problems with Implicit modules

### Build Speed

- While `.swiftmodule` files are cacheable by build system, Clang module compilation works differently. When using implicit modules, you do not directly import a `pcm` file. Instead, you import a <mark>modulemap</mark> file, which describes the headers that form the module. Clang will then search a global module cache folder to see if there exists a precompiled module(pcm) for this modulemap and the set of flags being used, if not, it compiles one and writes to the module cache.
- It is impossible to cache the implicit module output as it is built as a side effect and untracked by build system. The produced modules are also not relocatable or deterministic
- This means for every build some amount of Clang modules needs to be recompiled, depending on the local state of the module cache on the users’s machine.

### Debugging

- This is the status quo for building, which has a few implications. One of the victims of implicit module compilation is debugging.
- When debugging Swift, LLDB needs the PCM files used during compilation to be able to create an expression context for Swift
- When building in Xcode, this works by embedding PCM paths in the object files DWARF, which allows the debugger to fairly quickly find and load them at attach time.
- When building remotely, the built output, however, the referenced PCM paths do not exist as they are in some other machines’ module cache. This means the path embedding has to be disabled and LLDB has to recompile the modules itself.
- This can take a substantial amount of time, on the order of 2 min for Stella.

### Remote Execution

- Another significant issue is remote execution, as all actions are remotely executed by default.
- As each action is isolated and transient, they cannot rely on a shared module cahce. This would cause build errors as the cached modules would refer to inputs no longer present
- This means every remotely executed Swift compilation has to compile its entire set of dependent Clang modules every time, including the SDK modules.
- Google saw a 60 - 80% reduction in build time when migrating their remote builds to use explicit modules by avoiding this step.
- So until we have this in place, Swift can only be efficiently built locally.

## Explicit Module

- The idea of Explicit module is to move the compilation of textual modules out of the Swift compiler instance that imports the module, and prebuild them in build systems.
- The build system needs to ensure that all the binary modules needed by a Swift compilation job have already been built before that compilation job executes.
- Explicit module builds are meant to eliminate the problems with implicit module builds, improving parallelism,  reducing redundant work among Swift compiler instances.

### How to enable Explicit modules

- The complexity arises from a couple of issues, but primarily the hard part is working out which modules you need to provide for each compilation
- There a few categories of modules we have to worry about
    - First party Swift modules (ie from targets containing Swift)
    - First party Clang modules (ie from modular library dependencies in fbobjc)
    - SDK Swift modules, eg from SDK frameworks or Swift overlays in `usr/lib.swift`
    - SDK Clang modules, either from frameworks or usr/include

### First Party Swift Modules

- First party Swift code is the simplest case. Every target already specifies its deps in its BUCK/Bazel file.
- For each compile action then we traverse the deps collecting all the SwiftCompile rules and collecting the paths to their output.
- These paths are collected in JSON file which is passed to the compiler using `-explicit-swift-module-map-file`
- We need the transitive dependencies at each step as each Swift module embeds references to its own dependencies.
- The size of this set can be reduced by adopting `@_implementationOnly` imports, which allow for private deps in Swift code. Only the exported_deps need to be traversed.

### First Party Clang Modules

- First party C code is a bit trickier. Finding the dependencies is similar to the Swift case: we can traverse the exported_deps of all direct deps to find all the modular libraries with exported headers.
- Where this varies from Swift is that there is no pre-existing module output from these headers. Not only that, the pcm files need to be compiled with the flags of the importing library rather than the library that contains the headers.
- This can result in poor sharing and a lot of wasted work if each target has varying compiler flags, as all the pcm files have to be compiled for each set of flags.
- The algorithm to collect the pcm files becomes:
    - Collect Swift compiler flags for this target
    - Traverse deps to get modulemap output
    - From the bottom up, compile a modulemap into a pcm using the Swift compiler’s `-emit-pcm` flag, passing the modulemap file as the source
    - At each stage we pass in the dependent pcm files that were just compiled until the entire set of pcm files is now built
    - Finally pass each of these pcm files to the Swift compilation using `-fmodule-file=path/to/each/pcm`
- Build graph implications
    - This has some implications for the build graph. We have gone from a seemingly parallel build graph to a serial one: each module needs to be compiled in sequence with its dependencies already compiled.
    - This differs from traditional C compilation where all headers are available to all compile actions and everything can be done in parallel.
    - In reality: the situation is not that much changed. While it seemed that the Swift compile actions could run in parallel, internally they would be serialized when compiling the implicit modules. Worse, there would be serialization between actions as the shared module cache used file locking to ensure each module was written correctly.
    - Effectively the hidden internal actions have now been changed to explicit cacheable actions in the build system instead.

### SDK Swift Modules

- So now we have our first party modules and we’re ready to go. All except the SDK modules.
* This is where we hit a bit of a mismatch between the build rule API and the requirements for module compilation. The only way targets have of expressing their SDK dependencies is via the frameworks attribute. This has a couple of issues:
    - It can only be used to express a framework dependency, and Swift overlays and other Clang modules exist outside the frameworks
    - As this attribute is only used at link time , it is very poorly specified. Almost no targets correctly specify their frameworks
- To solve those two issues, we introduce a sdk_module attribute
    - The first problem required the addition of the sdk_modules attribute. This new attribute is used to specify a list of the SDK modules that a target depends on, regardless of their location in the SDK.
    - The build macros were modified to populate this field from the frameworks attribute, and in cases where there were deps outside of frameworks they were added explicitly.
    - The next step was to fix the frameworks specified in each target. A series of codemods were developed to fetch the correct list from each targets sources and to modify the frameworks field. There are many pitfalls here with eg cross platform libraries, ifdefs, unimported headers, platform specific frameworks, frameworks introduced later than the deployment target, etc

### SDK Dependencies

- So now we have our list of SDK frameworks we depend on for each target, are we ready yet? Of course not, there’s at least 10 more slides
- What we are missing are the exported dependencies of each module. Both Swift and Clang modules can export dependent modules, either explicitly or implicitly.
- This can happen through the swiftinterface files, the modulemap or simply via transitively importing a header that belongs to another modulemap.
- We need a way to scan the SDK modules to determine what their dependencies are
- `-scan-dependencies`
    - Fortunately the compiler has a mode designed for exactly this: `-scan-dependencies`
    - When run against a Swift file it will output a JSON file that lists all the imported modules and their dependencies.
    - A script was developed to scan the SDK folder, create an empty Swift file that imports all the available modules for each platform and to run it through the compiler using the `-scan-dependencies` flag.
    - After some massaging to relativize paths we output this JSON into fbsource. Its connected to a new build attribute on swift_toolchain called sdk_dependencies_path.
- Build SDK dependencies
    - Given a JSON file describing the dependencies of the SDK modules, we added a slightly unconventional class to parse this, create a graph and to build SDK modules on demand.
    - By creating flavours for each module, with a hash of the flags used for Clang modules, it is possible to leverage build system to handle the caching and recursion for us.
    - So finally we have everything in place to build all the dependent modules required for each Swift compilation action
