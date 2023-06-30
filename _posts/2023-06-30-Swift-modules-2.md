---
list_title: Understand Swift module | Part 2
title: Build Swift module from scratch
layout: post
categories: ["Swift", "Compiler", "iOS", "Apple"]
---

### Static Libraries

Let's start off by creating a Swift static library without using Xcode:

```
xcrun swiftc
-emit-library
-emit-object MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator16.4.sdk
```
This produces an object file `MyLogger.o`. To use it in our applications, we need to archive it to make it a static library:

```
ar rcs libMyLogger.a MyLogger.o
```
Typically, with static libraries, a header file is necessary. However, in the case of Swift, header files do not exist. Instead, we will need a `.swiftmodule` as mentioned in the previous article. Let's use the following command to create a swiftmodule object

```
xcrun swiftc
-emit-module MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator16.4.sdk
-module-name MyLogger
```

In addition to the swiftmodule file, this command also generates a bunch of other files:

```
├── MyLogger.abi.json
├── MyLogger.swiftdoc
├── MyLogger.swiftmodule
├── MyLogger.swiftsourceinfo
```

With the static library file and the swiftmodule file now at our disposal, it's time to drag those files into a Xcode project and test our library. However, before running the test, there's an additional step we need to do to make Xcode locate the library and our Swift module file:

1. Making sure the **Library Search Paths** contain the path to `MyLogger.a`
2. Adding the `MyLogger.swiftmodule` path to the **Import Paths** under **Swift Compiler - Search Paths** settings

With these steps completed, we are now prepared to test our library. Within any Swift file, type the following code:

```swift
import MyLogger

let logger = MyLogger(prefix: "> ");
logger.log(object: "Hello!");
```

The Xcode builds and runs perfectly!

> You might observe that the syntax color of `MyLogger` is off, which means there is something wrong with the Xcode indexing. While the autocomplete feature still functions, we'll overlook this indexing issue for now since addressing it is beyond the scope of this post.
