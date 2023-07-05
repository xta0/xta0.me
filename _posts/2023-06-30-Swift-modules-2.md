---
list_title: Understand Swift module | Part 2 | Static libraries
title: Use Swift module as static libraries
layout: post
categories: ["Swift", "Compiler", "iOS", "Apple"]
---

### Static Libraries

Let's start off by creating a Swift static library without using Xcode:

```shell
xcrun swiftc
-emit-library
-emit-object MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk
```
This produces an object file `MyLogger.o`. To incorporate it into our applications, it must be archived to create a static library:

```shell
ar rcs libMyLogger.a MyLogger.o
```

Typically, with static libraries, a header file is necessary. However, in the case of Swift, header files do not exist. Instead, we will need a Swift module file as mentioned in the previous article. Let's use the following command to create `MyLogger.swiftmodule`:

```shell
xcrun swiftc
-emit-module MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk
-module-name MyLogger
```

In addition to produce the Swift module file, the `-emit-module` option also generates a bunch of other files:

```shell
├── MyLogger.abi.json
├── MyLogger.swiftdoc
├── MyLogger.swiftmodule
├── MyLogger.swiftsourceinfo
```

With the static library file and the swiftmodule file now at our disposal, it's time to add those files into a Xcode project and test our library. However, before running the test, there's an additional step we need to do to make Xcode locate the library and our Swift module file:

1. Making sure the **Library Search Paths** contain the path to `MyLogger.a`
2. Adding the `MyLogger.swiftmodule` path to the **Import Paths** under **Swift Compiler - Search Paths** settings

With these steps completed, we are now ready to test our library. Within any Swift file, type the following code:

```swift
import MyLogger

let logger = MyLogger(prefix: "> ");
logger.log(object: "Hello!");
```

The Xcode builds and runs perfectly!

> You might observe that the syntax color of `MyLogger` is off, which means there is something wrong with the Xcode indexing. While the autocomplete feature still functions, we'll overlook this indexing issue for now since addressing it is beyond the scope of this post.

### The compiler version issue

As previously discussed, a Swift module file is tied to a specific version of the Swift compiler. In our case, we are using Xcode 14.3.1, the Swift compiler version is

```shell
xcrun swiftc --version                                                    ─╯
swift-driver version: 1.75.2 Apple Swift version 5.8.1 (swiftlang-5.8.0.124.5 clang-1403.0.22.11.100)
Target: arm64-apple-macosx13.0
```

Why does this matter? Suppose we distribute this precompiled Swift module to other developers, their Swift compiler version may differ. To illustrate this, let's install Xcode 15 and execute our code. We should see the following error:

```shell
Compiled module was created by a different version of the compiler '5.8.0.124.5'; rebuild 'MyLogger' and try again
```
This is because Xcode 15's toolchain uses a newer Swift compiler:

```shell
cd /Applications/${path_to_xcode15}/Contents/Developer/Toolchains/XcodeDefault.xctoolchain

./usr/bin/swiftc --version
swift-driver version: 1.82.2 Apple Swift version 5.9 (swiftlang-5.9.0.114.10 clang-1500.0.29.1)
Target: arm64-apple-macosx13.0
```

The solution to this problem is to use XCFramework which contains Swift interface files as discussed in the previous artical. We will talk more about how to build a XCFramework in the next article. In the meantime, let's continue to explore some other scenarios where a Swift module imports an Objective-C module and vice versa.

### Import Objective-C modules into Swift

Let's first modify our `MyLogger.swift` to import an Objective-C module

``` swift
import MyLoggerInternal

private var objc_logger = MyLoggerInternal()

@objc public func objc_log(object: String) {
    let messasge = prefix + "[OBJC]" + " \(object as NSString)";
    self.objc_logger.log(messasge);
}
```
Given that we're `import`ing a Clang module, the Swift compiler will seek the module definition of MyLoggerInternal. Let's go ahead and create a module map for this Objective-C module:

```shell
// module.modulemap

module MyLoggerInternal {
    header "MyLoggerInternal.h"
    export *
}
```

Next, we'll regenerate the static library and the Swift module. This time, we'll use the `-static` option to produce these two files using a single command:

```shell
swiftc
-emit-library
-emit-module
-static
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk
-Xcc -fmodule-map-file=../MyLoggerInternal/module.modulemap
../MyLogger.swift
```

In the above command, we use `-Xcc -fmodule-map-file` explicitly to direct the Swift-embedded Clang compiler to the module map file location. Alternatively, we can just use `-I` to point to the directory containing `MyLoggerInternal.h`. This is because, by default, `MyLoggerInternal` is an implicit import, and the Swift compiler will look for `module.modulemap` in the same directory as `MyLoggerInternal.h`.


```shell
-I ../MyLoggerInternal/
```

Lastly, let's swap out the old files and incorporate the new ones in Xcode. We should now encounter this error from Xcode:

```shell
Missing required module 'MyLoggerInternal'
```

This happens when Xcode `import`ing the `MyLogger` module, as `MyLogger` imports another moduel as dependencies. A naive solution would be to define a module map for `MyLogger` that contains the `MyLoggerInternal`.

```shell
module MyLogger{
    export *
}

module MyLoggerInternal {
    export *
}
```

This makes the error go away. However, when we compile the app, we will hit a linker error

```shell
Undefined symbol: _OBJC_CLASS_$_MyLoggerInternal
```

The issue here is that the static library doesn't contain any symbols from the `MyLoggerInternal` module. We will resolve this error shortly. Now let's revisit our module map definition. As the name suggests, `MyLoggerInternal` is a module private to `MyLogger`. It is implementation details that shouldn't be exposed externally. Therefore, this module shouldn't be revealed in the module map.

To workaround this, we can use [a undocumented feature](https://forums.swift.org/t/update-on-implementation-only-imports/26996) called `@implementation_detail`. Essentially, we just need to add this keyword before the import directive:

```swift
@_implementationOnly import MyLoggerInternal
```

Now we can safely delete the `module MyLoggerInternal` from the module map and regenerate the module file. Xcode will no longer complain the `Missing required module` error.

### Resolve the linking error

Finally, let resolve the linking error. When we created the static library, it only contained the symbols from the `MyLogger` module. What we need to do is to generate the object file for `MyLoggerInternal` and combine it with the `MyLogger.o`

```shell
// emit MyLoggerInternal.o
 clang -c  ../MyLoggerInternal/MyLoggerInternal.mm -I ../MyLoggerInternal -target arm64-apple-ios16.4-simulator

 // emit MyLogger.o
 xcrun swiftc
-emit-library
-emit-object MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk

// create libMyLogger.a
libtool -static -arch_only arm64 ./MyLogger.o ./MyLoggerInternal.o -o libMyLogger.a
```

Now replacing the old `libMyLogger.a` with the new one. Everything should now work seamlessly! To recap, when distributing a static library without using a framework format, we need

1. A swift module file
2. A static library file
3. A moduel.modulemap file (this may for may not be needed, depending on what we intend to expose to the outside)

### XCFramework