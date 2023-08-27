---
list_title: Swift | Swift binaries
title: Swift binaries
layout: post
categories: ["Swift", "Compiler", "iOS", "Apple"]
---

## Swift only static Libraries

Let's start off by creating a Swift static library without using Xcode:

```shell
xcrun swiftc
-emit-library
-emit-object MyLogger.swift
-target arm64-apple-ios16.4-simulator
-sdk /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk
```
This produces an object file named `MyLogger.o`. To incorporate it into our applications, it must be archived into a static library:

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

In addition to producing the Swift module file, the `-emit-module` flag also generates a bunch of other files:

```shell
├── MyLogger.abi.json
├── MyLogger.swiftdoc
├── MyLogger.swiftmodule
├── MyLogger.swiftsourceinfo
```

Now we have the static library file and the swiftmodule file at our disposal, it's time to add those files to our Xcode project and test our library. However, before running the test, there's an additional step we need to do to make Xcode locate the library and our Swift module file:

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
xcrun swiftc --version
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

The solution to this problem is to use XCFramework which contains Swift module interface files as discussed in the previous article. We will talk more about how to build a XCFramework in the last section. For now, let's continue to explore some other scenarios where a Swift module imports an Objective-C module and vice versa.

## Import Objective-C modules into Swift

Let's first modify our `MyLogger.swift` to import an Objective-C module

``` swift
import MyLoggerInternal

private var objc_logger = MyLoggerInternal()

@objc public func objc_log(object: String) {
    let messasge = prefix + "[OBJC]" + " \(object as NSString)";
    self.objc_logger.log(messasge);
}
```

Given that we're `import`ing a Clang module, the Swift compiler will seek the module definition of `MyLoggerInternal`. Let's go ahead and create a module map for this Objective-C module:

```shell
// module.modulemap

module MyLoggerInternal {
    header "MyLoggerInternal.h"
    export *
}
```

Next, we'll regenerate the static library and the Swift module. This time, we'll use the `-static` option to produce these two files together using a single command:

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

In the above command, we use `-Xcc -fmodule-map-file` explicitly to direct the Swift-embedded Clang compiler to the module map file location. Alternatively, we can just use `-I` to point to the directory containing `MyLoggerInternal.h`. This is because, by default, `MyLoggerInternal` is an implicit import, and the Swift compiler will look for `module.modulemap` in the same directory as `MyLoggerInternal.h`, as mentioned in the previous article.


```shell
-I ../MyLoggerInternal/
```

Lastly, let's swap out the old libraries with the new ones in Xcode. We should now encounter this error from Xcode:

```shell
Missing required module 'MyLoggerInternal'
```

This happens because `MyLogger` depends on the `MyLoggerInternal` module. It turns out any client code that imports the module must also be able to import all the modules it imported. A naive solution would be to define a module map for `MyLoggerInternal`.

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

The issue here is that the static library doesn't contain any symbols from the `MyLoggerInternal` module. We will resolve this error later. Now let's revisit our module map definition. As the name suggests, `MyLoggerInternal` is a module private to `MyLogger`. It is implementation details that shouldn't be exposed externally. Therefore, this module shouldn't show up in the module map.


To workaround this, we can use [an undocumented feature](https://forums.swift.org/t/update-on-implementation-only-imports/26996) called `@_implementationOnly`. This means we are ensuring that the imported module can only be used for the implementation of our module, not as part of the module's API.


```swift
@_implementationOnly import MyLoggerInternal
```

Now we can safely delete the `module MyLoggerInternal` from the module map and regenerate the module file. Xcode will no longer complain about the `Missing required module` error.

### Resolve the linking error

Finally, let's resolve the linking error. When we created the static library, it only contained the symbols from the `MyLogger` module. What we need to do is to generate the object file for `MyLoggerInternal` and combine it with `MyLogger.o`

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

Now replacing the old `libMyLogger.a` with the new one. Everything should now work seamlessly! To recap, when distributing a static library without using a `.xcframework`` format, we need

1. A swift module file
2. A static library file
3. A `moduel.modulemap` file (this may or may not be needed, depending on what we intend to expose to the outside)

As you can see, this process is not quite elegant and error-prone, as users have to manually create those files and put them in the right location. The additional steps needed to configure the Xcode project are also quite complicated. In the next section, we will explore how to use XCFramework to bundle the static libraries, which is the Apple recommended way of distributing prebuilt binaries.

## Binary frameworks (XCFramework)

Before we start building `MyLogger.framework`, it is useful to go over Apple's guidelines first

- [WWDC 2019 Session 417: Binary Frameworks in Swift](https://developer.apple.com/videos/play/wwdc2019/416/)
- [WWDC 2020 Session 10147: Distribute binary frameworks as Swift packages](https://developer.apple.com/videos/play/wwdc2020/10147/)

To get started, we need to create an Xcode project and choose the framework template. Then follow the [document](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) here. Basically, we need to create static libraries for different platforms that we want to support. And then package them together into a `.xcframework`.

> Once we have figured out the framework structure, we can replicate the process without using Xcode. For now, let's just follow Apple's guidelines

```shell
#!/usr/bin/env bash

xcodebuild archive \
-project MyDummyLogger.xcodeproj \
-scheme MyDummyLogger \
-destination "generic/platform=iOS Simulator" \
-archivePath "archives-sim/MyDummyLogger-sim"

sleep 1

xcodebuild archive \
-project MyDummyLogger.xcodeproj \
-scheme MyDummyLogger \
-destination "generic/platform=iOS" \
-archivePath "archives-arm64/MyDummyLogger-arm64"

sleep 1

xcodebuild \
-create-xcframework \
-archive archives-sim/MyDummyLogger-sim.xcarchive \
-framework MyDummyLogger.framework \
-archive archives-arm64/MyDummyLogger-arm64.xcarchive \
-framework MyDummyLogger.framework \
-output xcframework-static/MyDummyLogger.xcframework
```

The script creates two static libraries for different architectures, then zip them into a `.xcframeworkk` file. Now let's take a look at the structure of generated `.xcframework`

```shell
└── MyDummyLogger.xcframework
    ├── Info.plist
    ├── ios-arm64
    │   └── MyDummyLogger.framework
    │       ├── Headers
    │       ├── Info.plist
    │       ├── Modules
    │       ├── MyDummyLogger
    │       └── _CodeSignature
    └── ios-arm64_x86_64-simulator
        └── MyDummyLogger.framework
            ├── Headers
            ├── Info.plist
            ├── Modules
            ├── MyDummyLogger
            └── _CodeSignature
```

By default, Xcode produces static libraries. The `MyDummyLogger` under the simulator directory is a FAT binary

```shell
Architectures in the fat file: MyDummyLogger are: arm64 x86_64
```

As mentioned in the previous section, XCFrameworks solves the problem of shipping precompiled Swift modules. If we take a look at the `Modules` folder, we will see a bunch of `.swiftinterface` files

```shell
// MyDummyLogger.swiftmodule

arm64-apple-ios.abi.json                arm64-apple-ios.swiftdoc
arm64-apple-ios.private.swiftinterface  arm64-apple-ios.swiftinterface

// arm64-apple-ios.swiftinterface

// swift-interface-format-version: 1.0
// swift-compiler-version: Apple Swift version 5.8.1 (swiftlang-5.8.0.124.5 clang-1403.0.22.11.100)
// swift-module-flags: -target arm64-apple-ios16.4-simulator -enable-objc-interop -enable-library-evolution -swift-version 5 -enforce-exclusivity=checked -O -module-name MyDummyLogger
// swift-module-flags-ignorable: -enable-bare-slash-regex
import Foundation
@_exported import MyDummyLogger
import Swift
import _Concurrency
import _StringProcessing
public class MyDummyLogger {
  public init(prefix: Swift.String)
  public func log<T>(object: T)
  @objc public func objc_log(object: Swift.String)
  @objc deinit
}
```

It is also worth noting that `.xcframework` seems to be the only supported format when it comes to releasing precompiled Swift code.
