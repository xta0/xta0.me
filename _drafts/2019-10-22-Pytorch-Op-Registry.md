---
list_title: PyTorch中Operator的注册与派发
title: PyTorch中Operator的注册与派发
layout: post
categories: ["AI", "Machine Learning","Deep Learning"]
---

今天我们接着聊PyTorch的源码，我们将把重点放在Operator的注册与派发上，具体来说，我们要搞清楚两个问题

1. Operator是如何注册到PyTorch里面的，然后再来分析这些Operator是如何被调用的。

### `gen.py`

我们需要先从code-gen开始，CMake在构建PyTorch的时候会多次调用`gen.py`这个脚本

```shell
set(cwrap_files
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/Declarations.cwrap
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/THNN/generic/THNN.h
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/THCUNN/generic/THCUNN.h
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/nn.yaml
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml)

SET(GEN_COMMAND
      "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen.py
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${GEN_ROCM_FLAG}
      ${cwrap_files}
  )

  EXECUTE_PROCESS(
      COMMAND ${GEN_COMMAND}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt
      RESULT_VARIABLE RETURN_VALUE
  )
```
为了研究方便，我们也可以手动调用这个python文件，看看它输出什么

```python
python aten/src/ATen/gen.py \
--source-path aten/src/ATen \
--install_dir ~/Desktop/codegen/gen \
aten/src/ATen/Declarations.cwrap \
aten/src/THNN/generic/THNN.h \
aten/src/THCUNN/generic/THCUNN.h \
aten/src/ATen/nn.yaml \
aten/src/ATen/native/native_functions.yaml \
--output-dependencies aten/src/ATen/generated_cpp.txt
```
上面我们指定了ATen的源码路径，生成代码路径，输入的文件，以及`generated_cpp.txt`的路径。如果我们运行它，会发现它并不会生成具体的代码，而是生成了一份`generated_cpp.txt`里面包含了一些文件的名字，不难猜到，这些文件将是我们最终会生成的。

```python
//gen.py
...
declare_outputs()
if options.output_dependencies is not None:
    file_manager.write_outputs(options.output_dependencies)
    core_file_manager.write_outputs(options.output_dependencies + "-core")
    cuda_file_manager.write_outputs(options.output_dependencies + "-cuda")
else:
    generate_outputs()
```
看源码可知，之所以没有真正生成代码，是由于我们传入了`output_dependencies`，因此我们可以猜到后面cmake还会再次调用这个脚本，并去掉这个参数，来做真正的代码生成。为了研究方便，我们手动将`output-depedencies`去掉之后再运行。可以看到在我们指定的路径下生成了我们想要文件。

```shell
├── CPUType.cpp
├── CPUType.h
├── CUDAType.cpp
├── CUDAType.h
├── Declarations.yaml
├── Functions.h
├── LegacyTHFunctionsCPU.cpp
├── LegacyTHFunctionsCPU.h
├── LegacyTHFunctionsCUDA.cpp
├── LegacyTHFunctionsCUDA.h
├── MkldnnCPUType.cpp
├── MkldnnCPUType.h
├── NativeFunctions.h
├── QuantizedCPUType.cpp
├── QuantizedCPUType.h
├── RegistrationDeclarations.h
├── SparseCPUType.cpp
├── SparseCPUType.h
├── SparseCUDAType.cpp
├── SparseCUDAType.h
├── TypeDefault.cpp
├── TypeDefault.h
└── core_tmp
├── Tensor.h
└── TensorMethods.h
```
这些文件里面比较重要的是`Declarations.yaml`和`Functions.h`。前者将会用于后续代码的生成，后者有近17000多行代码，可以认为是PyTorch所有的Operator的声明。以`conv2d`的实现为例，我们可以在`Function.cpp`中找到其定义

```cpp
static inline Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode _var_guard(true);
    return TypeDefault::conv2d(input, weight, bias, stride, padding, dilation, groups);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton()
        .findSchema({"aten::conv2d", ""}).value();
    return c10::Dispatcher::singleton().callUnboxedOnly<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(
        op, input, weight, bias, stride, padding, dilation, groups);
#endif
}
```
可以看到如果是static dispatch的情况，该函数会直接调用`TypeDefault`中的实现，如果不是static dispatch的情况，则会走到`c10::Dispatcher`，进行二次派发，后面我们还会详细讨论Operator的dispatch，目前我们还是先专注Operater的注册，因此可以先沿着静态派发这条路来分析。

注意到`TypeDefault.cpp`也是我们刚才生成的，我们看看它里面`conv2d`是怎么实现的

```cpp
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::conv2d(input, weight, bias, stride, padding, dilation, groups);
}
```
我们看到`TypeDefault.cpp`并没有真正的实现`conv2d`而是又调了`at::native::conv2d`。继续向下trace可见真正的`conv2d`的实现在`src/ATen/native/Convolution.cpp`中，关于Operator的具体实现，我们后面还会讨论。现在只需知道的是`gen.py`生成了一些Operator的dispatcher文件，其中`Function.h`是源头。

此外，如果查看`TypeDefault.cpp`和`CPUType.cpp`以及`CUDAType.cpp`，我们还会看下面的代码

```cpp
namespace {
auto registerer = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor")
    .catchAllKernel<Tensor (const Tensor &, bool)>(&TypeDefault::_cast_Byte)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
...
```
这是一个典型的C++ Register的Pattern，这说明Operator以Schema的方式被注册进来。


此外如果熟悉Linker，可以知道`registerer`这个全局变量在`main`函数执行之前就会被创建，因此这些ops实际上在binary进行load时就被注册进来了，其中的原因我们后面再详细讨论

小结一下，目前我们知道了`gen.py`干了下面几件事

1. 根据`native_functions.yaml`生成了一些dispatcher文件
2. 向`c10::RegisterOperators`中注册了很多ops
3. 生成了`Function.h`，作为ops的声明
4. 生成了`Declaration.yaml`，作为下一个code-gen脚本的输入

但是我们还不清楚下面几个问题

1. 为什么要用code-gen的方式生成Operator的dispatcher
2. 为什么要设计dispatcher
3. 这些ops的Call Site在哪里
4. 这些ops被注册到了哪里

回答这些问题之前，我们还要继续看完code-gen

### `generate_code.py`

`generate_code.py`是另一个code-gen脚本，会在`gen.py`之后被调用，它依赖上一步生成的`Declaration.yaml`做输入，同样为了研究方便，我们也可以手动的调它

```python
python tools/setup_helpers/generate_code.py \
--declarations-path /Users/taox/Desktop/codegen/gen/Declarations.yaml \
--nn-path aten/src \
--install_dir ~/Desktop/codegen/autograd
```



