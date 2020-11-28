---
list_title: Notes | CppCon 2020 | Part 1
title: CppCon 2020 Notes
layout: post
categories: ["C++"]
---

## Back to Basics

- Move Semantics
- The Hidden Secret of Move Semantics

### Move Semantics

```cpp
template<class T>
constexpr remove_reference_t<T>&& move(T&& t) noexcept {
    return static_cast<remove_reference_t<T>&&>(t)
}
```

`std::move` doesn't move anything. It converts any expression into an rvalue so it can be bound to an rvalue reference.

- RValue References Guideline: 
    - **No rvalue reference to const type**
        - Use a non-const rvalue reference instead.
        - Most uses of rvalue references modify the object being referenced
        - Most rvalues are not const
    - **No rvalue reference as function return type**
        - Core Guideline F.45 : Don't return a `T&&`
        - Return by value instead.
            - see [C++中的右值引用与std::move]()
        - Rvalue references often bind to temporaries, which don't outlive the function

- `std::move` Guidelines
    - **Next operation after std::move is destruction or assignment**
        - The object that was moved from shouldn't be used again
    - **Don't `std::move` the return of a local variable**
        -  Core Guideline F.48: Don't `return std::move(local)`
        - C++ Standard has a special rule for this
            - The return expression is an rvalue if it is a local variable of parameter

        ```cpp
        std::string func(std::string param, std::string* ptr){
            std::string local = "Hello";
            *ptr = local;
            if(some_condition()){
                return local; //rvalue, calls the move constructor
            }
            return *ptr; //lvalue, calls the copy constructor
        }
        ```
- Move Constructor
    - Implicitly declared move constructor if there no user-defined 
        - destructor
        - copy constructor
        - copy assignment operator
        - move assignment operator
    - Implicitly declared or explicitly use `=default` to let compiler generate move constructor for you
        - Move constructs each base and non-static data member
        - Deleted if any base or non-static data member cannot be move constructed

    ```cpp
    struct s {
        double* data;
        s( s&& other) noexcept 
            : data(std::move(other.data)){
            other.data = nullptr;
        }
        //a more elegant way is using std::exchange
        s (s && other) noexcept
            : data(std::exchange(other.data, nullptr))
        {}
    };
    ```

- Move Assignment Operator
    - Implicitly declared move constructor if there no user-defined 
        - destructor
        - copy constructor
        - copy assignment operator
        - move constructor
    - Implicitly declared or explicitly use `=default` to let compiler generate move assignment operator for you
        - Move assigns each base and non-static data member
        - Deleted if any base or non-static data member cannot be move assigned

    ```cpp
    struct s {
        double* data;
        S& operator=(S&& other) noexcept{
            if(this == &other) return *this;
            delete[] data;
            data = std::exchange(other.data, nullptr)
            return *this;
        }
    };
    ```
- Move Copy/Assignment Guidelines:
    - **Move constructor/assignment should be explicitly noexcept**
        - Core Guideline C.66: Make move operators no except
        - Moves are supposed to transfer resources, not allocate or acquire resources. No exceptions should be thrown.
        - Declare it `noexcept` even when it is defined as default
            - `foo(foo&&) noexcept = default`
    - **Move-from object must be left in a valid state**
        - Core Guideline C.64: A move operation should move and leave its source in a valid state
        - Prefer to leave it in the default constructed state
            - But that is not always practical

        ```cpp
        struct s{
            std::string str;
            std::size_t len;
            //Invariant: len == str.length()
            s(s&& other) noexcept : str(std::move(other.str)),
                                    len(std::move(other.len)){
                //reset other to a valid state
                other.str.clear();
                other.len = 0;
            }
        }
        ```
    - **Use `=default` when possible**
        - Core Guideline C.80: Use `=default` if you have to be explicit about using the default semantics
    
    - **Rule of 5 / Rule of 0**
        - Core Guideline C.21: If you define or `=delete` any copy, move, or destructor function, define or `=delete` them all
            - destructor
            - copy constructor
            - copy assignment operator
            - move constructor
            - move assignment operator
        - Rule of 0: If default behavior is correct for all five, let compiler do everything
        - Rule of 1: If you must define one of the five, declare all of them explicitly

### The Hidden Secret of Move Semantics



### Resources

- [模板与泛型（二]()
- [C++中的右值引用与std::move]()
- [Back to Basic: Move Semantics](https://www.youtube.com/watch?v=ZG59Bqo7qX4&t=1421s)