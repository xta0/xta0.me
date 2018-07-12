- Pure functions
    - have no hidden inputs.
    - have no hidden outpus.
    - do not execute any other side effect.

- Higher Order Functions

```cpp
// ((T->T),[T]) -> [T]
template <typename F, typename T>
vector<T> transform_vec(F f, const vector<T>& xs){
    vector<T> ys;
    ye.reserve(xs.size());
    std::transform(begin(xs),end(xs),back_inserter(ys),f);
    return ys
}
///usr/bin/time --format="%e s" ./a.out
```

- 



### Resources

- [fplus](https://github.com/Dobiasd/FunctionalPlus/)
- [fplus API search](http://www.editgym.com/fplus-api-search/)