---
layout: post
list_title: 数据结构基础 | Data Structure | 数组 | Array
title: 向量
mathml: true
categories: [DataStructure]
---

### 基本概念

- 所谓向量采用<mark>定长</mark>的一维数组存储结构
- 主要特性
	- 元素类型相同
	- <mark>元素顺序的存储在连续的存储空间中，每个元素有唯一的索引值</mark>
	- 使用常数作为向量长度
- 数组存储
- 读写元素很方便，通过下标即可指定位置
	- 只要确定了首地址，线性表中任意数据元素都可以<mark>随机访问</mark>

- 元素地址计算

```
Loc(ki) = Loc(k0)+c*i, c=sizeof(ELEM)
```

#### 顺序表类定义

```cpp
class Array:public List<T>{
	private:
		T* list;
		int maxSize; //向量实际的内存空间
		int curLen; //向量当前长度，这个值小于maxSize，要预留一部分存储空间，放置频繁的内存开销
		int position;
	public:
		Array(const int size){
			maxSize = size;
			list = new T[maxSize];
			curLen = position = 0;
		}
		~MyList(){
			delete []list;
			list = nullptr;
		}
		void clear(){
			delete []list;
			list = nullptr;
			position = 0;
			list = new T[maxSize];
		}
		int length();
		bool append(const T value);
		bool insert(const int p, const T value);
		bool remove(const int p);
		bool setValue(const int p, const T value);
		bool getValue(const int p, T& value);
		bool getPos(int &p, const T value);
}
```

### 顺序表上的运算

- 插入运算

```cpp
template<class T>
bool Array<T>::insert(const int p, const T value){
	if(curLen >= maxSize){
		//重新申请空间
		//或者报错
		return false;
	}
	if(p<0 || p>=curLen){
		return false;
	}
	for(int i=curLen;i>p;i--){
		list[i] = list[i-1];
	}
	list[p] = value;
	curLen ++;

	return true;
}
```
- 删除运算

```cpp
template<class T>
bool Array<T>::delete(const int p){
	if(curLen == 0){
		return false;
	}
	if(p == 0 || p>=curLen){
		return false;
	}
	for(int i=p; i<curLen-1; i++){
		list[i] = list[i+1];
	}
	curLen--;

	return true;
}
```

- 表中元素的移动
	- 插入：移动<mark>n-i</mark>个元素
	- 删除：移动<mark>n-i-1</mark>个元素
	- <mark>时间复杂度为O(n)</mark>


## 数组相关的问题

### K-Sum问题

- [1. Two Sum](https://leetcode.com/problems/two-sum/description/)
- [15. 3Sum](https://leetcode.com/problems/3sum/description/)
- [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/description/)
- [18. 4Sum](https://leetcode.com/problems/4sum/description/)

### 最优SubArray/SubSequence 问题

- [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/)
- [325. Maximum Size Subarray Sum Equals k]()
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/description/)




## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)










