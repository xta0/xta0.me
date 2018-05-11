---
title: LeetCode Solutions Summary Part 1
post: layout
---

求职需要，开始刷LeetCode，之前没刷过题，不得不翻出之前算法和数据结构的文章再看一遍。LeetCode上有些题目着实难解

## 字符串

### 滑动窗口

- 最长不重复子串

如果没有刷题经验，这个问题还是很不好想的


## 重复子问题

### Problem #20

- Approach #1

```cpp
vector<string> generateParenthesis(int n) {
    if(n == 1){
        return {"()"};
    }else {
        vector<string> last = generateParenthesis(n-1);
        vector<string> ret (last);
        for(auto str : last){
            //left
            string l = "()" + str;
            ret.push_back(l);
            //right
            string r = str + "()";
            ret.push_back(r);
            //enclose
            string e = "("+str+")";
            ret.push_back(e);
        }
        //去重
        auto itor = std::unique(ret.begin(),ret.end());
        ret.erase(itor,ret.end());
        return ret;
    }   
}
```

- Approach #2

```cpp
vector<string> generateParenthesis(int n) {
    if(n == 1){
        return {"()"};
    }else {
        vector<string> last = generateParenthesis(n-1);
        vector<string> ret;
        for(auto str : last){
            string tmp;
            for(int i=0;i<=str.length();++i){
                tmp = str;
                string s = tmp.insert(i, "()");
                ret.push_back(s);
            }
        }
        std::sort(ret.begin(), ret.end());
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
        return ret;
    }
}
```

### Problem #78

## Tree

### BinaryTree

- Traversal
    - [【#144 PreOrderTraversal】](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)
    - [【#145 PostOrderTraversal】](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)
    - [【#94 InOrderTraversal】](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)