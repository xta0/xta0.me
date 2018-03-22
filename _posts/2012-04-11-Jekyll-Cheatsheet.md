---
layout: post
---

## Jeyll

### 资料

* [Jekyll](https://jekyllrb.com)
* [Liquid Syntax](http://shopify.github.io/liquid/basics/introduction/)

### Liquid Syntax

* include parameters

使用`{% include %}`是可以带参数，参数可以传递对应的`HTML`文件

```javascript
<footer>
{% include nav.html class='nav-bottom' %}
</footer>

//in nav.html
<nav>
    <ul class="nav {{include.class}}">
    ...
    </ul>
</nav>
```

### Jekyll Tips

* looping over posts

```javascript
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
```

* post categories
  * 使用`site.categories[some_key]`，在所有的 category 中，根据自定义的 key 来选出某个的 category 的所有 post
  * 使用`site.categories.some_category`，在所有的 category 中，选出某个的 category 的所有 post

```javascript
{% for post in site.categories[some_key] %}
    <a href="{{ post.url }}">
      {{ post.title }}
    </a>
{% endfor %}
```

* pretty URLs

页面链接后缀为`.html`，不显示`.html`需要在`_config.yml`中，指定:

```
permlink: pretty
```

## Build CI
