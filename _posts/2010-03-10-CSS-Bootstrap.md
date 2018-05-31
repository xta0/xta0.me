---
list_title: CSS Bootstrap
layout: post
---

## Bootstrap

- [Bootstrap Instroduction](https://getbootstrap.com/docs/4.0/getting-started/introduction/)
	- One single CSS file + One single Javascript file

### Grid System

- Bootstrap将每个容器分成12个column，每个显示区域可以指定占有column的数量。

```html
<!-- 使用boostrap container -->
<div class="container">
   <!-- div横向展示 -->
    <div class="row">
        <!-- column-large-2 占两个格子 -->
        <div class="col-lg-2 pink">COL LG #2</div>
        <div class="col-lg-8 pink">COL LG #10</div>
        <div class="col-lg-2 pink">COL LG #2</div>
    </div>
    <div class="row">
        <div class="col-lg-4 pink">COL LG #4</div>
        <div class="col-lg-4 pink">COL LG #4</div>
        <div class="col-lg-4 pink">COL LG #4</div>
    </div>
</div>
```

- 四种不同尺寸

<table>
    <thead>
        <tr>
            <th></th>
            <th> 手机屏幕(<768px)</th>
  			  <th> 平板屏幕(≥768px)</th>
  			  <th> 中等屏幕PC (≥992px)</th>
  			  <th> 大屏幕PC(≥1200px)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th class="text-nowrap" scope="row">Grid behavior</th>
            <td>Horizontal at all times</td>
            <td colspan="3">Collapsed to start, horizontal above breakpoints</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Container width</th>
            <td>None (auto)</td>
            <td>750px</td>
            <td>970px</td>
            <td>1170px</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Class prefix</th>
            <td><code>.col-xs-</code></td>
            <td><code>.col-sm-</code></td>
            <td><code>.col-md-</code></td>
            <td><code>.col-lg-</code></td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row"># of columns</th>
            <td colspan="4">12</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Column width</th>
            <td class="text-muted">Auto</td>
            <td>~62px</td>
            <td>~81px</td>
            <td>~97px</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Gutter width</th>
            <td colspan="4">30px (15px on each side of a column)</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Nestable</th>
            <td colspan="4">Yes</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Offsets</th>
            <td colspan="4">Yes</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Column ordering</th>
            <td colspan="4">Yes</td>
        </tr>
    </tbody>
</table>

- 不同比例间缩放

```html
<div class="container">
    <div class="row">
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
    </div>
</div>
```

1. 第一层`div`类型为bootstrap容器
2. 第二层`div`类型为`row`，子元素横向布局
3. 第三层`div`有三个类型，分别为
	- 大屏下和中屏下，每个元素占屏幕的3个格子，1/4
	- 小屏下，每个元素占6个格子

- 嵌套

```html
<div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB
    <div class="row">
        <div class="col-lg-6 orange">FIRST HALF</div>
        <div class="col-lg-6 orange">FIRST HALF</div>
    </div>
</div>
```
1. 嵌套父容器遵从12格子等分
2. 嵌套子元素按12各自划分

  

## Bootstrap 4

- New to boostrap 3
    - Moved from Less to Saas
    - As of V3 a Sass port was created and maintained
    - Sass is favored in the web dev community
    - Use of Libsass to compile faster
- Improved Grid System & Layout
    - Changes in the underlying architechure
    - `rem` & `em` units instead of pixels
    - New `-xl` tier for extra large screens
    - Grid now uses flexbox
    - No more `offset-md-x`, Now use margin auto classes


 