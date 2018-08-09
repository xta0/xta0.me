
- 编辑内容后需要使用`rake generate`重新部署

- `rake preview`本地预览页面

- 部署到github
	- github repo: `username.github.io`
	- `rake setup_github_pages`
	- `rake deploy`

- 托管源码到github
	- `git add .`
	- `git commit -m 'message'`
	- `git push origin source`  

- 修改head.html:
	- 替换:`<script src="{{site.baseurl}}//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>` 为`  <script src="{{site.baseurl}}//libs.baidu.com/jquery/1.9.1/jquery.min.js"></script>`

- 新建blog
	- `rake new_post["title"]`

- 新建单页面
	- `rake new_page[title]`
		- 路径：`creates/source/title/index.markdown`
	-  `rake new_page[title/page.html]`
		- 路径：`creates/source/title/page.html`

- 添加评论
	- 在`config.yml`中添加自定义key:`duoshuo_comment: true`
	- 在`source/layout/post`下面修改：

```
{% if site.duoshuo_comments == true %}
  <section>
    <h1>Comments</h1>
    <div id="disqus_thread" aria-live="polite">{% include post/duoshuo_comment.html %}</div>
  </section>
{% endif %}

```  

- 主题
	- 在github上找到主题并clone
	- `rake install['themename']`
	- `rake generate` 
	- 重新生成的主题会将原来添加的评论内容覆盖掉



