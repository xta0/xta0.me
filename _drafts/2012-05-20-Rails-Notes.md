---

title: Rails Dev Notes
layout: post

---

<em>所有文章均为作者原创,转载请注明出处</em>

>本文为笔者Rails项目的开发笔记 

## Rails 命令

- 创建一个新Rails项目:

`% rails new movies`


## 部署Nginx+Passenger

- 安装passenger : `gem install passenger`

MacOS/Linux 会安装在~/.rvm/gems/ 中的global gem中



##View and ViewControllers

###Controller

http://localhost:3000/movies:

- 数据流转：

request -> Router -> Controller -> model ->database -> controller -> view ->controller -> browser

当访问http://localhost:3000/movies 时

server会把/movies传给router，router用来找movies资源的位置。

为此，我们要先配置config/routes.rb：

routers.rb提供了一个block：

```ruby
Rails.application.routes.draw do

		//inside block:
		//增加这段代码
		verb "url" => "name_of_controller#name_of_action"
end
```

上面代码的意思是：

- verb : get/post/update/delete
- url  : 请求的url
- name_of_controller : 页面的controller
- name_of_action : 响应这个页面请求的方法名，一个action就是controller的一个方法

```ruby
Flix::Application.routes.draw do
  	get "movies" => "movies#index"
end
```

下面需要创建movies这个controller

`% rails generate controller movies`

会在这个目录下生成文件：
app/controllers/movies_controller.rb 

```ruby
class MoviesController < ApplicationController

end
```

然后需要定义index方法：

```ruby
class MoviesController < ApplicationController
  
  def index
  
  end

end
```

这个index方法会指定一个模板用来渲染view

rails会默认在这个路径：

app/views/movies下寻找：index.html.erb

###View & ERB

erb是一模板类，它会生成最终的HTML代码

在上面目录下创建：index.html.erb：


	<h1>Movies</h1>
	<ul>
  	<li>Iron Man</li>
  	<li>Superman</li>
  	<li>Spider-Man</li>
	</ul>


这里将页面erb进行hardcode，但是如果我们想动态生成电影名称该怎么办？

首先在controller中定义一个成员变量`@movies`

然后再index方法中给@movies赋值：

```ruby
def index
  @movies = ['Iron Man', 'Superman', 'Spider-Man']
end
```

然后在erb中：


	<ul>
	<% @movies.each do |movie| %>
  		<li><%= movie %></li>
	<% end %>
	</ul>


这里有个细节是，在erb文件中使用ruby语法，需要用<% %>，这样将不在HTML上显示代码文本，如果想要显示，则需要用<%= %>标签


##Models


前面在controller的index方法中，定义的数据是静态的，hardcode的，如果想要动态的从数据库中取该如何做呢？

有这么几部：

1，创建Movie的Model

2，使用migration创建movies的数据库

3，创建几个movies并把它们存进去

4，从数据库中读取movies

5，更新数据库中movies item的属性值

6，删除movies



- 创建model：

命令为：
`% rails generate model Movie title:string rating:string total_gross:decimal`

在app/models/movie.rb下：

```ruby
class Movie < ActiveRecord::Base
end
```

在db/migrate/下创建了数据库类：

`YYYYMMDDHHMMSS_create_movies.rb`


```ruby
class CreateMovies < ActiveRecord::Migration
  def change
    create_table :movies do |t|
      t.string  :title
      t.string  :rating
      t.decimal :total_gross

      t.timestamps
    end
  end
end
```

migration文件是用来和数据库操作打交道的类

注意：model的名称为Movie是单数形式，然而数据库的名称为movies为复数，rails使用这种规则来自动连接数据和数据库。

在create_table的block中，t对应每行的object，可以指定类型和名称。有两列是个默认：

created_at和updated_at,id列也是默认的。


- 使用Migration命令行

通常情况下，rails有多个数据库，在config/database.yml中：

```ruby
development:
  adapter: sqlite3
  database: db/development.sqlite3
  pool: 5
  timeout: 5000

test:
  adapter: sqlite3
  database: db/test.sqlite3
  pool: 5
  timeout: 5000

production:
  adapter: sqlite3
  database: db/production.sqlite3
  pool: 5
  timeout: 5000

```

yml会根据开发环境选择相应的数据库。

Mirgration使用的rake命令：

`% rake -T db` ： 查看所有db的命令

注意到第一步创建好model后，指定了数据库的schema，但是并没有真正创建数据库，需要通过migrate命令创建数据库：

`% rake db :migrate`

这行命令也是去db/migrate下面找YYYYMMDDHHMMSS_create_movies.rb,然后执行change方法

`% rake db:migrate:status`

查看数据库状态：

`database: /Users/moxinxt/codingForFun/rails/flix/db/development.sqlite3`

 	Status   Migration ID    Migration Name
	--------------------------------------------------
   	up     20140712094828  Create movies

status为up，说明数据库已经在运行，此时再执行% rake db :migrate是不起作用的。

- 使用migration做增，删，改，查

对db进行CRUD操作可以都是通过model提供api接口来完成的:

`% rails console`: 启动命令行

`% Movie.all` ：这句相当于SELECT "movies".* FROM "movies"

```
Movie Load (0.3ms)  SELECT "movies".* FROM "movies"
=> #<ActiveRecord::Relation []>
```

增：

`% movie = Movie.new`

输出：

```
=> #<Movie id: nil, title: nil, rating: nil, total_gross: nil, created_at: nil, updated_at: nil>
```

rails的console实际上是ruby的irb，上面的命令，我们创建了一个object叫movie，它的类型是Movie

这里有个问题，Movie是model的类名，new是其类方法，但是我们并没有在Movie中声明id,rating等这些成员，那么它是哪里来的？

答案是，Movie继承了ActiveRecord，ActiveRecord会认为Movie这个类和数据库中movies表是对应起来的，于是将movies表中的属性动态的增加为Movie这个类的成员变量（元编程能力）。

于是我们便可以对movie的成员进行赋值：

`% movie.title = 'Iron Man'`

然后将rating和total_gross也赋值。

最后将movie这个object存入数据库

`% movie.save`

movie对象保存后，它的id，created_at,updated_at会自动赋值。

`% Movie.count` //返回数据库中数据的个数
`% Movie.first` //返回第一条数据

也可以使用Movie的构造方法：

`%  movie = Movie.new(title: 'Superman', rating: 'PG', total_gross: 134218018)`
`%  movie.save`

如果使用create方法，则不用save：

`% Movie.create(title: 'Spider-Man', rating: 'PG-13', total_gross: 403706375)`

查：

Movie的find方法，find参数是数据库的主键，这里是id

`% iron_man = Movie.find(1)`

Movied的find_by方法，参数是属性

`% spider_man = Movie.find_by(title: 'Spider-Man')`

改：

一种方法是找到对象，然后修改对象的属性：

```ruby
 iron_man = Movie.find(1)
 iron_man.title = 'Iron Man 2'
 iron_man.total_gross *= 2
 iron_man.save
```

另一种方法是直接使用对象的update接口：

`% iron_man.update(title: 'Iron Man', total_gross: 318412101)``

删：

使用Movie的destroy方法：

`% spider_man.destroy`


- 直接操作数据库

上面所有对db的操作都是通过model的接口完成的，但是我们也可以直接操作数据库：

`% rails dbconsole`

会进入sqlite的命令行：

`% sqlite>`

查看表：
`% sqlite> .tables`

查看表的schema：
`% sqlite> .schema movies`

查看内容
`% sqlite> select * from schema_migrations`

##Connecting MVC

我们chap3中controller由于没有model，因此数据是静态的，hardcode的。chap4中我们创建了基于数据库的model，这章我们将controller中的数据改为从model中获取。

原来代码为：

```ruby
def index
  @movies = ['Iron Man', 'Superman', 'Spider-Man']
end
```

改为：

def index
  @movies = Movie.all
end

改变erb的展示样式

	<h1>Movies</h1>

	<ul>
	<% @movies.each do |movie| %>
  	<li>
    	<strong><%= movie.title %></strong> (<%= movie.rating %>):
    	<%= number_to_currency(movie.total_gross) %>
  	</li>
	<% end %>
	</ul>


##Migration

当数据库中的表创建好后，我们可能忘记了某个属性，要为这个表再增加一个属性。这时候就要创建一个新的mirgation。

1，创建一个新的migration文件，对movies这个表增加两个fields

2，更新当前movies表中的数据

3，修改erb，展示新的属性


给db新增一个field：

`% rails g migration AddFieldsToMovies description:text released_on:date`

这会在 db/migrate 下
创建一个新的migration文件：
`YYYYMMDDHHMMSS_add_fields_to_movies.rb`：

```ruby
class AddFieldsToMovies < ActiveRecord::Migration
  def change
    add_column :movies, :description, :text
    add_column :movies, :released_on, :date
  end
end
```

generator如何知道我是想给哪个表增加field呢？

答案还是在命名上，AddXXXToYYY : YYY就是要修改的表名

此时查看db的状态：

`% rake db:migrate:status`

	database: db/development.sqlite3

 	Status   Migration ID    Migration Name
	--------------------------------------------------
   	up     20120914152106  Create movies
  	down    20120916151045  Add fields to movies

这是由于db还没有run，我们需要让它run起来：

`% rake db:migrate`

注意这个命令实际上是去遍历db/migrate下的文件，看看哪个没有run，将它run起来

修改movie表中数据的状态

进入命令行：`% rails c`

`% reload! `
=> true

reload的意思是保证数据是最新的。

这时随便找一条数据看看：

`% movie = Movie.find(1)`

```
#<Movie id: 1, title: "Iron Man 2", rating: "PG-13", total_gross: #<BigDecimal:7f968e4f4e90,'0.636824202E9',9(27)>, created_at: "2014-07-12 13:03:10", updated_at: "2014-07-12 13:43:49", description: nil, released_on: nil>
```

发现新建的两个字段都为空

```ruby
movie.description = "Tony Stark builds an armored suit to fight the throes of evil"movie.released_on = "2008-05-02"
movie.save
```
为其赋值并保存.更新erb：

	<p>
  		<%= movie.description %>
	</p>
	<p>
  		<%= movie.released_on %>
	</p>

##Helper

经常erb中耦合的一些helper方法

- 使用Rails自带的ViewHelper:

`% rails console` 进入命令行

```ruby
helper.pluralize(0,'Movie')
helper.pluralize(1,'Movie')
helper.pluralize(2,'Movie')

```

```ruby
helper.truncate("aaa",length:50, separator:'')
```

- 在ERB里穿插各种ruby代码，比较乱，因此可以在自己的helper中定义好方法

例如：

当创建Controller时，默认创建了view helper：

```ruby
module EventsHelper

	def display_price(event)
	
		if event.free?
			"<strong>Free!</strong>".html_safe
		else
			event.price.to_s
		end

	end

end

```
在event.html.erb中默认import了这个EventsHelper类，因此可以直接使用,注意ViewHelper中返回的string不会作为html解析，如果想返回html字符串，需要加html_safe

关于时间格式修改，通常情况下，event的starts_at字段被声明为datetime类型，在输出的时候实际上是:

`event.starts_at.to_s`

` Sat, 15 Nov 2014 00:00:00 UTC +00:00`

datetime的格式为：

```
irb(main):005:0> Time::DATE_FORMATS
=> 
{:db=>"%Y-%m-%d %H:%M:%S", 
:number=>"%Y%m%d%H%M%S", 
:nsec=>"%Y%m%d%H%M%S%9N", 
:time=>"%H:%M", 
:short=>"%d %b %H:%M", 
:long=>"%B %d, %Y %H:%M", 
:long_ordinal=>#<Proc:0x007ffdbbc65820@/Users/moxinxt/.rvm/rubies/ruby-2.1.1/lib/ruby/gems/2.1.0/gems/activesupport-4.1.0.rc1/lib/active_support/core_ext/time/conversions.rb:12 (lambda)>, 
:rfc822=>#<Proc:0x007ffdbbc657f8@/Users/moxinxt/.rvm/rubies/ruby-2.1.1/lib/ruby/gems/2.1.0/gems/activesupport-4.1.0.rc1/lib/active_support/core_ext/time/conversions.rb:16 (lambda)>, 
:iso8601=>#<Proc:0x007ffdbbc657d0@/Users/moxinxt/.rvm/rubies/ruby-2.1.1/lib/ruby/gems/2.1.0/gems/activesupport-4.1.0.rc1/lib/active_support/core_ext/time/conversions.rb:20 (lambda)>}

```
`event.starts_at.to_s` 默认调用的是[:default]的格式

如何自定义时间格式？可以为Time::DATE_FORMATS增加一个key，value为自己想要的格式:

```
Time::DATE_FORMATS[:default] = "%B %d, %Y at %I:%M %p"

```
这样会覆盖default格式

如何初始化？这种全局数据结构的修改应该在App启动前执行一次

在Rails的 conifg/initializers/ 下面的所有文件都会在启动前执行一次，我们可以创建一个time_format.rb的文件，将`Time::DATE_FORMATS[:default] = "%B %d, %Y at %I:%M %p"`拷进去


### Show Page

对于events/1 这种详情页，在route中定义如下：

```ruby
get "events/:id" => "events#show", as: :events_detail

```

对应controller中的show方法

```ruby
def show
	@event = Event.find(params[:id])
end
	
```
参数存在了params里面

### Link Page

使用`% rake routes` 打印出搜索的url：

```
Prefix Verb   URI Pattern                    Controller#Action

events 		  GET    /events(.:format)       events#index
              GET    /events/:id(.:format)   events#show

```
其中Prefix Verb比较重要，使用 `% rails console`进入命令行

```
irb(main):001:0> app.events_path
=> "/events"

irb(main):008:0> app.events_url
=> "http://www.example.com/events"

```

因此我们可以使用 Prefix Verb + "_" + "path" 来返回path

如果在erb中加入link，可以用link_to方法

```ruby
<%= link_to 'Back', events_path %>

```

但是我们发现 events#index 和events#show都对应events,如何区分？

方法如下，在route.rb：

```ruby
get "events", to: "events#index", as: :events
get "events/:id" => "events#show", as: :event
  
```

然后再运行 `% rake routes` : 

```ruby
 events GET    /events(.:format)              events#index
 event GET     /events/:id(.:format)          events#show

```

我们便可以使用event_path,注意event路径要传入一个id作为参数：

```
irb(main):002:0> app.event_path(e.id)
=> "/events/1"

```

指定默认路径：

```ruby
root "statuses#index"

```
### Test 

- 安装 RSpec & Capybara

```ruby
group :test, :development do
	gem "rspec-rails"
end

group :test do
	gem "capybara"
end

```

```
rails  rails g rspec:install

```

- 写specfile

测试内容展示正确与否：

```ruby
require "spec_helper"

describe "Viewing the list of events" do 

	it "Shows the events" do 

		##mock data:
		e = Event.new
		e.name = "Brain Storm"
		e.content = "Discuss How to build a to-do list"
		e.location = "my house"
		e.starts_at = 10.days.from_now
		e.save

		visit events_url
		expect(page).to have_text("1 Event")
		expect(page).to have_text(e.name)
	end

end

```

测试连接跳转正确与否

```ruby
it "From event index page to detail page " do 

	##mock data:
	e = Event.new
	e.name = "Brain Storm"
	e.content = "Discuss How to build a to-do list"
	e.location = "my house"
	e.starts_at = 10.days.from_now
	e.save

	visit events_url

	click_link(e.name)

	expect(current_path).to eq(event_path(e.id)) 
end

```

- 注意RSpec跑在test数据库中，因此，需要将test数据库和development数据库同步表结构：

```
% rake db:test:prepare
```


seeds.rb:可以为数据库初始化一些数据

```ruby
Event.create!( [ {:name:"a", :location:"b"}, {}, {}, ] )

```

然后运行：

```
% rake db:seed

```

### Creating Forms

增加edit path：

`/events/1/edit` 对应 routes.rb：

`get "events/:id/eidt" => "events#edit", as: :event_edit`


在edit.html.erb中插入模板:

```ruby
<%= form_for(@event) do |f| %>
	
	<p>
		<%= f.label :name %></br>
		<%= f.text_field :name %>
	</p>

	<p>
		<%= f.label :content %></br>
		<%= f.text_area :content %>
	</p>

	<p>
		<%= f.label :location %></br>
		<%= f.text_field :location %>
	</p>

	<p>
		<%= f.label :starts_at %></br>
		<%= f.datetime_select :starts_at %>
	</p>

	<p>
		<%= f.submit %>
	</p>

<% end %>

```

- 注意form_for前面的erb tag为 "<%= "

- @event会传给form_for方法，因此，event的属性会被form_for知晓

- 在给text_field或者text_area等元素赋值时，无需访问@event的property，根据property的名字符号即可访问

### Updating Forms


然后再controller中增加update方法

```ruby
def update
	@event = Event.find(params[:id])
	@event.update(params[:event])
end
	
```
- @event必须重新赋值，在rails中，成员变量的生命周期和只在当前方法内

- 需要被更新的数据存放到了hash中 : params[:event]
但是这样写会有风险，如果params[:event]中的数据被篡改了，那么会有安全风险 

改为：

```ruby
def update
	
	@event = Event.find(params[:id])
	safe_form = params[:event].require(:event).permit(:name,:content,:location,:starts_at)
	@event.update(params[:event])
end

```
require意思是确保:event这个key存在，permit用来允许访问的key，这样safe_form中只会有name,content,location,starts_at几个字段

- 统一route：

rails对增删改查的route有默认的表示：`resources :events`

- create form:


对于创建form的操作，路径是/events/new，因此，controller要有个new方法。

```ruby
def new
	@event = Event.new
end
```

点击create后，会产生一个post请求：

` POST   /statuses(.:format)            statuses#create`

对应controller的create方法

```ruby
def create
	@event = Event.new(event_params)
	@event.save

	redirect_to event_path(@event.id)

end
	
```

### PARTIAL

被复用的view模板叫做partial，用_开头，调用的时候：

`<%= render 'form' %>`
 
render 方法不需要加_form

### DELETE

删除是POST请求，因此没有界面展示，对应的route为：

`event DELETE /events/:id(.:format)          events#destroy`

调用controller的destroy方法：

```ruby
def destroy
	@event = Event.find(params[:id])
	@event.destroy

	redirect_to events_path
end

```
link_to点击默认的是用GET操作加载path，如果path不是GET，要告诉link_to：

```ruby
<%= link_to 'Delete', event_path(@event.id),method: :delete, data:{confirm:'Are you sure?'} %>

```

### Query

如果列表要返回满足query条件的数据，该怎么做？

做法是使用Event的方法，这些方法是ActiveRecord的方法，可以生成sql，比如按时间倒排：

`Event.order("starts_at desc")`

筛选：

`Event.where("starts_at >= ?", Time.now).order("starts_at")`

等等,注意的是，不要将query放到controller中，这个逻辑属于model，可以给model增加方法来实现

### Validation

model层面的数据校验：

```ruby
validates :name, presence:true
```
在rails console中，可以测试:

```ruby
e = Event.new
e.valid? 

=> false

e.errors
e.errors.full_message
e.errors[:name]

=> ["can't be blank"]

```
valid方法会触发model去校验数据

model的valid方法会影响controller的create和edit

修改controller这两个方法：

```ruby
def update

	@event = Event.find(params[:id])

	if @event.update(event_params)
	
		redirect_to event_path(@event.id)
	else
		render :edit
	end
	
end

def create

	@event = Event.new(event_params)

	if @event.save
	
		redirect_to event_path(@event.id)
	else
		render :new
	end

end

```

### 创建Resource

创建resource，相当于创建一个新的feature，包括MVC：

```
rails g resource user name:string email:string password:digest

```


### one-many

如果Event对应多个registration，那么对于Event,声明为:

```ruby
has_many :registrations
```
这样便给Event增加了一个property，叫做registraions,可以访问:

```ruby
e.registrations
```

对于registration来说，声明为:

```ruby
belongs_to :event
```
这样便给registration增加一个property，叫做event，可以访问:

```ruby
r.event
```

原理是，registration会增加一个外键叫做event_id，对应event的主键。

当某个Event要获取对应的registration时，它实际上是拿自己的event_id去registration表里查找对应的event_id

### nested resouse

当Event和Registration建立起has_many关系后，一个Event就可以对应多个Registration，而Registration又是一个完整的MVC，本身也可以支持增删改查，同样可以通过

```
rails g resource user name:string email:string password:digest

```
创建资源，只不过它的URL格式是:

0.0.0.0:3000/events/1/registrations

可见它依赖Event传过来的Id，为了生成上面类似的path，可以将registrations做如下声明：

```ruby
resources :events do 
    resources :registrations
end

```
生成的routes为：

```
 event_registrations    GET    /events/:event_id/registrations(.:format)          registrations#index
                        POST   /events/:event_id/registrations(.:format)          registrations#create
 new_event_registration GET    /events/:event_id/registrations/new(.:format)      registrations#new
edit_event_registration GET    /events/:event_id/registrations/:id/edit(.:format) registrations#edit
     event_registration GET    /events/:event_id/registrations/:id(.:format)      registrations#show
                        PATCH  /events/:event_id/registrations/:id(.:format)      registrations#update
                        PUT    /events/:event_id/registrations/:id(.:format)      registrations#update
                        DELETE /events/:event_id/registrations/:id(.:format)      registrations#destroy

```
### session

用户登录后，会通过浏览器生成cookie，通过session带过来，Rails要保存cookie，从而识别该用户是否登录，但是cookie是不能存到数据库中的。

- 创建session的url_path

增加sesssion的url_path到routes.rb

```ruby
  resources :sessions 

```

但是通过这个命令生成的url_path，包括了index，和一些需要id的不相关的path，因此我们要换一个命令:

```
resource:session
```

将上面命令改为单数形式:

```

session 				POST   /session(.:format)              sessions#create

new_session 			GET    /session/new(.:format)          sessions#new

edit_session 			GET    /session/edit(.:format)         sessions#edit
                       
                       GET    /session(.:format)              sessions#show
                       
                       PATCH  /session(.:format)              sessions#update
                       
                       PUT    /session(.:format)              sessions#update
                        
                       DELETE /session(.:format)              sessions#destroy

```

少了index，和一些需要id的path

