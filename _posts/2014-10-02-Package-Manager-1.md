---
title: 各种Package Manager
list_title: Package Manager
layout: post
categories: [ Tools ]
updated: "2018-09-10"
---

每种语言或者框架都有配套的包管理系统，本质上他们的作用都一样：**环境隔离与版本控制**。只是每种生态都爱自己造轮子，所以每次配环境都是一场灾难

## HomeBrew

[HomeBrew](http://brew.sh/)是MAC下的包管理系统。它会将`/usr/local`初始化为git环境:
- bin: 用于存放所安装程序的启动链接（相当于快捷方式）
- Cellar: 所以brew安装的程序，都将以[程序名/版本号]存放于本目录下
- etc: brew安装程序的配置文件默认存放路径
- Library: Homebrew 系统自身文件夹

## RVM

[rvm](https://rvm.io/rvm/basics)是Ruby的版本管理工具，项目中可以通过Rvm来创建多套ruby环境。

- rvm路径：`~/.rvm`
- 管理ruby：

```
//rvm 管理的ruby斑斑
% rvm list
//使用系统ruby版本
% rvm use system
//使用rvm管理的default版本
% rvm use default
```

- 管理gemset

```
//创建gemset
% rvm gemset create xxx

//使用某个gemset
% rvm gemset use xxx

//查看所有的gemset
% rvm gemset list
```
- [More on RVM](https://ruby-china.org/wiki/rvm-guide)

## Gem

Gem是Ruby的包管理工具，可以通过Gem为Ruby程序提供第三方依赖包。

- 项目中需要通过Gemfile来引入第三方包:

```ruby
gem 'rails', '4.1.0.rc1'
gem 'sqlite3'
gem 'mysql2'
```

### Gemspec 

- 创建Gem包需要提供gemspec, gemspec是一段Ruby代码:

```ruby
Gem::Specification.new do |s|
  s.name        = 'hola'
  s.version     = '0.0.0'
  s.date        = '2010-04-28'
  s.summary     = "Hola!"
  s.description = "A simple hello world gem"
  s.authors     = ["Nick Quaranto"]
  s.email       = 'nick@quaran.to'
  s.files       = ["lib/hola.rb"]
  s.homepage    =
    'http://rubygems.org/gems/hola'
  s.license       = 'MIT'
end
```

- gem包会安装到`INSTALLATION DIRECTORY`下
- 查看gem配置: `%gem environment`

```
RubyGems Environment:
  - RUBYGEMS VERSION: 2.2.2
  - RUBY VERSION: 2.1.1 (2014-02-24 patchlevel 76) [x86_64-darwin13.0]
  - INSTALLATION DIRECTORY: /Users/xt/.rvm/gems/ruby-2.1.1
  - RUBY EXECUTABLE: /Users/xt/.rvm/rubies/ruby-2.1.1/bin/ruby
  - EXECUTABLE DIRECTORY: /Users/xt/.rvm/gems/ruby-2.1.1/bin
  - SPEC CACHE DIRECTORY: /Users/xt/.gem/specs
  - RUBYGEMS PLATFORMS:
    - ruby
    - x86_64-darwin-13
  - GEM PATHS:
     - /Users/xt/.rvm/gems/ruby-2.1.1
     - /Users/xt/.rvm/gems/ruby-2.1.1@global
  - GEM CONFIGURATION:
     - :update_sources => true
     - :verbose => true
     - :backtrace => false
     - :bulk_threshold => 1000
     - :sources => ["https://rubygems.org/"]
  - REMOTE SOURCES:
     - https://rubygems.org/
  - SHELL PATH:
     - /Users/xt/.rvm/gems/ruby-2.1.1/bin
     - /Users/xt/.rvm/gems/ruby-2.1.1@global/bin
     - /Users/xt/.rvm/rubies/ruby-2.1.1/bin
     - /Users/xt/.rvm/bin
     - /usr/local/heroku/bin
     - /usr/bin
     - /bin
     - /usr/sbin
     - /sbin
     - /usr/local/bin
     - /opt/X11/bin
```

### 管理Gem Sources

- 查看当前的Gem源

```shell
$gem sources -i

*** CURRENT SOURCES ***

https://ruby.taobao.org/
http://gems.alibaba-inc.com/
```
- 增删Gem源

```shell
#删除
$gem sources -r https://ruby.taobao.org/
#增加
$gem sources -a https://rubygems.org/
```

### Resources

- [How to: Adding Ruby Gems and Gem Sources](https://www.atlantic.net/cloud-hosting/how-to-adding-ruby-gems-gem-sources/)

## CocoaPods

[CocoaPods](http://cocoapods.org/)是iOS的包管理工具,用法和语法和Gem基本一样。

### 引入Podfile

```ruby
pod 'AFNetworking', 
pod 'Paper'
```

### 制作podspec

- 需要MIT LICENSE
- 需要README.md
- 需要项目的Example
- 代码打tag,push到仓库
- 创建podsepcs
  - `pod spec create PROJ_NAME`

> 2014年5月20日之后，cocoaPods不再接受pull Request的提交方式，而转为用[trunk](http://blog.cocoapods.org/CocoaPods-Trunk/)。使用trunk需要cocoapods的版本大于0.33。

- 注册trunk:  
  - 命令为: `pod trunk register orta@cocoapods.org 'Orta Therox' --description='macbook air'`
    - `pod trunk register JohnDoe@gmail.com 'John' --verbose`
  - 注册成功后,会返回下面信息:

  ```shell
  [!] Please verify the session by clicking the link in the verification email that has been sent to JohnDoe@gmail.com
  ``` 

- 在邮箱激活trunk, 查看注册信息:
  - `pod trunk me`

- 提交pod,在podsepc的目录下: :
  - `pod spec lint NAME.podspec --verbose --allow-warnings`
  - `pod trunk push NAME.podspec --verbose --allow-warnings`
  - 成功后回返回podspec的json格式的url

- 含lib的podspec  

```ruby
pod spec lint --verbose --use-libraries --skip-import-validation --allow-warnings
pod trunk push --verbose --use-libraries --skip-import-validation --allow-warnings
```

### 删除项目中的Podfile

- 安装两个gem包

```shell
$ sudo gem install cocoapods-deintegrate
$ sudo gem install cocoapods-clean
```

- 删除pod相关依赖

```shell
$ cd proj
$ pod deintegrate
$ pod clean
```

### 管理Pod Repo

```shell
pod repo list #查看pod源列表
pod repo add xxx #增加源
pod repo remote xxx #删除源
```

### Resources

- [Uninstalling PodFile form the Xcode project](https://medium.com/@satishios25/uninstalling-podfile-form-the-xcode-project-7e96874ae7f4)


## NVM

NVM是node.js的包管理系统

- 解决VSCode命令行问题
  - [The Reason](https://github.com/Microsoft/vscode-docs/blob/master/docs/editor/integrated-terminal.md#why-is-nvm-complaining-about-a-prefix-option-when-the-integrated-terminal-is-launched)
  - [The Fix](https://github.com/creationix/nvm/issues/855)

## Lua

[LuaRocks]("http://luarocks.org/")是Lua的包管理系统，对于管理lua的module很方便，类似Ruby的gem。如果是Mac安装的时候可以使用HomeBrew，也可以手动安装。手动安装要费一点劲儿，由于我机器上已经安装了Lua5.2，装LuaRocks时怎么都不成功，原因是不兼容Lua5.2的Module，解决办法是，回退到Lua 5.1版本。

LuaRock的用法和gem相同:

- 搜索: `luarocks search json`
- 安装: `luarocks install json4lua` 
- 查看: `luarocks list`
- 查看某个包: `luarocks show json4lua`
- 删除某个包: `luarocks remove json4lua`

## Python

### Virtualenv

用来隔离python开发环境以及包管理，必备良药

- install  & config

```
#install using pip
pip3 install virtualenv
#create enviroment
> virtualen proj
#create enviroment with python path
> virutalen proj -p python_path
#activate & deactive
> source bin/activate
> deactivate
```
由于改变了python路径，使用vscode可能会导致lint出错，配置新的python路径

```json
<!-- .vscode/settings.json -->
{
    "python.pythonPath": "${workspace}/bin/python",    
    "python.linting.pylintEnabled": false,
    "python.linting.enabled": true,
    "python.linting.pylintPath": "${workspace}/lib/python2.7/"    
}
```

### Anaconda

Anaconda是另一个Python的包管理软件，基本用法如下

- 查看当前所有Python环境 
  - `conda info --envs`
  
  ```
  # conda environments:
  #
  base                     /home/taox/anaconda3
  py36                  *  /home/taox/anaconda3/envs/py36
  ```

- 创建一个新的Python环境 
  - `conda create --name py36 python=3.6`

- 切换/注销Python环境
  - `conda activate py36`
  - `conda deactivate`

- 删除一个Python环境
  - `conda remove --name py36`

- 查看当前环境已安装的包
  - `conda list`
- 安装/删除包
  - `conda install -n py36 numpy`
  - `conda uninstall -n py36 numpy`

