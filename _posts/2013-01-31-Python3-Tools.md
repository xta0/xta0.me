---
list_title: Python 3 工具 | Tools
layout: post
title: Python 3 Tools
---

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


### Resource

- [Virtualenv Doc](https://docs.python.org/3/library/venv.html)
