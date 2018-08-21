
[![Build Status](https://travis-ci.org/xta0/xta0.me.svg?branch=master)](https://travis-ci.org/xta0/xta0.me)

## xta0.me

This is my personal website built in Jekyll. Jekyll is a static website generator widely used for building blogs. 

## Theme

I've been reading [Yihui Xie's](https://yihui.name/) articles for quite a few days, I really enjoy it. He's a greate scientist and also a good writter. I like the desgin of his website which is pretty clean and neat. So I reproduced the theme on my own using Sass and Flexbox. So far the code has not been abstracted as a framework. If anyone interesed, feel free to open an issue or checkout the code and mess around with it.

## CI

There are lots of ways to implement CI for Jekyll, the most common way is to use github pages since its easy to deploy and maintain. For this website, I choose to use a Github webhook approach which is a little more complicated. There is another [project that demostrates how to setup a nodejs server for it](https://github.com/xta0/Github-Webhook), feel free to checkout it out. 

## Todo

- [x] Add .travis.yml
- [x] Add Webhook for CI
- [x] Add Google Analysis
- [ ] Category Page
- [ ] Make it responsive
- [ ] Add CSS animaiton effect
- [ ] English Version
- [ ] Mario Game

## License

xta0.me is available under the MIT license. See the LICENSE file for more info.
