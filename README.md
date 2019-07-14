
[![Build Status](https://travis-ci.org/xta0/xta0.me.svg?branch=master)](https://travis-ci.org/xta0/xta0.me)

## xta0.me

This is my personal website built in Jekyll. Jekyll is a static website generator widely used for building blog applications. 

## Theme

I've been reading [Yihui Xie's](https://yihui.name/) articles for a while, and really enjoy reading his articles. I like the desgin of his website that looks clean and neat. So I reproduced the theme using Sass and Flexbox. So far, the code has not been abstracted as a framework. If anyone interesed, feel free to open an issue or checkout the code and play around with it.

## CI

There are lots of ways to implement CI for Jekyll, the most common way is to use github pages since it's easy to deploy and maintain. For this website, I choose to use a Github webhook which is a little more complicated. There is another [project that demostrates how to setup a nodejs server for it](https://github.com/xta0/Github-Webhook), feel free to checkout it out. 

## Todo

- [x] Add .travis.yml
- [x] Add Webhook for CI
- [x] Add Google Analysis
- [ ] Add sidebar
- [ ] Add Pagination
- [ ] Add category page
- [ ] Make it responsive
- [ ] Add CSS animaiton effect
- [ ] English Translation

## License

xta0.me is available under the MIT license. See the LICENSE file for more info.
