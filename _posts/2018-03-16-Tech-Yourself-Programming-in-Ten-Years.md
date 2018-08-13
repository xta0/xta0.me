---
layout: post
list_title: 十年学会编程 | Teach Yourself Programming in Ten Years 
title:  Teach Yourself Programming in Ten Years
categories: [Life,Translation]
---

> Chinese translation of Peter Norvigs's essay [Teach Yourself Programming in 10 Years](http://norvig.com/21-days.html)

### 为什么大家都急于求成

每当走进各大书店总能看到一些诸如《24小时自学Java开发》之类的书籍，类似的还有几小时内学会C,Ruby,SQL,Algorthms等等。如果使用亚马逊的高级索索功能[title: teach, yourself, hours, since: 2000](http://www.amazon.com/gp/search/ref=sr_adv_b/?search-alias=stripbooks&unfiltered=1&field-keywords=&field-author=&field-title=teach+yourself+hours&field-isbn=&field-publisher=&node=&field-p_n_condition-type=&field-feature_browse-bin=&field-subject=&field-language=&field-dateop=After&field-datemod=&field-dateyear=2000&sort=relevanceexprank&Adv-Srch-Books-Submit.x=16&Adv-Srch-Books-Submit.y=5) 可以找到512种这类书籍，在搜索结果的前十名中有九本是编程类书籍（其它的是关于记账类书籍）。如果把搜索词" yourself"换成"learn"，或者把"days"换成"hours"，搜索结果也是类似的。

从上面的案例中我们能出的结论是，要么大家都非常着急的想学编程，或者学习编程和其它事情相比是一件很容易的完成的事。Felleisen et al.在它的[《How to Design Programs》](http://felleisen.org/matthias/)一书中也提到了这个问题，当然他们也是不认同这种思路的，他们的原话是："糟糕的编程很简单，傻瓜也可以在21天内学会，哪怕他们傻到家"。风靡网络的漫画公司[Abtruse Goose](https://abstrusegoose.com/249)也对此冷嘲热讽了一把，漫画如下

<img class="md-img-center" src="/assets/images/2018/03/ars_longa_vita_brevis.png">

让我们来一起分析一下《24小时学会C++》意味着什么

- **自学**：在24小时内你甚至没有时间给你写几段像样的，有难度的程序，更不用说从中学习什么了。你也没有时间和有经验的C++程序员工作，不会了解真正的C++项目是怎样的。简而言之就是你没有时间学到很多东西，因此书中只能浅显的让你熟悉下语法，没法做更深入的讲解。就像 Alexander Pope 曾经说过的，浅尝辄止的学习是一件非常危险的事情。

- **C++**: 在24小时内只够你学习一些基本的C++语法（假设你已经有过其它编程语言的经验），但是无法学到该如何使用这门语言。简单的说，如果你曾经是一名BASIC程序员，你可能会学会如何将原来的BASIC代码按照C++的语法重写一遍，但却不清楚为什么要用C++，也不清楚这门语言的优点（或者缺点）在哪里。[Alan Perlis](http://pu.inf.uni-tuebingen.de/users/klaeren/epigrams.html)曾经说过，"如果一种语言不能改变你对编程的理解，那么这门语言就不值得学习"。现实生活中，一种可能的情况是，为了完成某种任务你需要快速学一点C++来对接某个三方库，但是这种并不是真正的学习某种语言，只是为了完成某种任务。

- **24小时内**: 不幸的是，24小时时间实在是太短了，下一小节会讨论这个问题

### 给自己十年的时间来学习编程

很多学者已经证明了如果某个人想要成为某个领域的专家，需要差不多10年的时间，这些领域包括象棋，作曲，电报通信，绘画，弹钢琴，游泳，网球 等等。在这10年中，最重要的是保持<em>刻意练习</em>: 不仅仅是不断重复，还要不断挑战自己去完成超出目前能力的任务，然后总结自己的表现，改正错误后继续重复。这件事貌似没有捷径，即使像莫扎特这种4岁就成为音乐奇才的人也花了13年才创作出世界级的音乐。另一个领域的例子是Beatles，他们似乎在1964年凭借一系列热门单曲和其在艾德沙利文秀（The Ed Sullivan show）上的演出一炮而红，但是你也许不知道，他们早在1957年就在利物浦和汉堡两地进行小规模演出了，而在此之前的非正式演出更是不计其数，即使在他们走红之后，他们第一部最具影响力的唱片《Sgt. Peppers》也是在1967年才发布。

Malcolm Gladwell 曾经提出过一个很流行的理论，他聚焦于一万小时练习理论，并非10年。Henri Cartier-Bresson (1908-2004) 曾经说过，"你的前一万张照片是最糟糕的"（他并没有预料到数码相机的出现，人们能在1周内完成一万张的拍摄）。真正的专家们可能会在某个领域投入自己一生的时间，Samuel Johnson (1709-1784) 曾经说，"想要在某个领域做到卓越的唯一办法就是终身投入"。Hippocrates (c. 400BC) 有一段著名的引用"ars longa, vita brevis"，这句话出自"Ars longa, vita brevis, occasio praeceps, experimentum periculosum, iudicium difficile"，翻译过来是说，"生命很短暂，但是技艺却很高深，机遇转瞬即逝，探索难以捉摸，抉择困难重重"。当然，没有任何一个精确的数字能够回答这个问题，而且对于不同领域(比如，编程，象棋，作曲等)所需要的时间也不同。就像 K. Anders Ericsson 教授所说的，"在绝大多数的领域内，即使是那些极具天赋的人，也需要非常大量的时间才能达到卓越的高度。所谓一万小时只是给你一种感觉，即使是那些非常有天分的人也需要年复一年的保持每周10到20个小时的练习才能达到最高境界"

### 你想成为一名程序员吗

下面是我列举的程序员成功“食谱”

- 对编程产生**兴趣**，因为有趣才去编程。一定要确保你能在编程中找到乐趣，这样你才会有动力去完成10年，每年一万小时的练习
- **写代码**，实践出真知，如果从学术的角度讲:"某个个体在某个领域内能达到的最高水平并不会随着时间的增长而自动增长，而必须要通过可以练习来不断提高"。并且"最有效的学习还需要一系列针对某个个体专门设计的，难度适中的任务以及不断重复的正向反馈机制"，对于这个观点，可以参考[《 Cognition in Practice: Mind, Mathematics, and Culture in Everyday Life》](https://www.amazon.com/exec/obidos/ASIN/0521357349)这本书
- 和其它程序员保持**交流**，阅读其它人的代码。这个比阅读任何书籍或者参加任何培训都要重要。



<p class="md-h-center">(全文完)</p>

### 关于Peter Novig

Peter Norvig任职于Google，其职位是研究主管（Director of  Research). Peter Norvig是享誉世界的计算机科学家和人工智能专家。他是 AAAI 和 ACM 的会员，是业界内经典书籍《Artificial Intelligence: A Modern Approach 人工智能：一种现代方法》的作者之一。在加入Google之前，他曾经是NASA计算科学部门的主要负责人，并在南加州大学以及伯克利大学任教。