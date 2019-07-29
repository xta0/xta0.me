---
layout: post
updated: "2018-08-18"
list_title: 十年学会编程 | Teach Yourself Programming in Ten Years 
title:  Teach Yourself Programming in Ten Years
categories: [Life,Translation]
---

> Chinese translation of Peter Norvigs's essay [Teach Yourself Programming in 10 Years](http://norvig.com/21-days.html).


### 关于Peter Novig

> Peter Norvig任职于Google，其职位是研究主管（Director of  Research). Peter Norvig是享誉世界的计算机科学家和人工智能专家。他是 AAAI 和 ACM 的会员，是业界内经典书籍《Artificial Intelligence: A Modern Approach 人工智能：一种现代方法》的作者之一。在加入Google之前，他曾经是NASA计算科学部门的主要负责人，并在南加州大学以及伯克利大学任教。



### 为什么大家都急于求成

每当走进各大书店总能看到一些诸如《24小时自学Java开发》之类的书籍，类似的还有几小时内学会C,Ruby,SQL,Algorthms等等。如果使用亚马逊的高级索索功能[title: teach, yourself, hours, since: 2000](http://www.amazon.com/gp/search/ref=sr_adv_b/?search-alias=stripbooks&unfiltered=1&field-keywords=&field-author=&field-title=teach+yourself+hours&field-isbn=&field-publisher=&node=&field-p_n_condition-type=&field-feature_browse-bin=&field-subject=&field-language=&field-dateop=After&field-datemod=&field-dateyear=2000&sort=relevanceexprank&Adv-Srch-Books-Submit.x=16&Adv-Srch-Books-Submit.y=5) 可以找到512种这类书籍，在搜索结果的前十名中有九本是编程类书籍（其它的是关于记账类书籍）。如果把搜索词"yourself"换成"learn"，或者把"days"换成"hours"，搜索结果也是类似的。

从上面的案例中我们能出的结论是，要么大家都非常着急的想学编程，要么学习编程和其它事情相比是一件很容易的完成的事。Felleisen et al.在它的[《How to Design Programs》](http://felleisen.org/matthias/)一书中也提到了这个问题，当然他也是不认同这种思路的，他们的原话是："编写糟糕的程序很简单，傻瓜也可以在21天内学会，哪怕他们傻到家"。风靡网络的漫画公司[Abtruse Goose](https://abstrusegoose.com/249)也对此冷嘲热讽了一把，漫画如下

<img class="md-img-center" src="/assets/images/2018/03/ars_longa_vita_brevis.png">

让我们来一起分析一下《24小时学会C++》意味着什么

- **自学**：在24小时内你甚至没有时间写几段有难度的程序，更不用说从中学习什么了。你也没有时间和有经验的C++程序员工作，不会了解真正的C++项目是怎样的。简而言之就是你没有时间学到很多东西，因此书中只能浅显的让你熟悉下语法，没法做更深入的讲解。就像 Alexander Pope 曾经说过的，浅尝辄止的学习是一件非常危险的事情。

- **C++**: 在24小时内只够你学习一些基本的C++语法（假设你已经有过其它编程语言的经验），但是无法学到该如何使用这门语言。简单的说，如果你曾经是一名BASIC程序员，你可能会学会如何将原来的BASIC代码按照C++的语法重写一遍，但却不清楚为什么要用C++，也不清楚这门语言的优点（或者缺点）在哪里。[Alan Perlis](http://pu.inf.uni-tuebingen.de/users/klaeren/epigrams.html)曾经说过，"如果一种语言不能改变你对编程的理解，那么这门语言就不值得学习"。现实生活中，一种可能的情况是，为了完成某种任务你需要快速学一点C++来对接某个三方库，但是这种并不是真正的学习某种语言，只是为了完成某种任务。

- **24小时内**: 不幸的是，24小时时间实在是太短了，下一小节会讨论这个问题

### 给自己十年的时间来学习编程

很多学者已经证明了如果某个人想要成为某个领域的专家，需要差不多10年的时间，这些领域包括象棋，作曲，电报通信，绘画，弹钢琴，游泳，网球 等等。在这10年中，最重要的是保持<em>刻意练习</em>: 不仅仅是不断重复，还要不断挑战自己去完成超出目前能力的任务，然后总结自己的表现，改正错误后继续重复。这件事貌似没有捷径，即使像莫扎特这种4岁就成为音乐奇才的人也花了13年才创作出世界级的音乐。另一个例子是Beatles，貌似他们在1964年就凭借一系列热门单曲以及在The Ed Sullivan show上的演出而一炮而红，但是你也许不知道，他们早在1957年就在利物浦和汉堡两地进行小规模演出了，而在此之前的非正式演出更是不计其数，即便是在他们走红之后，他们的第一部最具影响力的唱片《Sgt. Peppers》也是在1967年才发布。

Malcolm Gladwell 曾经提出过一个很流行的理论，即一万小时的刻意练习。Henri Cartier-Bresson (1908-2004) 曾经说过，"你的前一万张照片是最糟糕的"（他并没有预料到数码相机的出现，人们能在1周内完成一万张的拍摄）。真正的专家们可能会在某个领域投入自己一生的时间，Samuel Johnson (1709-1784) 曾经说，"想要在某个领域做到卓越的唯一办法就是终身投入"。Hippocrates (c. 400BC) 有一段著名的引用"ars longa, vita brevis"，这句话出自"Ars longa, vita brevis, occasio praeceps, experimentum periculosum, iudicium difficile"，翻译过来是说，"生命很短暂，但是技艺却很高深，机遇转瞬即逝，探索难以捉摸，抉择困难重重"。当然，没有任何一个精确的数字能够回答这个问题，而且对于不同领域(比如，编程，象棋，作曲等)所需要的时间也不同。就像 K. Anders Ericsson 教授所说的，"在绝大多数的领域内，即使是那些极具天赋的人，也需要非常大量的时间才能达到卓越的高度。所谓一万小时只是给你一种感觉，即使是那些非常有天分的人也需要年复一年的保持每周10到20个小时的练习才能达到最高境界"

### 你想成为一名程序员吗

下面是我列举的程序员成功的必备要素

- 对编程产生**兴趣**，因为有趣才去编程。一定要确保你能在编程中找到乐趣，这样你才会有动力去完成10年，每年一万小时的练习
- **写代码**，实践出真知，如果从学术的角度讲:"某个个体在某个领域内能达到的最高水平并不会随着时间的增长而自动增长，而必须要通过可以练习来不断提高"。并且"最有效的学习还需要一系列针对某个个体专门设计的，难度适中的任务以及不断重复的正向反馈机制"，对于这个观点，可以参考[《 Cognition in Practice: Mind, Mathematics, and Culture in Everyday Life》](https://www.amazon.com/exec/obidos/ASIN/0521357349)这本书
- 和其它程序员保持**交流**，阅读其它人的代码。这个比阅读任何书籍或者参加任何培训都要重要。
- 如果可能的话，在大学里读四年书，一方面文凭对就业找工作有帮助，另一方面你能对这个领域都更深层次的理解。如果你不喜欢去学校，也可以自学或者在工作中积累经验。不管怎么说，仅靠书本上的知识是远不够的。Eric Raymond曾说："仅接受计算机科学的教育是不可能让人成为编程专家的，就像学习画笔和颜料不可能让你成为画家一样"。他是我雇佣过的最优秀的程序员，他开发很多[优秀的软件](http://www.xemacs.org/)，创立了属于自己的社区，并且通过股票期权赚足了钱买下了一家[自己的夜店](https://en.wikipedia.org/wiki/DNA_Lounge)
- 和其他的程序员一起合作项目积累经验。在有些项目中你可能是最好的程序员，在另一些项目中你可能是最差的，当你是项目中最好的程序员时，尝试培养自己的领导力，用你的视野鼓励大家。当是最差的时候，向项目中的高手学习，学习他们不做什么（因为他们让你帮他们做）。
- 维护已有的项目。了解其他人的编程思路，尝试在作者不在的时候修复bug。换位思考，想象如果你是作者，怎么可以让别人轻松的维护你的代码
- 至少学习一半的编程语言。包括一门强调类型抽象的语言（比如Java或者C++），一门强调函数抽象的语言（比如Lisp或者ML或者Haskell）,一门声明式的语言（比如Prolog，C++模板）还有一门强调并行计算的语言（比如Clojure或者Go）
- 记住"计算机科学"中包含着"计算机"这个词，要了解你的计算机需要花费多长时间执行一条指令，从内存取一个WORD（考虑命中或者没命中缓存的情况）的时间，以及从磁盘上连续读取多个WORD时定位文件指针的时间（答案见附录）。
- 参加语言的标准化制定工作，这些工作种类多样，可以是加入标准C++委员会，也可以是制定编码风格（每行缩进是2个还是4个空格）等等。不论哪种工作，你能通过反馈收集到其他人是否喜欢这门语言，如果喜欢，他们喜欢哪些东西，倾听他们的感受，了解他们的想法
- 有良好的意识，能尽快适应语言标准化的成果。

考虑到上面所说的一切，你会发现只靠看书学习是很难成功的。当我的第一个孩子出生的时候，我几乎阅读了市面上所有的《如何…》指南书籍，但是我读完了以后还是觉得自己是个没有头绪的新手。30个月以后，我的第二个孩子快出生时，我还要回去读书将所有的知识复习一遍吗？不，相反，我此时更依赖我的个人经验，这些经验相比于那些专家写的上千页的书更加有效和让我放心。

Fred Brooks 在他的散文[No Silver Bullet](http://worrydream.com/refs/Brooks-NoSilverBullet.pdf)中给出了一个寻找软件设计师的三步走计划：

1. 尽早开始系统性的寻找找顶尖的软件设计师
2. 给有潜力的工程师指派一名职业规划导师，并仔细规划并记录他们的职业档案
3. 为处在职业上升期中的软件设计师提供和互动交流的机会

上面的讨论是假设某个人已经具备足够的能力成为一名软件设计师，那么他只需要一份工作去督促他。Alan Perlis更简洁地指出，“每个人都可以被教会如何雕刻，但是对于米开朗基罗来说，则不应该被'教会'，伟大程序员也是如此”。Perli认为所有伟大的人其内心都有一种内在的特质(自我驱动的能力)，这种特质往往可以超越训练，达到更高的高度。但是这些特质是从哪里来的呢？是天生的吗？还是他们通过后天的勤奋习得的？就像Auguste Gusteau（动画片《料理鼠王》里的大厨）所说的，“每个人都能成为厨师，但只有那些内心“无所畏惧”的人才能成为伟大的厨师。" 我认为伟大更多的是来自与愿意将自己全身心的投入到某项事业中，并保不断持刻意练习。也许“无所畏惧”可以概括这种精神。或者像 Gusteau's critic, Anton Ego所说的，“不是每个人都能成为伟大的艺术家，但是伟大的艺术家可以来自任何地方。”

所以尽管去买 Java/Ruby/Javascript/PHP 这类的书吧；你可能会从中学到点儿东西。但作为一个程序员，你并不会在21天内或24小时内改变自己的人生，或你的综合水平。你是否想过努力不间断的学习超过24个月？如果是的话，那么恭喜你，你已经在路上了...

<hr>

### 参考文献

- Bloom, Benjamin (ed.) [Developing Talent in Young People](http://www.amazon.com/exec/obidos/ASIN/034531509X), Ballantine, 1985.
- Brooks, Fred, [No Silver Bullets](http://citeseer.nj.nec.com/context/7718/0), IEEE Computer, vol. 20, no. 4, 1987, p. 10-19.
- Bryan, W.L. & Harter, N. "Studies on the telegraphic language: The acquisition of a hierarchy of habits. Psychology Review, 1899, 8, 345-375
- Hayes, John R., [Complete Problem Solver](http://books.google.com/books?id=dYPSHAAACAAJ&dq=%22perception+in+chess%22+simon&ei=z4PyR5iIAZnmtQPbyLyuDQ) Lawrence Erlbaum, 1989.
- Chase, William G. & Simon, Herbert A. "Perception in Chess" Cognitive Psychology, 1973, 4, 55-81.
- Lave, Jean, [Cognition in Practice: Mind, Mathematics, and Culture in Everyday Life](https://www.amazon.com/exec/obidos/ASIN/0521357349), Cambridge University Press, 1988.

<hr>

### 答案

PC上一些常用操作的大概耗时

|execute typical instruction |	1/1,000,000,000 sec = 1 nanosec|
|fetch from L1 cache memory	| 0.5 nanosec|
|branch misprediction |	5 nanosec|
|fetch from L2 cache memory	 | 7 nanosec|
|Mutex lock/unlock | 25 nanosec|
|fetch from main memory | 100 nanosec|
|send 2K bytes over 1Gbps network |	20,000 nanosec|
|read 1MB sequentially from memory|	250,000 nanosec|
|fetch from new disk location (seek)|	8,000,000 nanosec|
|read 1MB sequentially from disk|	20,000,000 nanosec|
|send packet US to Europe and back|	150 milliseconds = 150,000,000 nanosec|

<hr>

### 附录：语言的选择

很多人询问如果要入门编程，应选择哪一门语言作为第一门编程语言。这个问题没有统一的答案，下面三点可供参考

- 看看你的朋友们用什么。当被问起“我该用哪种操作系统，Windows，Unix， 还是 Mac？”，我总是回答：“你朋友用什么，你就用什么。“ 你从朋友那学习得到的好处可以抵销不同操作系统或语言之间本质的差异。同样，你也要考虑你将来的朋友们：程序员社区，因为在未来如果你继续前行，你将会是他们中的一部分。你选择的语言是否有一个快速增长的社区？ 有没有书籍、网站或者论坛能解答你的问题？你喜欢论坛里的那些人吗？

- 保持简单。像 C++ 和 Java 这样的语言是为具有丰富编程经验的开发团队设计的，他们更关注代码的执行效率。因此，这些语言为了优化性能，有些部分设计的非常复杂。 而你关注的是如何学会编程，不需要那些复杂的设计。你需要的是一些设计简单的，易于上手的语言。

- 及时反馈。你偏爱哪种学弹钢琴的方式：是简单的互动的方式，你一按下琴键就能听到音符；还是完整的批量的模式，你只有弹完整首曲子才能听到音符？ 显然，用基于互动的学习更容易些，对编程也一样。坚持用支持交互模式的语言进行学习，这样可以得到快速反馈。

参考上面几点，我推荐的第一个编程语言是[Python](https://www.python.org/) 或[Scheme](https://schemers.org/)。另一个选择是 Javascript，并不是因为它的设计对初学者很友好，而是因为有大量的在线教程，比如[Khan Academy’s tutorial](https://www.khanacademy.org/computing/cs/programming)。但是这些选择视个人情况而定，除了这几门语言之外也还有更好的选择。如果你的年纪是10岁以下，你可以尝试[Alice](http://www.alice.org/) 或 [Squeak](https://squeak.org/) 或 [Blockly](https://blockly-demo.appspot.com/static/demos/index.html) （大人们也可能会喜欢）关键是你下定决心后要快速行动。

<hr>

### 附录：编程书籍和其它资源

很多人问我该看那些编程类的书籍或者学习哪些编程类的网站。我的回答是“书本上的知识是远远不够的”，但是我可以推荐下面一些书籍

- **Scheme**: [Structure and Interpretation of Computer Programs (Abelson & Sussman) ](https://www.amazon.com/gp/product/0262011530)（译者注：中文名称为《计算机的构造与解释》），这本书可能是对于计算机科学入门来说最好的一本书了，它从理解计算机工作原理的角度来教你编程。你可在这里找到[在线视频](http://groups.csail.mit.edu/mac/classes/6.001/abelson-sussman-lectures/)和配套的[文本教材](http://mitpress.mit.edu/sicp/full-text/book/book.html)。但是这本书有一定的挑战性，会淘汰一些之前学习过其它语言的老手。
- **Scheme**: [ How to Design Programs (Felleisen et al.)](https://www.amazon.com/gp/product/0262062186)，这本书可能是在所有介绍使用函数型语言设计程序的书中最好的一本
- **Python**: [Python Programming: An Intro to CS (Zelle)](http://www.amazon.com/gp/product/1887902996)这本书是比较不错的一本入门Python的书籍
- **Python**: [Python.org](Python.org.)上提供一系列[在线的教程](http://wiki.python.org/moin/BeginnersGuide)
- **Oz**: [Concepts, Techniques, and Models of Computer Programming (Van Roy & Haridi)](http://www.amazon.com/gp/product/0262220695)这本书被某些人认为是对Abelson＆Sussman观点的传承。它对重要的编程理念进行了回顾，覆盖的内容比Abelson＆Sussman更广泛，也很容易读懂。它使用Oz语言，一门不常用但是适合学习编程的语言。

<hr>

### 备注

T. Capey指出，在Amazon的[Complete Problem Solver](https://www.amazon.com/exec/obidos/ASIN/0805803092)的商品页面中，查看浏览“购买此书的用户还购买过这些产品”区域，会发现"Teach Yourself Bengali in 21 days"和"Teach Yourself Grammar and Style"这两本书。我估计浏览这两本书的大部分人是从这篇文章的链接过去的。感谢Ross Cohen的帮助。


{% include _partials/post-footer-1.html %}


