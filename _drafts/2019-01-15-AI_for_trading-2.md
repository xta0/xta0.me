---
layout: post
list_title: 计算投资 | Machine Learning For Trading | Basics
title:  Basic Terms
categories: [Machine Learning, Trading]
mathjax: true
---


### Security (证券)

Financial instrument that has some type of monetary value. Some popular securities are stock, bond and options. Security can be classified into three broad types

- Debt Security (债权证券)
    - Debt Security represents the money owed and must be repaid like government or corporate bonds
- Derivative Securities (衍生债券)
    - values depends on other assets
        - Options
            - Option is a contract that gives the buyer the right, but not the obligation to buy or sell underlying assets at a specificed price on a specified date. A good example would be the employee stock options.
        - Future Contracts
            - A future contract obligates the buyer to buy or seller to sell an asset at a predetermined price at a specified time in the future.
- Equity Securities (股权证券)

### Market Mechanics

- Stock Exchange，股票交易所，比如上交所，纽交所
- Broker 券商
- Market Maker 做市商，理论上买卖双方通过券商进行股票交易，但实际上买卖的股票均来自做市商。有些券商也具有做市商的资质，因此我们买的股票实际上是券商提前买好的，再卖给我们。可见，这中间是有套利空间的。
- Liquidity
    - The ability to trade an asset at a stable price. 
    - 市场流通性通常简称流通性或流动性，是指资产能够以一个合理的价格顺利变现的能力
    - Cash is considered the most liquid asset, because it can be readily traded for other assets. This might not be intuitive, but it helps to think about cash as a financial instrument like any other. Changing currency values can make it worth more or less, just like stocks, but usually the value of cash reduces over time due to inflation.

### Stock Basics

- Adjusted stock price after dividend - $P = 1+D/S$, $D$ is the dividend, $S$ is the stock price at ex-dividend date.
- PE (price per earning) ratio - This is the stock’s current market price divided by its most recently reported earnings per share (EPS)
    - What does it mean to have a high PE ratio
        - 高PE值意味着对公司未来盈利能力保持乐观，当然这种乐观可能是盲目的
        - 低PE值意味着对公司目前盈利能力乐观，但对未来的增长持谨慎态度。当然股价的短阶段暴跌也会影响

- Returns 
    - Raw Returns - may be referred to simply as the return, or alternatively, as the percentage return, linear return, or simple return.
        - $r = \frac{p_t-p_{t-1}}{p_{t-1}}$ 
        - An example - Bought at $1000, sold at $1050 one month later, the return is $50/$1000 = 5%
    - Log Returns
        - $R = ln(\frac{p_t}{p_{t-1}})$
        - $R = ln(r+1)$
            - 如果$r<<1$，那么$R \approx r$
            - $r = e^R-1$
    - Continuous Compounding
        - Given rate %r%, the final returns $p_{t} = p_{t-1}{(1+\frac{r}{n})^n}$
            - $r$ is also called <mark>continuously compounded returns </mark>
        - $n$ can be year, month, daily, hour, seconds,etc. 
            - if $n$ goes to infinity, then we have $lim_{n→∞}(1+\frac{r}{n})^n = e^r$
        - In that case, $$p_{t} = p_{t-1}e^r$
            - As you can see, $r = ln(\frac{p_t}{p_{t-1}})$. This is just the log return. So the log return is also called continuously compounded returns.
        - An example
            - 假设有本金100，月复利是2%，则一年后将有$100*(1+2%)^12$的收入
            - 上面式子也可写为$100*(1+24%/12)^12$，则$r=24%$，即年连续复利为$24%$
            - 这个值也近似为$100*e^{0.24}$
        
        
## Momentum Trading

1. Choose a stock universe, fetch daily closing prices
2. Re-sample daily prices, extract month-end prices, compute log returns
3. Rank by month-end returns, select top-n and bottom-n
4. Compute long and short portfolio returns
5. Combine portfolio returns
    - Total = Avg_long - Avg_short


### Statistical Analysis

使用Monthly Portfolio Returns计算均值(比如12个月)，观察其值是否大于0

$$
x^{-} = 1/n\sum_{i=1}^{n}x_i
$$

用t-Statistic