---
layout: post
title: Computational Investment Part1
mathml: true
---

---
title: Computational Investing -1
layout: post

---

## Overview

- Course Objectives
    - Understand electronic markets
    - Understand market data
    - Write software to visualize
    - Write software to discover
    - Create market simulator
- Prerequisites
    - Advanced Programming (Python)
    - Fundamental concepts of investing, financial markets
- Course Logistics
    - 8 weeks, 2 modules per week
    - Projects in Excel and Python
- Course Resources
    - [wiki](http://wiki.quantsoftware.org/index.php?title=QuantSoftware_ToolKit)
    - [Active Portfolio Management](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826/ref=sr_1_1?ie=UTF8&s=books&qid=1263182044&sr=1-1)
    - [All about Hedge Funds](https://www.amazon.com/All-About-Hedge-Funds-Started/dp/0071393935)
    - (Applied Quantitative Methods for Trading and Investment)[https://www.amazon.com/Applied-Quantitative-Methods-Trading-Investment/dp/0470848855/ref=sr_1_1?ie=UTF8&s=books&qid=1263181752&sr=8-1]

    - [Other Resources](- [Resources](https://www.coursera.org/learn/computational-investing/supplement/TPxSD/course-resources))


## Portfolio Management

### Portfolio Manager Incentives
- Expense ratio
    - Used by mutual funds & ETFs
    - Usually less than 1%
- “Two and twenty”
    - Classic structure for hedge funds.
    - 2% of assets under management + 20% of the returns.
    - $1M with 20%/year = $60k/year
- How/Why different?
    - 共同基金或者ETF基金经理目标是增大基金数额来赚取管理费
    - 私募基金或者对冲基金经理目标是提高基金收益

### How to Attract Investors

- 资金来源
    - Individuals
        - 个人投资者，资金占比较少
    - Institusions
        - 机构投资者
            - Harvard Foundation
            - CalPERS
    - Funds of Funds
        - 来自其他基金的投资
- Two Main Types of Fund Goals
    - Reference to a benchmark
        - 有指标参考，比如跑赢S&P500等
    - Absolute return
        - 低风险策略，只关注收益回报

### Metrics for Assessing Fund Performance

- Common Metrics
    - Annual Return
        - metric:
            - (value[end] / value[star])) -1
            - (value[end] - value[start])/value[start]
        - Example: $100 to $110
            - (110/100) -1 = 0.1 = 10%
            - (110-100)/100 = 0.1 = 10%
    - Risk 
        - Standard deviation of return
            -  daily return

            ```
            //日回报保准差
            daily_rets[i] = (value[i]/value[i-1]) -1
            std_metrics = stdev(daily_rets)
            ```
        - Draw down
            - 最大跌幅
            - 平均跌幅
- Reward/Risk
    - How much reward you are getting for your risk?
    - Sharpe Ratio
        - Most "important" measure of asset performance.
        - How well does the return of an asset compensate the investor for the risk taken
        - <mark>The higher the Sharpe ratio the better.</mark>
        - When comparing two assets each with the same return, higher ratio gives more return for the same risk.
        
        <math display = "block">
            <mi>S</mi>
            <mo>=</mo>
            <mfrac>
                <mrow>
                    <mi>E</mi>
                    <mo stretchy="false">[</mo>
                    <mi>R</mi><mo>-</mo><msub><mi>R</mi><mi>f</mi></msub>
                    <mo stretchy="false">]</mo>
                </mrow>
                <mrow>
                    <mi>σ</mi>
                </mrow>
            </mfrac>
            <mo>=</mo>
            <mfrac>
                <mrow>
                    <mi>E</mi>
                    <mo stretchy="false">[</mo>
                    <mi>R</mi><mo>-</mo><msub><mi>R</mi><mi>f</mi></msub>
                    <mo stretchy="false">]</mo>
                </mrow>
                <mrow>
                    <msqrt>
                        <mi>var</mi>
                        <mi>E</mi>
                        <mo stretchy="false">[</mo>
                        <mi>R</mi><mo>-</mo><msub><mi>R</mi><mi>f</mi></msub>
                        <mo stretchy="false">]</mo>
                    </msqrt>
                </mrow>
           	</mfrac>
           	<mo>=</mo>
           	<mi>K</mi><mo>*</mo>
           	<mfrac>
           		<mtext>dailyRet</mtext>
           		<mtext>std(dailyret)</mtext>
           	</mfrac>
        	</math>
		
        - seudo code
				
            ```
            metric = k * mean(daily_rets)/stdev(daily_rets)
            # k = sqrt(250) for daily returns
            # 250: days in a trading year
            ```
    	

	- Sortino Ratio
	- Jensen's Alpha

- Example

假设有一只Fund，它参考Dow Jones指数的benchmark如下：

|      | Return | Sharpe | STDEV | D-down | Corr
|------| -------|--------|-------|--------|------|
| xxFund  | 33%	| .94	| 0.58%	| -8.67%	| 0.89  |
|   $DJI  | 43%	| .63	| 1.23%	| -27.38%	| 1.00  |

1. 该基金没有收益率没有跑赢大盘，但是Sharpe指数高于大盘，说明它的整体波动率较低，相对于收益的风险更小
2. 同样，STDEV，D-down 数据也偏低，说明日均波动较小



## Market Mechanics







## Resource

- 使用Excel计算