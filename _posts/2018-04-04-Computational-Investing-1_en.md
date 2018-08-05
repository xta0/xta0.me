---
layout: post
list_title: 计算投资（二）| Computational Investment Part 1
title: Portfolio Management and Market Mechanics
mathjax: true
---

> Online course from Coursera & Udacity

## Portfolio Management

- Expense ratio
    - Used by mutual funds & ETFs
    - Usually less than 1%
- “Two and twenty”
    - Classic structure for hedge funds.
    - 2% of assets under management + 20% of the returns.
    - 1M with 20% per year = 60k per year
- How/Why different?
    - The goal of a mutual fund or ETF fund manager is to increase the amount of the fund to earn management fees.
    - The goal of private equity funds or hedge fund managers is to increase fund returns

### How to Attract Investors

- Sources of funds
    - Individuals
        - Individual investors, less funds
    - Institusions
        - corporate investor
            - Harvard Foundation
            - CalPERS
        - Funds of Funds
    - Investment from other funds
        - Two Main Types of Fund Goals
            - Reference to a benchmark
                - There are indicators reference, such as running out of S&P500, etc.
        - Absolute return
            - Low risk strategy, focusing only on return on revenue

### Metrics for Assessing Fund Performance

- Common Metrics
    - Annual Return
        - metric:
            - `(value[end] / value[star])) -1`
            - `(value[end] - value[start])/value[start]`
        - Example: $100 to $110
            - `(110/100) -1 = 0.1 = 10%`
            - `(110-100)/100 = 0.1 = 10%`
    - Risk 
        - Definition
            - <mark>Standard deviation of return</mark>
            -  daily return

            ```
            //日回报保准差
            daily_rets[i] = (value[i]/value[i-1]) -1
            std_metrics = stdev(daily_rets)
            ```
        - Draw down
            - 最大跌幅
            - 平均跌幅
- <mark>Reward/Risk</mark>
    - 收益风险比，How much reward you are getting for your risk?
    - **Sharpe Ratio**
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
        - Sortino ratio only penalizes for negative volatility in the calculation of risk.
        - Sharpe ratio penalizes for both positive and negative volatility.
	- Jensen's Alpha

- Example

Suppose there is a mutal fund, which refers to the benchmark of the Dow Jones index as follows:

|      | Return | Sharpe | STDEV | D-down | Corr
|------| -------|--------|-------|--------|------|
| xxFund  | 33%	| .94	| 0.58%	| -8.67%	| 0.89  |
|   $DJI  | 43%	| .63	| 1.23%	| -27.38%	| 1.00  |

- The fund has no yield and has not outperformed the market, but the Sharpe index is higher than the broader market, indicating that its overall volatility is lower and its risk relative to earnings is smaller.
- Similarly, STDEV, D-down data is also low, indicating that the average daily fluctuation is small

### Market Mechanics

- Types of Orders
    - But,Sell
    - Market, Limit
    - Shares
    - Price(if Limit)
    - Additional possibilities:
        - Sell short
    - More complex orders

- The Order Book
    - Ask (Buy)
    - Bid (Sell)

- Mechanics of Short Selling
    1. Borrow the shares 
    2. Sell ​​shares immediately
        - Get cash
        - Need to return the stocks of brokers in the future
    3. If the price falls in the future
        - Just pay a lower price to buy back the stock and return it to the broker
        - The difference of the price is the profit

### Computing Inside a Hedge Fund

![](/assets/images/2018/04/CI-1.jpg)

The picture above is a system trading model.

## Company

- Key Terms
    - [Intrinsic Value](https://www.investopedia.com/terms/i/intrinsicvalue.asp#axzz2Lc5AZO48)
    - [Capital Assests Pricing Model](http://www.investopedia.com/terms/c/capm.asp#axzz2Lc5AZO48)

### Company Worth

- Future Value(Dividens)

Suppose we invest in a company that generates $1 a year in profits, so how much should we buy its shares now? In other words, how does the company's value per share now calculate, or how does the company's stubbornness count? For comparison, let's say we can also deposit the money from the company's stock to the bank, and earn $1 a year. Assuming the bank's interest is 1 cent, then we need to deposit $0.99 into the bank and get a dollar a year later. For the company, it is more risky than the bank, so its income is higher than the bank, assuming we only need to invest 0.95 US dollars a year to get 1 dollar income. this means

> One dollar profit a year later, now only worth $0.95. And the ratio of 0.95/1 is also called the discount rate.

```
|now          |year1      |year2     |year3
Company ----> $1.00 ----> $1.00 ---->$1.00 --->...
$0.95
Bank   -----> $1.00 ----> $1.00 ---->$1.00 --->...
$0.99
```
Note that the discount rate mentioned above is not only for the year of the year, but for an indefinite period of time. The longer the time, the lower the value of the discount of $1. If it is over a few hundred years, the price is $1. It is now basically equal to zero. This also means that a company with a fixed profit of $1 a year will not be infinitely valued over time, as the future $1 is basically zero for the present.

```
--->0.95---->0.95^2----->0.95^3......-->~0.0
    |year1  |year2      |year3       |year n
```
We can calculate the valuation of a company by the discount rate using the formula below.

$$
Value=Sum(dividen*gamma^i) = \sum_{i=1}^\infty dividen * gamma^i = {dividend * 1}/{(1-gamma)}
$$

Where, $dividen$ Represents the company's earnings per share, $gamma$ Indicates the discount rate. In the above example, the company is valued at $20.

$$
Value=\sum_{i=1}^\infty 1.00 * 0.95^i=\frac{1.00}{(1-0.95)}=20.00
$$

- Book Value
    - Net assets
    - "Total" assets minus intangible assets and liabilities.

In summary, the value of the company is
- `Future valuation + net assets`
-  Another more intuitive way to do this is `#(outstanding shares) * price`, ie. Shares times circulation Stock price

### CAPM

CAPM is the abbreviation of Capital Assets Pricing Model. In 1966, Jack Treynor, William Sharpe, John Linter and Jan Mossin jointly proposed that Sharpe, Markowitz and Merton Miller also won the Nobel Prize. The introduction of this theory changed the idea of ​​investment at that time. Based on this model and the efficient market hypothesis, there is a view that investment index funds or allocation of assets to track the index is a better choice, which is called passive investment.

> The investment perspective of this course (quantitative trading) is an active investment concept. Tucker Balch pointed out that this concept can also achieve good reporting, but the CAPM model has a very strong influence and is the basis of various investment management theory. It is necessary to understand carefully

- CAPM hypothesis
    - Return of stock has two components:
        - Systematic (the market)
        - Residual
    - Expected Value of residual = 0
    - Market return
        - Risk free rate of return + Excess return 

<img src="/assets/images/2018/04/CI-2.png" width="70%"/>


The above picture shows the trend of BABA and NASDAQ in the past three months. It can be seen that the trend of BABA and NASDQ is basically the same for most of the time. The volatile part is the BABA's own stock price change. CAPM believes that stock prices are determined by two aspects. On the one hand, market fluctuations, on the other hand, individual stocks fluctuate. The relationship between them is

$$ \gamma_i=\beta_i * \gamma_m + \alpha_i $$

$\gamma_i$ Express the stock price, its value is the market fluctuation coefficient, $\beta_i$ i Take the previous day's price plus the price of the stock the day volatility $\alpha_i$。According to CAPM theory, most of the time, the stock price of a company should fluctuate with the fluctuation of the index, ie $\alpha_i$ The value of expectation is 0, if there is a deviation, the company has some information that can be digged.






