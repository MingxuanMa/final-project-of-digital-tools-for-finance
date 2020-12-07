# final project of digital tools for finance

Our project is a replication of paper Anderson E W, Cheng A R. Robust bayesian portfolio choices[J]. The Review of Financial Studies, 2016, 29(5): 1330-1375. This paper introduces a model which uses Bayesian updating method to update asset weights with a storage of prior information. We replicate the robust Bayesian model and compare its performance with equally weighted model and standard Markowitz model on S&P 500 daily stock return from 2005 to 2020. Our main finding is that, with a small portfolio (N=30) and large rolling window (T=100), the robust bayesian model outperforms the other two before 2012, and this outperforming effect vanishes after 2012 when the paper is submitted. Also, we finds an interesting phenonmenon: robust bayesian model dominates the other two regardless of portfolio size from 2008 September to 2009 March, which is the period of financial crisis. A possible explanation could be that during crisis period, investors put more weight on prior information. Since investors are more likely to sell a stock which has a negative return yesterday due to panic rather than rational analysis during crisis period, prior information plays a more important role. 

Our project is divided into the following sections: 

Code: codefile has the code which replicates robust bayesian model and compares its performance with the other two benchmark models. 

Data: 1. contains the risk-free rate from 2005 to 2020 since we uses a time-varing risk-free rate. We regards 1-year T bill rate as risk free and convert it to daily rate. 2. contains S&P 500 daily stock return from 2005 to 2020.

Latex: 1. the folder "paper" contains latex file which builds the paper. 2. the folder "presentation" contains the beamer file which builds the presentation file. 

Jupyter: contains the jupyter notebook which shows the output of our code, including multiple periods optimization results and figures. 
