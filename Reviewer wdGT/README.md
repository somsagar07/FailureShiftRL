## Comparative study on finding failure modes with uncertainity estimation method

In our extended study, we delve into the process of identifying failure modes using an uncertainty estimation method, specifically vanilla Bayesian Optimization (BO). We employed an acquisition function that minimizes over a Gaussian prior, using `gp_hedge`. This function probabilistically selects one of the following acquisition functions at each iteration: lower confidence bound, negative expected improvement, or negative probability of improvement.

We observe that BO has a tendency to get trapped in local minima, limiting its exploration. Moreover, BO typically assumes a continuous parameter space, which poses challenges when dealing with disjoint boundaries or discrete action spaces. Without specific modifications, BO struggles in these scenarios. In contrast, reinforcement learning (RL) methods, which often include exploration bonuses, are inherently designed to promote environmental exploration, offering a strategic advantage in such contexts. This assists RL algorithms in avoiding an excessive focus on local optima, enabling them to explore a wider array of possibilities. Such exploration is vital for thoroughly comprehending complex environments and the full spectrum of the effects of actions.

<!-- Image : Figure 1-->
<!-- ![BO's tendency to get trapped in local minima, limiting its exploration capabilities](../images/figure1.png)
*Figure 1: Illustration of Bayesian Optimization's tendency to get trapped in local minima, highlighting its exploration limitations.* -->
<p align="center">
  <img src="../images/figure1.png" alt="BO's tendency to get trapped in local minima, limiting its exploration capabilities">
  <br>
  <em>Figure 1: Illustration of Bayesian Optimization's tendency to get trapped in local minima, highlighting its exploration limitations.</em>
</p>

<!-- Image : Figure 2 -->
<!-- ![Reinforcement learning's robust capability in exploring the parameter space](../images/figure2.png)
*Figure 2: Depiction of reinforcement learning's effective exploration of the parameter space, demonstrating its robustness.* -->

<p align="center">
  <img src="../images/figure2.png" alt="Reinforcement learning's robust capability in exploring the parameter space">
  <br>
  <em>Figure 2: Depiction of reinforcement learning's effective exploration of the parameter space, demonstrating its robustness.</em>
</p>


Figure 1 illustrates the tendency of BO to become stuck in local minima, showcasing its difficulties in exploring the entirety of the parameter space. This limitation underscores the necessity for more adaptable methodologies capable of overcoming such challenges. In contrast, as depicted in Figure 2, RL techniques exhibit a superior capability in navigating and exploring the parameter space, demonstrating their robustness.

## Scalability

We also evaluated the scalability of our approach by increasing the action space. As shown in Figure 3, we observed a non-exponential increase in computational time, suggesting that our method remains scalable even as the action space grows exponentially. 

<!-- Image : Figure 3 -->
<!-- ![Scalability assessment showing non-exponential increase in computational time with expanding action space](../images/figure3.jpg)
*Figure 3: Scalability assessment, demonstrating that computational time increases non-exponentially as the action space expands.* -->

<p align="center">
  <img src="../images/figure3.jpg" alt="Scalability assessment showing non-exponential increase in computational time with expanding action space">
  <br>
  <em>Figure 3: Scalability assessment, demonstrating that computational time increases non-exponentially as the action space expands.</em>
</p>

## Uncertainty estimation in failure mode 
