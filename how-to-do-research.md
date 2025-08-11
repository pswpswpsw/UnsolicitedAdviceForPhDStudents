_I am happy that you finally come to this page._

If I may only say one sentence about doing research, it is
> **Research is about standing on the shoulders of giants and then searching for something better.** 

## My standard for papers

1. Your paper should clearly state the hypothesis to test. 
2. Your paper should provides a completely reproducible description of the numerical experiments, which either support or falsify your hypothesis. 
3. The hypothesis must be novel, which include new methods or new applications of methods.
4. You are able to tell the limit of your work, and provide evidence to show that under the assumptions of your work, the results should be expected as it is, and the limit and advantages you mentioned in the work must exist as you said.
5. You should be able to explain about your choice of ``hyperparameters'' in your work. Why this has to be the one you choose, or if not, why the choice won't matter.
6. Your paper should include every other single journal paper that is very related to your work. 
7. Your contribution = the paper minus the existing literature. You must be able to tell your contribution.


## Standing on the shoulders: Literature Review
The first and foremost thing to do is a literature review. It may sound like a daunting task because, you might think
- OMG. Am I going to DO all the works that previous people have done? Then do my own thing?
- OMG. So many papers, where to start? Where to look for? It seems to be an endless list!

The biggest difference between a super undergraduate student and a graduate student, is that the latter can get key knowledge from random, unordered world of information, and form an opinion of himself/herself --- the ability of critical thinking. This is much more beyond the ability of understanding, as a super UG.

### How to climb to the giant's shoulder?
- I will start by sending you at most 3 papers that I think are the best intro-level papers. You need to read them carefully (see #6 ).
- Then, you need examples to reinforce what you have learned. Just like what you have done in the class: you listen to theory and have no idea, but then professor gives you examples, and you suddenly figure out what is it. Here is how you should do: 
   - Go to author's github repo (find that in the paper, search for keyword `github`), download his code, follow his instructions and for first time, run it "blindly". Reproduce his figures. This should be done in 1 hour. 
   - Great! **You just did what they spent 6 months in 1 hour` :)!** Feeling a bit comfortable now? See? You can do this!  
   - Now, then, it is FINALLY the time that you go to their code to figure out what they are writing - print the paper alongside with you, so you can have a reference. WARNING: You will LEARN a LOT during this process: from ideas to how to write that up in Python! (I can come up with ideas for you, but I cannot do the implementation for you! Remember! So your best teacher is not me, is the github repo!)
   - Now, you should have a **good and deep** understanding of what kind of problem are they trying to solve. This is universal for the field so you just save a lot of time figuring out what they are doing.

### Literature review
- Finally, you use **Google scholar** to find the three best 3 papers, and click `cited by`, then you will see who is citing them. Looking for those highly cited and highly relevant from the title, read their abstract, and prepare a list of references beyond those 3 papers that I send you. Now, you should be able to write down a paragraph describing what each group is doing. You can divide them into different topics that they focus. 
- Arrange the results in a document, either latex or word with their figures.

Now you just did a (in-house) literature review. This helps you and me identify what are the potential opportunities. I will also provide my suggestions based on the funding availability and so on. Besides, you should have a clear idea in your mind now: what is missing in the picture, and then it could be YOUR contribution to the field. 
 
## Looking for something better: creativity and innovations

My tricks-for-trade:
- **break their code**: if their code works on simple problems, now you try harder problems, more chaotic, less data, higher dimensional. See if it breaks. If it does, congrats. You find a problem to solve, which we call **pain points**. 
  - Wait, what if the problem is impossible to solve?
- **go realistic!**: another trick to identify opportunities for innovations is to apply those techniques to realistic engineering problems, e.g., design, control, etc. When you apply, you will often encounter unexpected challenges, and things are not going to work. Then it helps you to get lessons and most importantly, others are facing the same issues as well (unless they publish a paper and say they solved it:) then you should figure out if anyone solved it before...). Hence, you just identify those hidden problems when you apply those techniques. Maybe there is a domain-specific solution to that :) 
- **scaling up**: any algorithm will have issues when they scale up. So a no-brainer way to create something new is to scale up existing algorithms, but **do make sure no one else has done that, most likely in the recent 6 months!** 
- **go Bayesian**: any algorithm can be formulated into a Bayesian setting, which means one can make uncertainty taken into account when building models. An obvious benefit of Bayesian is making the model more robust to the noise, which is prevalent in experiments. 

## How to solve the problem we found?
- **do the math / debug the code**: identify the source of the problem, If the result is NaN, what causes NaN? If the result is unstable, what causes unstable? If the result is off, what causes this off?
- come to me if you don't know how to solve the problem you found.

The solution you get should be a hypothesis: 
- "I used tool/theory A to replace tool/theory B used in the original paper, and I believe tool A gives the following benefits"

## Is my hypothesis correct? How to verify?
- Now, you should design "experiments in computer" to verify your hypothesis: your model is better. In order to do that, we need to quantify the model performance over a bunch of representative examples. 
- An easy way to formulate such a verification process, is to "COPY" other experiments in their papers. You will find their paper contains a large number of verification. So you can pick the same problem. Because you have their code on github, you should be able to perform a side-by-side comparison against their results. 

## Before you write a paper, collecting all of the necessary information
If you have come this far, congrats! You almost get a paper done! What needs for a paper is the following three things:
1. **literature review**: what others have done and they miss something and the goal of this paper is to solve this problem
2. **description of your hypothesis**: describe your method/tool/theory and how it is different and better than other's. also, describe how your model is implemented, i.e., the algorithm
3. **verification of your hypothesis**: run a series of experiments, summarize results in a table, compare your model case-by-case with other's model. describe what are the problem setup. 

## Finally, start to write a paper
Please see another post for more details. 