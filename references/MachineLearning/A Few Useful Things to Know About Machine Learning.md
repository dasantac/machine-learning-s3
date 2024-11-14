- A classifier is a system that inputs (typically) a vector of discrete and/or continuous feature values and outputs a single discrete value, the class.
# Learning = Representation + Evaluation + Optimization
- Which algorithm to use?
	- The key to not getting lost in this huge space is to realize that it consists of combinations of just three components:
		- Representation
		- Evaluation
		- Optimization
	- Examples for each of these three components:

| Representation               | Evaluation            | Optimization               |
| ---------------------------- | --------------------- | -------------------------- |
| Instances                    | Accuracy/Error rate   | Combinatorial optimization |
| >> K-nearest neighbor        | Precision and recall  | >> Greedy search           |
| >> Support Vector Machines   | Squared error         | >> Beam search             |
| Hyperplanes                  | Likelihood            | >> Branch-and-bound        |
| >> Naive Bayes               | Posterior probability | Continuous optimization    |
| >> Logistic regression       | Information gain      | >> Unconstrained           |
| Decision trees               | K-L Divergence        | >>>> Gradient Descent      |
| Sets of rules                | Cost/Utility          | >>>> Conjugate gradient    |
| >> Propositional rules       | Margin                | >>>> Quasi-Newton methods  |
| >> Logic programs            |                       | >> Constrained             |
| Neural networks              |                       | >>>> Linear programming    |
| Graphical models             |                       | >>>> Quadratic programming |
| >> Bayesian networks         |                       |                            |
| >> Conditional random fields |                       |                            |
- Of course, not all combinations of one component from each column of the table make equal sense.
	- For example, discrete representations naturally go with combinatorial optimization, and continuous ones with continuous optimization.
- Nevertheless, many learners have both discrete and continuous components, and in fact the day may not be far when every single possible combination has appeared in some learner!
## Representation
- A classifier must be represented in some formal language that the computer can handle.
- Conversely, choosing a representation for a learner is tantamount to choosing the set of classifiers that it can possibly learn. This set is called the ==hypothesis space== of the learner.

## Evaluation
- An evaluation function (also called objective function or scoring function) is needed to distinguish good classifiers from bad ones.
- The evaluation function used internally by the algorithm may dif- fer from the external one that we want the classifier to optimize

## Optimization
- Finally, we need a method to search among the classifiers in the language for the high- est-scoring one.
- The choice of op- timization technique is key to the efficiency of the learner, and also helps determine the classifier pro- duced if the evaluation function has more than one optimum.

# It's Generalization that Counts
- ==The fundamental goal of machine learning is to generalize beyond the examples in the training set.==
- set some of the data aside from the beginning, and only use it to test your chosen classifier at the very end, followed by learning your final classifier on the whole data.
	- Of course, holding out data reduces the amount available for training.
		- This can be mitigated by doing cross-validation: randomly dividing your training data into (say) 10 subsets, holding out each one while training on the rest, testing each learned classifier on the examples it did not see, and averaging the results to see how well the particular parameter setting does.
- Notice that generalization being the goal has an interesting consequence for machine learning. ==Unlike in most other optimization problems, we do not have access to the function we want to optimize!==
	- We have to use training error as a surrogate for test error, and this is fraught with danger.
	- "On the positive side, since the objective function is only a proxy for the true goal, we may not need to fully optimize it; in fact, a local optimum returned by simple greedy search may be better than the global optimum."

# Data Alone is Not Enough
- Generalization being the goal has another major consequence: Data alone is not enough, no matter how much of it you have.
	- Consider learning a Boolean function of (say) 100 variables from a million examples. There are `2100 − 106` examples whose classes you do not know. How do you figure out what those classes are?
		- ==In the absence of further information, there is just no way to do this that beats flipping a coin.==
- ==Every learner must embody some knowledge or as- sumptions beyond the data it is given in order to generalize beyond it.==
	- This notion was formalized by Wolpert in his famous “no free lunch” theorems, according to which no learner can beat random guessing over all possible functions to be learned.
- This seems like rather depressing news. How then can we ever hope to learn anything?
	- ==Luckily, the functions we want to learn in the real world are not drawn uniformly from the set of all mathematically possible functions!==
		- In fact, very general assumptions—like smoothness, similar examples having similar classes, limited dependences, or limited complexity—are often enough to do very well, and this is a large part of why machine learn- ing has been so successful.
- Like deduction, ==induction (what learners do) is a knowledge lever==: ==it turns a small amount of input knowledge into a large amount of output knowledge.==
	- Induction is a vastly more powerful lever than deduction, requiring much less input knowledge to produce useful results, but it still needs more than zero input knowledge to work. ==And, as with any lever, the more we put in, the more we can get out.==
- A corollary of this is that ==one of the key criteria for choosing a representation is which kinds of knowledge are easily expressed in it.==
	- if we have a lot of knowledge about what makes examples similar in our domain, instance-based methods may be a good choice.
	- If we have knowledge about probabilistic dependencies, graphical models are a good fit.
	- And if we have knowledge about what kinds of preconditions are required by each class,“IF...THEN...” rules may be the best option.

- In retrospect, the need for knowledge in learning should not be surprising. ==Machine learning is not magic; it cannot get something from nothing. What it does is get more from less.==
	- ==Programming, like all engineering, is a lot of work: we have to build everything from scratch. ==
	- ==Learning is more like farming, which lets nature do most of the work. ==
	- ==Farmers combine seeds with nutrients to grow crops. Learners combine knowledge with data to grow programs.==

# Overfitting has Many Faces
- What if the knowledge and data we have are not sufficient to completely determine the correct classifier? 
	- Then we run the risk of just hallucinating a classifier (or parts of it) that is not grounded in reality, and is simply encoding random quirks in the data.
	- This problem is called overfitting, and is the bugbear of machine learning.
- ==One way to understand overfitting is by decomposing generalization error into bias and variance.==
	- ==Bias is a learner’s tendency to consistently learn the same wrong thing.==
	- ==Variance is the tendency to learn random things irrespective of the real signal.==

	- Choosing Reperesentation:
		- A linear learner has high bias, because when the frontier between two classes is not a hyperplane the learner is unable to induce it.
		- Decision trees do not have this problem because they can represent any Boolean function, but on the other hand they can suffer from high variance: decision trees learned on different training sets generated by the same phenomenon are often very different, when in fact they should be the same.
	- Choosing optimization methods:
		- beam search has lower bias than greedy search, but higher variance, because it tries more hypotheses. 
		- Thus, contrary to intuition, a more powerful learner is not necessarily better than a less powerful one.
		- Example: Naïve Bayes can outperform a state-of-the-art rule learner (C4.5rules) even when the true classifier is a set of rules.
			- Even though the true classifier is a set of rules, with up to 1,000 examples Naïve-Bayes is more accurate than a rule learner.
				- This happens despite naive Bayes’s false assumption that the frontier is linear!
				- Situations like this are common in machine learning: ==strong false assumptions can be better than weak true ones, because a learner with the latter needs more data to avoid overfitting==.

- Combatting overfitting
	- ==Cross-validation== 
		- for example by using it to choose the best size of decision tree to learn. 
		- But it is no panacea, since if we use it to make too many parameter choices it can itself start to overfit.
	- adding a ==regularization term== to the evaluation function
		- This can, for example, penalize classifiers with more structure, thereby favoring smaller ones with less room to overfit.
	- ==perform== a ==statistical significance== test like chi-square ==before adding new structure==, to decide whether the distribution of the class really is differ- ent with and without this structure

	- Nevertheless, you should be skeptical of claims that a particular technique “solves” the overfitting problem.
		- ==It is easy to avoid overfitting (variance) by falling into the opposite error of underfitting (bias)==
		- Simultaneously avoiding both requires learning a perfect classifier, and short of knowing it in advance there is no single technique that will always do best (no free lunch).

- A common misconception about overfitting is that it is caused by noise
	- This can indeed aggravate overfitting,
	- But severe overfitting can occur even in the absence of noise
	- Example
		- suppose we learn a Boolean classifier that is just the disjunction of the examples labeled “true” in the training set.
		- This classifier gets all the training examples right and every positive test example wrong, regardless of whether the training data is noisy or not.

- The problem of multiple testing is closely related to overfitting.
	- Standard statistical tests assume that only one hypothesis is being tested, but modern learners can easily test millions before they are done.
	- As a result what looks significant may in fact not be.
	- Example:
		- a mutual fund that beats the market 10 years in a row looks very impressive, until you realize that, if there are 1,000 funds and each has a 50% chance of beating the market on any given year, it is quite likely that one will succeed all 10 times just by luck.
	- solutions:
		- This problem can be combatted by correcting the significance tests to take the number of hypotheses into account, but this can also lead to underfitting.
		- A better approach is to control the fraction of falsely accepted non-null hypotheses, known as the false discovery rate.

# Intuition Fails in High Dimensions
- After overfitting, the biggest problem in machine learning is the ==curse of dimensionality==.
	- This expression was coined by Bellman in 1961 to refer to the fact that many algorithms that work fine in low dimensions become intractable when the input is high-dimensional.
	- But in machine learning it refers to much more:
		- ==Generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows== because a fixed-size training set covers a dwindling fraction of the input space.
			- example:
				- Even with a moderate dimension of 100 and a huge training set of a trillion examples, the latter covers only a fraction of about $10^{−18}$ of the input space.
			- explanation behind this:
				- the similarity-based reasoning that machine learning algorithms depend on (explicitly or implicitly) breaks down in high dimensions:
					- Consider a nearest neighbor classifier with Hamming distance as the similarity measure, and suppose the class is just $x_1 ∧ x_2$.
						- If there are no other features, this is an easy problem. 
						- But if there are 98 irrelevant features $x_3,..., x_{100}$, the noise from them completely swamps the signal in $x_1$ and $x_2$, and nearest neighbor effectively makes random predictions.
						- Even more disturbing is that nearest neighbor still has a problem even if all 100 features are relevant! ==This is because in high dimensions all examples look alike.==
							- Suppose, for instance, that examples are laid out on a regular grid, and consider a test example $x_t$.
							- If the grid is d-dimensional, $x_t$’s 2d nearest examples are all at the same distance from it.
							- ==So as the dimensionality increases, more and more examples become nearest neighbors of xt, until the choice of nearest neighbor (and therefore of class) is effectively random.==
		- This is what makes machine learning both necessary and hard.
	- This is only one instance of a more general problem with high dimensions: our intuitions, which come from a three-dimensional world, often do not apply in high-dimensional ones.
		- In high dimensions, most of the mass of a multivariate Gaussian distribution is not near the mean, but in an increasingly distant “shell” around it
		- most of the volume of a high-dimensional orange is in the skin, not the pulp
		- If a constant number of examples is distributed uniformly in a high-dimensional hypercube, beyond some dimensionality most examples are closer to a face of the hypercube than to their nearest neighbor.
		- And if we approximate a hypersphere by inscribing it in a hypercube, in high dimensions almost all the volume of the hypercube is outside the hypersphere.
			- This is bad news for machine learning, where shapes of one type are often approximated by shapes of another.