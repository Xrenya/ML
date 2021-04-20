## Deep Learning

concepts
- forward and backward propagation
- vanishing gradient
- image convolution operation
- feature map, filter/kernel
- receptive field
- embedding
- translation invariance

ideas
- transfer learning
- augmentation
- semantic segmentation, instance segmentation, panoptic segmentation
- object detection, localization

details
- bias trick
- weight initialization: xavier VS he (kaiming)
- early stopping, learning rate annealing / decay
- learning rate range test, cyclical learning rate, cosine learning rate
- learning rate vs batch size
- L1 vs L2, dropout regularization
- ensembling models trained on cross-validation folds
- online augmentation, test time augmentation
- encoder/decoder, autoencoder
- non maximum suppression (detection)

optimization
- SGD
- momentum, nesterov momentum
- adagrad, rmsprop
- adam, r-adam, n-adam

loss functions and metrics
- log loss, nll loss
- smooth L1 loss
- triplet loss
- softmax, softplus, softshrink, softsign
- cross entropy loss, binary cross entropy loss, balanced cross entropy loss
- focal loss
- huber loss
- hinge loss / multiclass SVM loss / max-margin loss
- lovasz-softmax loss
- dice coefficient, jaccard coefficient (IoU), soft dice, soft jaccard
- average precision (AP), average recall (AR), AP11, mAP@[.5, .95], localization-recall-precision (LRP)

layers
- fully connected (dense) layer
- rely, leaky relu, exponential linear unit, prelu
- dropout, maxout
- tanh, sigmoid
- (spatial, temporal) batch normalization layer
- convolutional layer
- maxpool layer, avgpool, global pooling
- unpooling layer
- deconvolutional layer
- shortcut connections
- dilated convs, depthwise separable convs, bottleneck blocks
- group convolution

tricks
- squeeze-and-excitation
- warm restarts
- learning rate warmup

classification
- LeNet
- AlexNet
- ZFNet
- VGG
- GoogLeNet
- ResNet, ResNeXt, ResNet SE, ResNet-D, WRN
- Inception, Inception-ResNet
- Xception
- MobileNet
- ShuffleNet

semantic segmentation
- FCN8
- SegNet
- UNet
- PSPNet
- FPN
- DeepLab, DeepLab v3
- ENet
- ICNet

object detection
- R-CNN
- Fast RCNN
- Faster RCNN
- SSD
- YOLO, YOLO9000
- RetinaNet
- CenterNet

instance segmentation
- Mask R-CNN
- YOLACT

pose estimation
- PoseNet
- DensePose

gan
- GAN
- DCGAN
- WGAN
- pix2pix
- CycleGAN

other
- SqueezeNet
- DetectNet

distributed training
- model parallel
- data parallel
- microbatches combination
- on-demand parameters loading
- ZeRO optimizer state, parameters and gradient partitioning


## Machine learning

concepts
- max likelihood, entropy, cross-entropy, conditional entropy
- mutual information
- gini impurity, information gain, KL-divergence, variance reduction
- stacking, bagging, boosting
- hyperparameters vs model parameters
- kolmogorov complexity

optimization
- gradient descent, SGD
- newton method, newton-raphson method, L-BFGS, quasi-netwon methids
- non-linear conjugate gradient method
- jacobian, hessian
- quadratic optimization
- line search, backtracking line search, trust region

linear algebra
- linear independence, rank of a matrix, determinant
- eigen decomposition, eigenvectors, eigenvalues
- SVD, truncated SVD
- matrix diagonalization
- positive definite matrix

regularization
- L1, L2 regularization
- trees regularization

transformation
- Box-Cox transform
- unbounded Johnson transform
- log/sqrt transform
- normalization (z-scoring)
- frequency encoding, target encoding, one-hot encoding

regression
- linear/non-linear/segmented/local regression, ordinary least squares
- lasso/ridge regression, elastic net
- gradient boosting, random forest
- coefficient of determination (R2), adjusted R2, fraction of variance unexplained
- residuals analysis: residuals vs predictors, residuals histogram, studentized residuals
- heteroskedasticity 
- k nearest neighboors

classification
- logistic regression, logit, odds, log-odds
- SVM, kernel trick
- contingency table, confusion matrix
- precision, recall, specificity/selectivity, accuracy, F1, informedness
- ROC AUC, precision-recall curve
- multiclass classification - separating hyperplanes
- one vs all, all vs all - multiclass classifications
- type I and II errors
- xgboost vs catboost vs lightgbm

clustering
- k-means

dimensionality reduction
- PCA, Robust PCA
- t-SNE
- feature selection

cross-validation
- exhaustive cross-validation, non-exhaustive cross-validation, nested cross-validation
- leave-p-out validation
- k-fold validation, stratified k-fold validation
- holdout validation method
- repeated random subsampling validation
- k*l-fold validation, k-fold validation with validation and test set

kernel
- window function
- kernel smoother
- kernel density estimation / parzen window
- multivariate kernel density estimation
- uniform, epanechnikov, triangle, tricube, quartic, gaussian, quadratic, cosine kernels

algorithms
- kd-tree, ball-tree


## Probability & statistics

concepts
- pdf, cdf, pmf
- central moment vs raw moment
- mean, median, mode, variance, covariance, skewness, kurtosis, moments
- quantiles, quartiles, interquartile range (IQR), box plot
- hypothesis testing
- sample mean and variance
- confidence intervals, prediction intervals
- correlation, pearson's correlation, spearman's rank correlation, kendall correlation
- bias-variance tradeoff
- likelihood function, maximum likelihood estimation (MLE), maximum a posteriori estimation (MAP)
- memoryless distribution
- kullback-leibler divergence, entropy of a distribution
- cross-entropy
- convolution of probability distributions, distribution of the sum of random values
- mixture distribution
- probability transformation rule
- kernel trick
- hilbert space

distributions
- bernoulli, binomial, multinomial, geometric, hypergeometric distributions
- uniform, normal, laplace, exponential, poisson, chi-square, gamma, beta, student's t distributions

statistical tests
- null hypothesis, p-value, one-tailed and two-tailed tests, statistical significance
- type I and type II errors
- z-score, t-statistic, z-test, t-tests
- fisher's method / combined probability test
- f-value, f-test, pearson's chi-squared test, fisher's exact test, kolmogorov-smirnov test
- bessel's correction
- bonferroni correction
- family-wise error rate


## Calculus

concepts
- support of a function

functions
- sign function, kronecker delta function
- dirac delta function, heaviside step function, ramp function
- gamma function, beta function

operations
- convolution, circular convolution, discrete convolution
- integral transform
- Fourier transform
- subderivative
- automatic differentiation, reverse-mode autodiff
- taylor expansion


## Algorithms

concepts
- big theta, big oh, big omega
- master theorem
- backtracking
- dynamic programming
- reductions
- Turing machine, nondeterministic Turing machine
- P, NP, NP-complete, NP-hard problems

sorting
- N*logN worst case proof (compare-based)
- sort stability
- insertion sort
- merge sort
- quick sort
- heap sort
- count/bucket sort
- reservoir sampling

selection and search
- quick select
- range search
- interval search

graphs
- biconnectivity, planarity, isomorphism
- representation: set of edges, adjacency matrix, adjacency list
- dfs, bfs
- find shortest (directed) path
- check (strong/directed) connectivity
- find (strong/directed) connected components - Kosaraju-Sharir algorithm
- check if graph is bipartite / has loops / has euler tour / has hamilton tour / is planar
- check if graphs are isomorphic
- topological sort
- minimum spanning tree (MST)
- Kruskal algorithm (using union-find)
- Prim algorithm (lazy, eager)
- shortest path tree (SPT)
- Dijkstra algorithm for non-negative weights
- topological sort for finding the SPT in edge-weighted DAGs
- Bellman-Ford algorithm for graphs without negative cycles
- st-flow, st-cut, maximum flow, mincut
- Ford-Fulkerson algorithm (shortest/fattest path) for max flow

strings
- key-indexed counting
- LSD radix sort
- MSD radix sort
- 3-way string quicksort
- key-in-context search
- longest repeating substring search
- Manber-Myers suffix array sort
- Knuth-Morris-Pratt algorithm (deterministic finite state machine DFA + stream search)
- Boyer-Moore algorithm (skip table)
- Rabin-Karp algorithm (fingerprint search)
- regex <=> DFA duality (Kleene theoreme), exponentiality of DFA space
- regex => NFA; NFA simulation, NFA construction
- RLE, Huffman compression, LZW compression

Data Structures
- disjoint-set (union-find)
- stack (linkedlist/array), queue (array/linkedlist), deque
- heap (binary heap, d-ary heap, fibonacci heap?, binomial heap?)
- bst (2-3 tree, red-black tree, b-tree)
- skip list, segment tree, dekart tree
- 2d space tree (grid, 2d tree, quadtree, bsp tree, kd-tree, interval search tree)
- hash table (separate chaining, linear probing, two-probe chaining, cuckoo hashing)
- set (hash table, bst)
- suffix array, suffix tree
- fenwick tree
- r-way trie, ternary search trie (TST), TST with r^2-branching at root, patricia trie/radix tree

additional
- Floyd algoritthm
- Kadan algorithm
- Brent algorithm

Copied from:
1. [xdralex](https://gist.github.com/xdralex/133f88592b4908090164ba72a6f0718d)
