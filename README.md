# Awesome Best Papers

本项目收集了各大顶会的最佳论文（*自 2013 年起*），包括*论文名、论文链接、作者名以及机构名*（多个机构并列时记录第一个）。所有的数据都是手工输入的，所以 *如果你发现有任何错误，欢迎开个 issue 提醒我*。你也可以通过 *创建 pull requests* 的方式参与到这个项目中来，加入机器人、优化、图形学等本项目未包含领域的顶会最佳论文信息（**非常欢迎**）。项目中的部分数据参考了 Jeff Huang 的网站，他收集了自 1996 年起计算机科学领域部分顶会的最佳论文，如果你感兴趣，也可以访问[他的网站](https://jeffhuang.com/best_paper_awards.html)。

This repo collects best papers from various top conferences (*since 2013*), including *paper names and links, author names and organization names* (use the first one when the author works for multiple organizations). All the data are collected and entered by hand, so *feel free to open an issue if you find anything wrong*. You can also contribute to this repo by *creating pull requests* and add best papers from top conferences in research areas that are not covered in this repo, e.g., robotics, optimization, computer graphics and etc (**very welcome**). Part of the data is collected by Jeff Huang. If you are interested in more best papers in computer science (since 1996), you can visit [his website](https://jeffhuang.com/best_paper_awards.html).

# Table of Content
| Domain | Conferences |
| :-  |  :-   |
| Cross-domain | [AAAI](#aaai), [IJCAI](#ijcai), [NeurIPS](#neurips), [ICML](#icml), [ICLR](#iclr), [WWW](#www) |
| Data Mining and Information Retrieval | [KDD](#kdd), [SIGIR](#sigir), [CIKM](#cikm), [ICDM](#icdm), [WSDM](#wsdm), [RecSys](#recsys) |
| Computer Vision | [CVPR](#cvpr), [ICCV](#iccv), [ECCV](#eccv) |
| Natural Language Processing | [ACL](#acl), [EMNLP](#emnlp), [NAACL](#naacl) |

# Format

This repo is writen in Markdown. Each entry follows the following format. 

```markdown
| Year | Paper |
| :-:  |  :-   |
| 2020 | **[Paper title](Link to pdf)**<br>AuthorName (Organization); AuthorName (Organization) |
```

When multiple best papers are available:

```markdown
| Year | Paper |
| :-:  |  :-   |
| 2020 | 1. **[Paper title](Link to pdf)**<br>AuthorName (Organization); AuthorName (Organization)<br>2. **[Paper title](Link to pdf)**<br>AuthorName (Organization); AuthorName (Organization) |
```

<a id="aaai"></a>
## AAAI
The full list of AAAI outstanding papers (including best student papers and their Honorable Mentions) is presented on [this website](https://aaai.org/Awards/paper.php).

| Year | Paper |
|:-:|:-|
| 2020 | **[WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/pdf/1907.10641.pdf)**<br>Keisuke Sakaguchi (Allen Institute for Artificial Intelligence); Ronan Le Bras (Allen Institute for Artificial Intelligence); Chandra Bhagavatula (Allen Institute for Artificial Intelligence); Yejin Choi (University of Washington) |
| 2019 | **[How to Combine Tree-Search Methods in Reinforcement Learning](https://arxiv.org/pdf/1809.01843.pdf)**<br>Yonathan Efroni (Technion); Gal Dala (Technion); Bruno Scherrer (INRIA); Shie Mannor (Technion) |
| 2018 | **[Memory-Augmented Monte Carlo Tree Search](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17139/15841)**<br>Chenjun Xiao (University of Alberta); Jincheng Mei (University of Alberta); Martin Müller (University of Alberta) |
| 2017 | **[Label-Free Supervision of Neural Networks with Physics and Domain Knowledge](https://arxiv.org/pdf/1609.05566.pdf)**<br>Russell Stewart (Stanford University);  Stefano Ermon (Stanford University) |
| 2016 | **[Bidirectional Search That Is Guaranteed to Meet in the Middle](https://people.engr.tamu.edu/guni/Papers/AAAI16-MM.pdf)**<br>Robert C. Holte (University of Alberta); Ariel Felner (Ben-Gurion University); Guni Sharon (Ben-Gurion University); Nathan R. Sturtevant (University of Denver) |
| 2015 | **[From Non-Negative to General Operator Cost Partitioning](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9983/9762)**<br>Florian Pommerening (University of Basel); Malte Helmert (University of Basel); Gabriele Röger (University of Basel); Jendrik Seipp (University of Basel) |
| 2014 | **[Recovering from Selection Bias in Causal and Statistical Inference](http://ftp.cs.ucla.edu/pub/stat_ser/r425.pdf)**<br>Elias Bareinboim (UCLA); Jin Tian (Iowa State University); Judea Pearl (UCLA) |
| 2013 | 1. **[HC-Search: Learning Heuristics and Cost Functions for Structured Prediction](http://web.engr.oregonstate.edu/~afern/papers/aaai13-hcsearch.pdf)**<br>Janardhan Rao Doppa (Oregon State University); Alan Fern (Oregon State University); Prasad Tadepalli (Oregon State University)<br>2. **[SMILe: Shufﬂed Multiple-Instance Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/view/6405/7131)**<br>Gary Doran (Case Western Reserve University); Soumya Ray (Case Western Reserve University) |

<a id="ijcai"></a>
## IJCAI

IJCAI Distinguished papers.

| Year | Paper |
|:-:|:-|
| 2019 | **[Boosting for Comparison-Based Learning](https://arxiv.org/pdf/1810.13333.pdf)**<br>Michaël Perrot (Max-Planck-Institute for Intelligent Systems); Ulrike von Luxburg (University of Tubingen) |
| 2018 | 1. **[Reasoning about Consensus when Opinions Diffuse through Majority Dynamics](https://www.ijcai.org/Proceedings/2018/0007.pdf)**<br>Vincenzo Auletta (University of Salerno); Diodato Ferraioli (University of Salerno); Gianluigi Greco (University of Calabria)<br>2. **[SentiGAN: Generating Sentimental Texts via Mixture Adversarial Networks](https://www.ijcai.org/Proceedings/2018/0618.pdf)**<br>Ke Wang (Peking University); Xiaojun Wan (Peking University)<br>3. **[From Conjunctive Queries to Instance Queries in Ontology-Mediated Querying](https://www.ijcai.org/Proceedings/2018/0250.pdf)**<br>Cristina Feier (University of Bremen); Carsten Lutz (University of Bremen); Frank Wolte (University of Liverpool)<br>4. **[What game are we playing? End-to-end learning in normal and extensive form games](https://arxiv.org/pdf/1805.02777.pdf)**<br>Chun Kai Ling (Carnegie Mellon University); Fei Fang (Carnegie Mellon University); J. Zico Kolter (Carnegie Mellon University)<br>5. **[Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/Proceedings/2018/0643.pdf)**<br>Hao Zhou (Tsinghua University); Tom Young (Beijing Institute of Technology); Minlie Huang (Tsinghua University); Haizhou Zhao, Sogou Inc.; Jingfang Xu, Sogou Inc.; Xiaoyan Zhu (Tsinghua University)<br>6. **[R-SVM+: Robust Learning with Privileged Information](https://www.ijcai.org/Proceedings/2018/0334.pdf)**<br>Xue Li (Wuhan University); Bo Du (Wuhan University); Chang Xu (University of Sydney); Yipeng Zhang (Wuhan University); Lefei Zhang (Wuhan University); Dacheng Tao (University of Sydney)<br>7. **[A Degeneracy Framework for Graph Similarity](https://www.ijcai.org/Proceedings/2018/0360.pdf)**<br>Giannis Nikolentzos, École Polytechnique; Polykarpos Meladianos (Athens University of Economics and Business); Stratis Limnios, École Polytechnique; Michalis Vazirgiannis (École Polytechnique) |
| 2017 | **[Foundations of Declarative Data Analysis Using Limit Datalog Programs](https://www.cs.ox.ac.uk/people/boris.motik/pubs/kcgkmh17limit-datalog.pdf)**<br>Mark Kaminski (University of Oxford); Bernardo Cuenca Grau (University of Oxford); Egor V. Kostylev (University of Oxford); Boris Motik (University of Oxford); Ian Horrocks (University of Oxford) |
| 2016 | **[Hierarchical Finite State Controllers for Generalized Planning](https://arxiv.org/pdf/1911.02887.pdf)**<br>Javier Segovia (Universitat Pompeu Fabra); Sergio Jiménez (Universitat Pompeu Fabra); Anders Jonsson (Universitat Pompeu Fabra) |
| 2015 | 1. **[Bayesian Active Learning for Posterior Estimation](https://www.cs.cmu.edu/~schneide/kandasamyIJCAI15activePostEst.pdf)**<br>Kirthevasan Kandasamy (Carnegie Mellon University); Jeff Schneider (Carnegie Mellon University); Barnabas Poczos (Carnegie Mellon University)<br>2. **[Recursive Decomposition for Nonconvex Optimization](https://arxiv.org/pdf/1611.02755.pdf)**<br>Abram L. Friesen (University of Washington); Pedro Domingos (University of Washington) |
| 2013 | 1. **[Bayesian Optimization in High Dimensions via Random Embeddings](https://www.cs.ubc.ca/~hutter/papers/13-IJCAI-BO-highdim.pdf)**<br>Ziyu Wang (University of British Columbia); Masrour Zoghi (University of Amsterdam); Frank Hutter (Freiberg University); David Matheson (University of British Columbia); Nando de Freitas (University of British Columbia)<br>2. **[Flexibility and Decoupling in the Simple Temporal Problem](https://www.ijcai.org/Proceedings/13/Papers/356.pdf)**<br>Michel Wilson (Delft University of Technology); Tomas Klos (Delft University of Technology); Cees Witteveen (Delft University of Technology); Bob Huisman (Delft University of Technology) |

<a id="neurips"></a>
## NeurIPS
| Year | Paper |
|:-:|:-|
| 2020 | 1. **[No-Regret Learning Dynamics for Extensive-Form Correlated Equilibrium](https://arxiv.org/pdf/2004.00603.pdf)**<br>Andrea Celli (Politecnico di Milano); Alberto Marchesi (Politecnico di Milano); Gabriele Farina (Carnegie Mellon University); Nicola Gatti (Politecnico di Milano)<br>2. **[Improved Guarantees and a Multiple-Descent Curve for Column Subset Selection and the Nyström Method](https://arxiv.org/pdf/2002.09073.pdf)**<br>Michal Derezinski (UC Berkeley); Rajiv Khanna (UC Berkeley); Michael W. Mahoney (UC Berkeley)<br>3. **[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)**<br>Tom B. Brown (OpenAI); Benjamin Mann (OpenAI); Nick Ryder (OpenAI); Melanie Subbiah (OpenAI); Jared D. Kaplan (Johns Hopkins University); Prafulla Dhariwal (OpenAI); Arvind Neelakantan (OpenAI); Pranav Shyam (OpenAI); Girish Sastry (OpenAI); Amanda Askell (OpenAI); Sandhini Agarwal (OpenAI); Ariel Herbert-Voss (OpenAI); Gretchen M. Krueger (OpenAI); Tom Henighan (OpenAI); Rewon Child (OpenAI); Aditya Ramesh (OpenAI); Daniel Ziegler (OpenAI); Jeffrey Wu (OpenAI); Clemens Winter (OpenAI); Chris Hesse (OpenAI); Mark Chen (OpenAI); Eric Sigler (OpenAI); Mateusz Litwin (OpenAI); Scott Gray (OpenAI); Benjamin Chess (OpenAI); Jack Clark (OpenAI); Christopher Berner (OpenAI); Sam McCandlish (OpenAI); Alec Radford (OpenAI); Ilya Sutskever (OpenAI); Dario Amodei (OpenAI) |
| 2019 | **[Distribution-Independent PAC Learning of Halfspaces with Massart Noise](https://arxiv.org/pdf/1906.10075.pdf)**<br>Ilias Diakonikolas (University of Wisconsin-Madison); Themis Gouleakis (Max Planck Institute for Informatics) |
| 2018 | 1. **[Non-delusional Q-learning and Value-iteration](https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration.pdf)**<br>Tyler Lu; Dale Schuurmans; Craig Boutilier<br>2. **[Optimal Algorithms for Non-Smooth Distributed Optimization in Networks](https://papers.nips.cc/paper/7539-optimal-algorithms-for-non-smooth-distributed-optimization-in-networks.pdf)**<br>Kevin Scaman ; Francis Bach ; Sebastien Bubeck ; Laurent Massoulié ; Yin Tat Lee<br>3. **[Nearly Tight Sample Complexity Bounds for Learning Mixtures of Gaussians via Sample Compression Schemes](https://papers.nips.cc/paper/7601-nearly-tight-sample-complexity-bounds-for-learning-mixtures-of-gaussians-via-sample-compression-schemes.pdf)**<br>Hassan Ashtiani ; Shai Ben-David ; Nick Harvey ; Christopher Liaw ; Abbas Mehrabian ; Yaniv Plan<br>4. **[Neural Ordinary Differential Equations](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf)**<br>Tian Qi Chen ; Yulia Rubanova ; Jesse Bettencourt ; David Duvenaud |
| 2017 | 1. **[Safe and Nested Subgame Solving for Imperfect-Information Games](http://papers.nips.cc/paper/6671-safe-and-nested-subgame-solving-for-imperfect-information-games.pdf)**<br>Noam Brown (Carnegie Mellon University); Tuomas Sandholm(Carnegie Mellon University)<br>2. **[Variance-based Regularization with Convex Objectives](https://papers.nips.cc/paper/6890-variance-based-regularization-with-convex-objectives.pdf)**<br>Hongseok Namkoong (Stanford University); John Duchi (Stanford University)<br>3. **[A Linear-Time Kernel Goodness-of-Fit Test](https://arxiv.org/pdf/1705.07673.pdf)**<br>Wittawat Jitkrittum (University College London), Wenkai Xu (University College London), Zoltan Szabo (École Polytechnique), Kenji Fukumizu (The Institute of Statistical Mathematics), Arthur Gretton (University College London) |
| 2016 | **[Value Iteration Networks](http://papers.nips.cc/paper/6046-value-iteration-networks.pdf)**<br>Aviv Tamar (UC Berkeley); Yi Wu (UC Berkeley); Garrett Thomas (UC Berkeley); Sergey Levine (UC Berkeley); Pieter Abbeel (UC Berkeley)|
| 2015 | 1. **[Competitive Distribution Estimation: Why is Good-Turing Good](http://papers.nips.cc/paper/5762-competitive-distribution-estimation-why-is-good-turing-good.pdf)**<br>Alon Orlitsky (UC San Diego); Ananda Suresh (UC San Diego)<br>2. **[Fast Convergence of Regularized Learning in Games](http://papers.nips.cc/paper/5763-fast-convergence-of-regularized-learning-in-games.pdf)**<br>Vasilis Syrgkanis (Microsoft Research); Alekh Agarwal (Microsoft Research); Haipeng Luo (Princeton University); Robert Schapire (Microsoft Research) |
| 2014 | 1. **[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](http://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)**<br>Anshumali Shrivastava (Cornell University); Ping Li (Rutgers University)<br> 2. **[A\* Sampling](http://papers.nips.cc/paper/5449-a-sampling.pdf)**<br>Christopher Maddison (University of Toronto); Daniel Tarlow (Microsoft Research); Tom Minka (Microsoft Research) |
| 2013 | 1. **[A memory frontier for complex synapses](http://papers.nips.cc/paper/4872-a-memory-frontier-for-complex-synapses.pdf)**<br>Subhaneil Lahiri (Stanford University); Surya Ganguli (Stanford University)<br>2. **[Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints](http://papers.nips.cc/paper/4911-submodular-optimization-with-submodular-cover-and-submodular-knapsack-constraints.pdf)**<br>Rishabh Iyer (University of Washington, Seattle); Jeff Bilmes (University of Washington, Seattle)<br>3. **[Scalable Influence Estimation in Continuous-Time Diffusion Networks](http://papers.nips.cc/paper/4857-scalable-influence-estimation-in-continuous-time-diffusion-networks.pdf)**<br>Nan Du (Georgia Tech); Le Song (Georgia Tech); Manuel Gomez-Rodriguez (MPI for Intelligent Systems); Hongyuan Zha (Georgia Tech) |

<a id="icml"></a>
## ICML

ICML Outstanding Papers.

| Year | Paper |
|:-:|:-|
| 2020 | 1. **[On Learning Sets of Symmetric Elements](https://proceedings.icml.cc/static/paper_files/icml/2020/1625-Paper.pdf)**<br>Haggai Maron (NVIDIA Research); Or Litany (Stanford University); Gal Chechik (Stanford University); Ethan Fetaya (Bar Ilan University)<br>2. **[Tuning-free Plug-and-Play Proximal Algorithm for Inverse Imaging Problems](https://proceedings.icml.cc/static/paper_files/icml/2020/4134-Paper.pdf)**<br>Kaixuan Wei (Beijing Institute of Technology); Angelica I Aviles-Rivero (University of Cambridge); Jingwei Liang (University of Cambridge); Ying Fu (Beijing Institute of Technology); Carola-Bibiane Schönlieb (University of Cambridge); Hua Huang (Beijing Institute of Technology) |
| 2019 | 1. **[Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](http://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf)**<br>Francesco Locatello (ETH Zurich); Stefan Bauer (MaxPlanck Institute for Intelligent Systems); Mario Lucic (Google Brain); Gunnar Rätsch (Google Brain); Sylvain Gelly  (ETH Zurich); Bernhard Schölkopf (MaxPlanck Institute for Intelligent Systems); Olivier Bachem (Google Brain)<br>2. **[Rates of Convergence for Sparse Variational Gaussian Process Regression](https://arxiv.org/pdf/1903.03571.pdf)**<br>David R. Burt (University of Cambridge); Carl Edward Rasmussen (University of Cambridge); Mark van der Wilk (PROWLER.io) |
| 2018 | 1. **[Delayed Impact of Fair Machine Learning](https://arxiv.org/pdf/1803.04383.pdf)**<br>Lydia T. Liu (University of California Berkeley); Sarah Dean (University of California Berkeley); Esther Rolf (University of California Berkeley); Max Simchowitz (University of California Berkeley); Moritz Hardt (University of California Berkeley)<br>2. **[Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/pdf/1802.00420.pdf)**<br>Anish Athalye (Massachusetts Institute of Technology); Nicholas Carlini (University of California Berkeley); David Wagner (University of California Berkeley) |
| 2017 | **[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf)**<br>Pang Wei Koh (Stanford University); Percy Liang (Stanford University) |
| 2016 | 1. **[Ensuring Rapid Mixing and Low Bias for Asynchronous Gibbs Sampling](http://proceedings.mlr.press/v48/sa16.pdf)**<br>Christopher De Sa (Stanford University); Chris Re (Stanford University); Kunle Olukotun (Stanford University)<br>2. **[Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf)**<br>Aaron Van den Oord (Google); Nal Kalchbrenner (Google); Koray Kavukcuoglu (Google)<br>3. **[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)**<br>Ziyu Wang (Google); Tom Schaul (Google); Matteo Hessel (Google); Hado van Hasselt (Google); Marc Lanctot (Google); Nando de Freitas (University of Oxford) |
| 2015 | 1. **[A Nearly-Linear Time Framework for Graph-Structured Sparsity](http://proceedings.mlr.press/v37/hegde15.pdf)**<br>Chinmay Hegde (Massachusetts Institute of Technology); Piotr Indyk (Massachusetts Institute of Technology); Ludwig Schmid (Massachusetts Institute of Technology)<br>2. **[Optimal and Adaptive Algorithms for Online Boosting](https://arxiv.org/pdf/1502.02651.pdf)**<br>Alina Beygelzimer (Yahoo! Research); Satyen Kale (Yahoo! Research); Haipeng Luo (Princeton University) |
| 2014 | **[Understanding the Limiting Factors of Topic Modeling via Posterior Contraction Analysis](http://proceedings.mlr.press/v32/tang14.pdf)**<br>Jian Tang (Peking University); Zhaoshi Meng (University of Michigan); XuanLong Nguyen (University of Michigan); Qiaozhu Mei (University of Michigan); Ming Zhang (Peking University) |
| 2013 | 1. **[Vanishing Component Analysis](https://mlai.cs.huji.ac.il/publications/Amir_Globerson/files/Vanishing%20Component%20Analysis.pdf)**<br>Roi Livni (The Hebrew University of Jerusalum); David Lehavi (Hewlett-Packard Labs); Sagi Schein (Hewlett-Packard Labs); Hila Nachlieli (Hewlett-Packard Labs); Shai Shalev Shwartz (The Hebrew University of Jerusalum); Amir Globerson (The Hebrew University of Jerusalum)<br>2. **[Fast Semidifferential-based Submodular Function Optimization](https://arxiv.org/pdf/1308.1006.pdf)**<br>Rishabh Iyer (University of Washington); Stefanie Jegelka (University of California Berkeley); Jeff Bilmes (University of Washington) |

<a id="www"></a>
## WWW
| Year | Paper |
|:-:|:-|
| 2020 | **[Open Intent Extraction from Natural Language Interactions](https://dl.acm.org/doi/abs/10.1145/3366423.3380268)**<br>Nikhita Vedula (The Ohio State University); Nedim Lipka (Adobe); Pranav Maneriker (The Ohio State University); Srinivasan Parthasarathy (The Ohio State University) |
| 2019 | 1. **[Ermes: Emoji-Powered Representation Learning for Cross-Lingual Sentiment Classification](https://arxiv.org/pdf/1806.02557.pdf)**<br>Zhenpeng Chen (Peking University); Sheng Shen (University of California, Berkeley); Ziniu Hu (University of California, Berkeley); Xuan Lu (Peking University); Qiaozhu Mei (University of Michigan); Xuanzhe Liu (Peking University)<br>2. **[OUTGUARD: Detecting In-Browser Covert Cryptocurrency Mining in the Wild](https://nikita.ca/papers/outguard-www19.pdf)**<br>Amin Kharraz (University of Illinois at Urbana-Champaign); Zane Ma (University of Illinois at Urbana-Champaign); Paul Murley (University of Illinois at Urbana-Champaign); Charles Lever (Georgia Institute of Technology); Joshua Mason (University of Illinois at Urbana-Champaign); Andrew Miller (University of Illinois at Urbana-Champaign); Nikita Borisov (University of Illinois at Urbana-Champaign); Manos Antonakakis (Georgia Institute of Technology); Michael Bailey (University of Illinois at Urbana-Champaign) |
| 2018 | **[HighLife: Higher-arity Fact Harvesting](https://dl.acm.org/doi/10.1145/3178876.3186000)**<br>Patrick Ernst (Saarland Informatics Campus in Saarbrücken); Amy Siu (Saarland Informatics Campus in Saarbrücken); Gerhard Weikum (Saarland Informatics Campus in Saarbrücken) |
| 2017 | [Currently missing.](http://www.www2017.com.au/) |
| 2016 | **[Social Networks Under Stress](https://arxiv.org/pdf/1602.00572.pdf)**<br>Daniel Romero (University of Michigan); Brian Uzzi (Northwestern University); Jon Kleinberg (Cornell University) |
| 2015 | **[HypTrails: A Bayesian Approach for Comparing Hypotheses About Human Trails on the Web](https://arxiv.org/pdf/1411.2844.pdf)**<br>Philipp Singer (GESIS - Leibniz Institute for the Social Sciences); Denis Helic (Graz University of Technology); Andreas Hotho, University of Würzburg; Markus Strohmaier (GESIS - Leibniz Institute for the Social Sciences） |
| 2014 | **[Efficient Estimation for High Similarities using Odd Sketches](https://www.itu.dk/people/pagh/papers/oddsketch.pdf)**<br>Michael Mitzenmacher (Harvard University); Rasmus Pagh (IT University of Copenhagen); Ninh Pham (IT University of Copenhagen) |
| 2013 | **[No Country for Old Members: User Lifecycle and Linguistic Change in Online Communities](https://nlp.stanford.edu/pubs/linguistic_change_lifecycle.pdf)**<br>Cristian Danescu-Niculescu-Mizil (Stanford University); Robert West (Stanford University); Dan Jurafsky (Stanford University); Jure Leskovec (Stanford University); Christopher Potts (Stanford University) |

---

<a id="kdd"></a>
## KDD

KDD has two paper tracks, i.e., the Research Track and the Applied Data Science Track. I only report the best papers for the research track in this repo but the ADS track is remarkable as well.

| Year | Paper |
|:-:|:-|
| 2020 | **[On Sampled Metrics for Item Recommendation](https://dl.acm.org/doi/pdf/10.1145/3394486.3403226)**<br>Walid Krichene (Google Research); Steffen Rendle (Google Research) |
| 2019 | **[Network Density of States](https://www.cs.cornell.edu/~bindel/papers/2019-kdd.pdf)**<br>Kun Dong (Cornell University); Austin Benson (Cornell University); David Bindel (Cornell University) |
| 2018 | **[Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf)**<br>Daniel Zügner (Technical University of Munich); Amir Akbarnejad (Technical University of Munich); Stephan Günnemann (Technical University of Munich) |
| 2017 | **[Accelerating Innovation Through Analogy Mining](https://arxiv.org/pdf/1706.05585.pdf)**<br>Tom Hope (Hebrew University of Jerusalem); Joel Chan (Carnegie Mellon University); Aniket Kittur (Carnegie Mellon University); Dafna Shahaf (Hebrew University of Jerusalem) |
| 2016 | **[FRAUDAR: Bounding Graph Fraud in the Face of Camouflage](https://bhooi.github.io/papers/fraudar_kdd16.pdf)**<br>Bryan Hooi (Carnegie Mellon University); Hyun Ah Song (Carnegie Mellon University); Alex Beutel (Carnegie Mellon University); Neil Shah (Carnegie Mellon University); Kijung Shin (Carnegie Mellon University); Christos Faloutsos (Carnegie Mellon University) |
| 2015 | **[Efficient Algorithms for Public-Private Social Networks](https://www.epasto.org/papers/kdd2015.pdf)**<br>Flavio Chierichetti (Sapienza University of Rome); Alessandro Epasto (Brown University); Ravi Kumar (Google); Silvio Lattanzi (Google); Vahab Mirrokni (Google) |
| 2014 | **[Reducing the Sampling Complexity of Topic Models](http://www.sravi.org/pubs/fastlda-kdd2014.pdf)**<br>Aaron Li (Carnegie Mellon University); Amr Ahmed (Google); Sujith Ravi (Google); Alexander Smola (Carnegie Mellon University) |
| 2013 | **[Simple and Deterministic Matrix Sketching](https://arxiv.org/pdf/1206.0594.pdf)**<br>Edo Liberty (Yahoo! Research) |

<a id="sigir"></a>
## SIGIR
The full list of SIGIR best papers is presented on [this website](https://sigir.org/awards/best-paper-awards/).

| Year | Paper |
|:-:|:-|
| 2020 | **[Controlling Fairness and Bias in Dynamic Learning-to-Rank](https://arxiv.org/pdf/2005.14713.pdf)**<br>Marco Morik (Technische Univerität Berlin); Ashudeep Singh (Cornell University); Jessica Hong (Cornell University); Thorsten Joachims (Cornell University) |
| 2019 | **[Variance Reduction in Gradient Exploration for Online Learning to Rank](https://arxiv.org/pdf/1906.03766.pdf)**<br> Huazheng Wang (University of Virginia); Sonwoo Kim (University of Virginia); Eric McCord-Snook (University of Virginia); Qingyun Wu (University of Virginia); Hongning Wang (University of Virginia) |
| 2018 | **[Should I Follow the Crowd? A Probabilistic Analysis of the Effectiveness of Popularity in Recommender Systems](http://ir.ii.uam.es/pubs/sigir2018.pdf)**<br>Rocío Cañamares (Universidad Autónoma de Madrid); Pablo Castells (Universidad Autónoma de Madrid) |
| 2017 | **[BitFunnel: Revisiting Signatures for Search](https://danluu.com/bitfunnel-sigir.pdf)**<br>Bob Goodwin (Microsoft); Michael Hopcroft (Microsoft); Dan Luu (Microsoft); Alex Clemmer (Heptio); Mihaela Curmei (Microsoft); Sameh Elnikety (Microsoft); Yuxiong He (Microsoft) |
| 2016 | **[Understanding Information Need: An fMRI Study](https://core.ac.uk/download/pdf/42370897.pdf)**<br>Yashar Moshfeghi (University of Glasgow); Peter Triantafillou (University of Glasgow); Frank E. Pollick (University of Glasgow) |
| 2015 | **[QuickScorer: A Fast Algorithm to Rank Documents with Additive Ensembles of Regression Trees](https://www.cse.cuhk.edu.hk/irwin.king/_media/presentations/sigir15bestpaperslides.pdf)**<br>Claudio Lucchese (Istituto di Scienza e Tecnologie dell'Informazione); Franco Maria Nardini (Istituto di Scienza e Tecnologie dell'Informazione); Salvatore Orlando, Università di Venezia; Raffaele Perego (Istituto di Scienza e Tecnologie dell'Informazione); Nicola Tonellotto (Istituto di Scienza e Tecnologie dell'Informazione); Rossano Venturini (Istituto di Scienza e Tecnologie dell'Informazione) |
| 2014 | **[Partitioned Elias-Fano Indexes](http://groups.di.unipi.it/~ottavian/files/elias_fano_sigir14.pdf)**<br>Giuseppe Ottaviano (Istituto di Scienza e Tecnologie dell'Informazione); Rossano Venturini (Università di Pisa) |
| 2013 | **[Beliefs and Biases in Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/WhiteSIGIR2013.pdf)**<br>Ryen W. White (Microsoft Research) |

<a id="cikm"></a>
## CIKM

| Year | Paper |
|:-:|:-|
| 2020 | 1. **[Do People and Neural Networks Pay Attention to the Same Words? Studying Eye-tracking Data for Non-factoid QA Evaluation](http://marksanderson.org/publications/my_papers/CIKM__Do_People_and_Neural_Networks_Pay_Attention_to_the_Same_Words_.pdf)**<br>Valeriya Bolotova-Baranova (RMIT University); Vladislav Blinov (Ural Federal University); Yukun Zheng (Tsinghua University); Mark Sanderson (University of Massachusetts Amherst); Falk Scholer (RMIT University); Bruce Croft (RMIT University)<br>2. **[FANG: Leveraging Social Context for Fake News Detection Using Graph Representation](https://arxiv.org/pdf/2008.07939.pdf)**<br>Van-Hoang Nguyen (National University of Singapore); Kazunari Sugiyama (Kyoto University); Preslav Nakov (Qatar Computing Research Institute; HBKU); Min-Yen Kan (National University of Singapore) |
| 2019 | **[AutoGRD: Model Recommendation Through Graphical Dataset Representation](https://dl.acm.org/doi/10.1145/3357384.3357896)**<br>Noy Cohen-Shapira (Ben Gurion University of the Negev); Lior Rokach (Ben Gurion University of the Negev); Bracha Shapira (Ben Gurion University of the Negev); Gilad Katz (Ben Gurion University of the Negev); Roman Vainshtein (Ben Gurion University of the Negev) |
| 2018 | **[Relevance estimation with multiple information sources on search engine result pages](http://www.thuir.cn/group/~YQLiu/publications/CIKM18Zhang.pdf)**<br>Junqi Zhang (Tsinghua University); Yiqun Liu (Tsinghua University); Shaoping Ma (Tsinghua University); Qi Tian (University of Texas at San Antonio) |
| 2017 | **[Hike: A Hybrid Human-Machine Method for Entity Alignment in Large-Scale Knowledge Bases](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/CIKM2017-Hike.pdf)**<br>Yan Zhuang (Tsinghua University); Guoliang Li (Tsinghua University); Zhuojian Zhong (Tsinghua University); Jianhua Feng (Tsinghua University) |
| 2016 | **[Vandalism Detection in Wikidata](https://webis.de/downloads/publications/papers/stein_2016m.pdf)**<br>Stefan Heindorf (Paderborn University); Martin Potthast (Bauhaus-Universität Weimar); Benno Stein (Bauhaus-Universität Weimar); Gregor Engels (Paderborn University) |
| 2015 | **[Assessing the Impact of Syntactic and Semantic Structures for Answer Passages Reranking](http://casa.disi.unitn.it/moschitti/since2013/2015_CIKM_Tymoshenko_Assessing_Impact_Syntactic.pdf)**<br>Kateryna Tymoshenko (University of Trento); Alessandro Moschitti (Qatar Computing Research Institute) |
| 2014 | **[Cross-Device Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/MontanezCIKM2014.pdf)**<br>George Montanez (Carnegie Mellon University); Ryen White (Microsoft Research); Xiao Huang (Microsoft) |
| 2013 | **[Penguins in Sweaters, or Serendipitous Entity Search on User-generated Content](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.418.5058&rep=rep1&type=pdf)**<br>Ilaria Bordino (Yahoo! Research); Yelena Mejova (Yahoo! Research); Mounia Lalmas (Yahoo! Research) |

<a id="icdm"></a>
## ICDM
The full list of ICDM best papers and best student papers is presented on [this website](http://39.104.72.142:802/Awards/BestPapers.shtml).

| Year | Paper |
|:-:|:-|
| 2020 | **[Co-Embedding Network Nodes and Hierarchical Labels with Taxonomy Based Generative Adversarial Networks](https://jiyang3.web.engr.illinois.edu/files/taxogan.pdf)**<br>Carl Yang (University of Illinois, Urbana Champaign); Jieyu Zhang (University of Illinois, Urbana Champaign); Jiawei Han (University of Illinois, Urbana Champaign) |
| 2019 | **[Deep Multi-attributed Graph Translation with Node-Edge Co-evolution](https://mason.gmu.edu/~lzhao9/materials/papers/ICDM_2019_NEC_DGT-final.pdf)**<br>Xiaojie Guo (George Mason University); Liang Zhao (George Mason University), Cameron Nowzari (George Mason University); Setareh Rafatirad (George Mason University); Houman Homayoun (George Mason University); Sai Manoj Pudukotai Dinakarrao (George Mason University) |
| 2018 | **[Discovering Reliable Dependencies from Data: Hardness and Improved Algorithms](https://arxiv.org/pdf/1809.05467.pdf)**<br>Panagiotis Mandros (Max Planck Institute for Informatics); Mario Boley (Max Planck Institute for Informatics); Jilles Vreeken (Max Planck Institute for Informatics) |
| 2017 | **[TensorCast: Forecasting with Context using Coupled Tensors](http://www.cs.cmu.edu/~christos/PUBLICATIONS/icdm17-tensor-cast-CR.pdf)**<br>Miguel Ramos de Araujo (Carnegie Mellon University); Pedro Manuel Pinto Ribeiro (University of Porto); Christos Faloutsos (Carnegie Mellon University) |
| 2016 | **[KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift]()**<br>Viktor Losing (Bielefeld University); Barbara Hammer (Bielefeld University); Heiko Wersing (HONDA Research Institute Europe) |
| 2015 | **[Diamond Sampling for Approximate Maximum All-pairs Dot-product (MAD) Search](https://arxiv.org/pdf/1506.03872.pdf)**<br>Grey Ballard (Sandia National Laboratories); Seshadhri Comandur (University of California, Santa Cruz); Tamara Kolda (Sandia National Laboratories); Ali Pinar (Sandia National Laboratories) |
| 2014 | **[Ternary Matrix Factorization](https://ieeexplore.ieee.org/document/7023357/)**<br>Sam Maurus (Technische Universität München); Claudia Plant (Technische Universität München) |
| 2013 | **[Reconstructing Individual Mobility from Smart Card Transactions: A Space Alignment Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/icdm13_embed.pdf)**<br>Nicholas Jing Yuan (Microsoft Research); Yingzi Wang (University of Science and Technology of China); Fuzheng Zhang (University of Science and Technology of China); Xing Xie (Microsoft Research); Guang-Zhong Sun (University of Science and Technology of China) |

<a id="wsdm"></a>
## WSDM

| Year | Paper |
|:-:|:-|
| 2020 | **[The Power of Pivoting for Exact Clique Counting](https://arxiv.org/pdf/2001.06784.pdf)**<br>Shweta Jain (University of California, Santa Cruz); C. Seshadhri (University of California, Santa Cruz) |
| 2019 | **[Slice: Scalable Linear Extreme Classifiers trained on 100 Million Labels for Related Searches](http://manikvarma.org/pubs/jain19.pdf)**<br> Himanshu Jain (Indian Institute of Technology Delhi); Venkatesh Balasubramanian (Microsoft AI & Research); Bhanu Chunduri (Microsoft AI & Research); Manik Varma (Microsoft AI & Research) |
| 2018 | **[Index Compression Using Byte-Aligned ANS Coding and Two-Dimensional Contexts](https://dl.acm.org/doi/10.1145/3159652.3159663)**<br> Alistair Moffat (The University of Melbourne); Matthias Petri (The University of Melbourne) |
| 2017 | **[Unbiased Learning-to-Rank with Biased Feedback](https://dl.acm.org/doi/10.1145/3018661.3018699)**<br> Thorsten Joachims (Cornell University); Adith Swaminathan (Cornell University); Tobias Schnabel (Cornell University)|
| 2016 | **[Beyond Ranking: Optimizing Whole-Page Presentation](http://www-personal.umich.edu/~qmei/pub/wsdm2016-ranking.pdf)**<br> Yue Wang (University of Michigan); Dawei Yin (2Yahoo Labs); Luo Jie (Snapchat); Pengyuan Wang (Yahoo Labs); Makoto Yamada (Yahoo Labs); Yi Chang (Yahoo Labs); Qiaozhu Mei (University of Michigan)|
| 2015 | **[Inverting a Steady-State](http://theory.stanford.edu/~sergei/papers/wsdm15-cset.pdf)**<br> Ravi Kumar (Google); Andrew Tomkins (Google); Sergei Vassilvitskii (Google); Erik Vee (Google) |
| 2014 | **[Scalable Hierarchical Multitask Learning Algorithms for Conversion Optimization in Display Advertising](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/42498.pdf)**<br>  Amr Ahmed (Google); Abhimanyu Das (Microsoft Research); Alex J. Smola (Carnegie Mellon University) |
| 2013 | **[Optimized Interleaving for Online Retrieval Evaluation](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/02/Radlinski_Optimized_WSDM2013.pdf.pdf)**<br> Filip Radlinski (Microsoft); Nick Craswell (Microsoft) |

<a id="recsys"></a>
## RecSys

| Year | Paper |
|:-:|:-|
| 2020 | **[Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/pdf/10.1145/3383313.3412236)**<br> Hongyan Tang (Tencent PCG); Junning Liu (Tencent PCG); Ming Zhao (Tencent PCG); Xudong Gong (Tencent PCG) |
| 2019 | **[Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/pdf/1907.06902.pdf)**<br>  Maurizio Ferrari Dacrema (Politecnico di Milano); Paolo Cremonesi (Politecnico di Milano); Dietmar Jannach (University of Klagenfurt) |
| 2018 | **[Causal Embeddings for Recommendation](https://arxiv.org/pdf/1706.07639.pdf)**<br> Stephen Bonner (Criteo AI Labs); Flavian Vasile (Criteo AI Labs) |
| 2017 | **[Modeling the Assimilation-Contrast Effects in Online Product Rating Systems: Debiasing and Recommendations]()**<br> Xiaoying Zhang (The Chinese University of Hong Kong); Junzhou Zhao (The Chinese University of Hong Kong); John C.S. Lui (The Chinese University of Hong Kong) |
| 2016 | **[Local Item-Item Models for Top-N Recommendation](https://www-users.cs.umn.edu/~chri2951/recsy368-christakopoulouA.pdf)**<br> Evangelia Christakopoulou (University of Minnesota); George Karypis (University of Minnesota) |
| 2015 | **[Context-Aware Event Recommendation in Event-based Social Networks](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.714.8211&rep=rep1&type=pdf)**<br> Augusto Q. Macedo (Federal University of Campina Grande); Leandro B. Marinho (Federal University of Campina Grande); Rodrygo L.T. Santos (Federal University of Campina Grande) |
| 2014 | **[Beyond Clicks: Dwell Time for Personalization](http://www.hongliangjie.com/publications/recsys2014.pdf)**<br> Xing Yi (Yahoo Labs); Liangjie Hong (Yahoo Labs); Erheng Zhong (Yahoo Labs); Nathan Nan Liu (Yahoo Labs); Suju Rajan (Yahoo Labs) |
| 2013 | **[A Fast Parallel SGD for Matrix Factorization in Shared Memory Systems](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf.pdf)**<br> Yong Zhuang (National Taiwan University); Wei-Sheng Chin (National Taiwan University); Yu-Chih Juan (National Taiwan University); Chih-Jen Lin (National Taiwan University) |

---

<a id="cvpr"></a>
## CVPR

The full list of CVPR best papers is presented on [this website](https://www.thecvf.com/?page_id=413#CVPRBest).

| Year | Paper |
|:-:|:-|
| 2020 | **[Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf)**<br>Shangzhe Wu (University of Oxford); Christian Rupprecht (University of Oxford); Andrea Vedaldi (Oxford University) |
| 2019 | **[A theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction](https://imaging.cs.cmu.edu/fermat_paths/assets/cvpr2019.pdf)**<br>Shumian Xin (Carnegie Mellon University); Sotiris Nousias (University of Toronto); Kiriakos N. Kutulakos (University of Toronto); Aswin C. Sankaranarayanan (Carnegie Mellon University); Srinivasa G. Narasimhan (Carnegie Mellon University); Ioannis Gkioulekas (Carnegie Mellon University) |
| 2018 | **[Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/pdf/1804.08328.pdf)**<br>Amir R. Zamir (Stanford University); Alexander Sax (Stanford University); William Shen (Stanford University); Leonidas Guibas (Stanford University); Jitendra Malik (University of California Berkeley); Silvio Savarese (Stanford University) |
| 2017 | 1. **[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)**<br>Zhuang Liu (Tsinghua University); Gao Huang (Cornell University); Laurens van der Maaten (Facebook AI Research); Kilian Q. Weinberger (Cornell University)<br>2. **[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828.pdf)**<br>Ashish Shrivastava (Apple Inc.); Tomas Pfister (Apple Inc.); Oncel Tuzel (Apple Inc.); Josh Susskind (Apple Inc.); Wenda Wang (Apple Inc.); Russ Webb (Apple Inc.) |
| 2016 | **[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)**<br>Kaiming He (Microsoft Research); Xiangyu Zhang (Microsoft Research); Shaoqing Ren (Microsoft Research); Jian Sun (Microsoft Research) |
| 2015 | **[DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time](https://rse-lab.cs.washington.edu/papers/dynamic-fusion-cvpr-2015.pdf)**<br>Richard A. Newcombe (University of Washington); Dieter Fox (University of Washington); Steven M. Seitz (University of Washington) |
| 2014 | **[What Camera Motion Reveals About Shape with Unknown BRDF](https://cseweb.ucsd.edu/~mkchandraker/pdf/cvpr14_cameramotion.pdf)**<br>Manmohan Chandraker (NEC Labs America) |
| 2013 | **[Fast, Accurate Detection of 100,000 Object Classes on a Single Machine](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Dean_Fast_Accurate_Detection_2013_CVPR_paper.pdf)**<br>Thomas Dean (Google); Mark A. Ruzon (Google); Mark Segal (Google); Jonathon Shlens (Google); Sudheendra Vijayanarasimhan (Google); Jay Yagnik (Google) |

<a id="iccv"></a>
## ICCV

The ICCV Best Paper Award is also called the Marr Prize, named after British neuroscientist [David Marr](https://en.wikipedia.org/wiki/David_Marr_(neuroscientist)). The full list of ICCV best papers is presented on [this website](https://www.thecvf.com/?page_id=413#ICCVBest).

| Year | Paper |
|:-:|:-|
| 2019 | **[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/pdf/1905.01164.pdf)**<br>Tamar Rott Shaham (Israel Institute of Technology); Tomer Michaeli (Israel Institute of Technology); Tali Dekel (Google Research) |
| 2017 | **[Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)**<br> Kaiming He (Facebook AI Research); Georgia Gkioxari (Facebook AI Research); Piotr Dollar (Facebook AI Research); Ross Girshick (Facebook AI Research) |
| 2015 | **[Deep Neural Decision Forests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)**<br>Peter Kontschieder (Microsoft Research); Madalina Fiterau (Carnegie Mellon University); Antonio Criminisi (Microsoft Research); Samuel Rota Bulò (Microsoft Research) |
| 2013 | **[From Large Scale Image Categorization to Entry-Level Categories](https://homes.cs.washington.edu/~yejin/Papers/iccv13_entrylevel.pdf)**<br>Vicente Ordonez (University of North Carolina at Chapel Hill); Jia Deng (Stanford University); Yejin Choi (Stony Brook University); Alexander Berg (University of North Carolina at Chapel Hill); Tamara Berg (University of North Carolina at Chapel Hill) |

<a id="eccv"></a>
## ECCV

The full list of ECCV best papers is presented on [this website](https://www.thecvf.com/?page_id=413#ECCVBest).

| Year | Paper |
|:-:|:-|
| 2020 | **[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)**<br> Zachary Teed (Princeton University); Jia Deng (Princeton University) |
| 2018 | **[Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection from RGB Images](https://arxiv.org/pdf/1902.01275.pdf)**<br> Martin Sundermeyer (German Aerospace Center); Zoltan-Csaba Marton (German Aerospace Center); Maximilian Durner (German Aerospace Center); Rudolph Triebel (German Aerospace Center) |
| 2016 | **[Real-Time 3D Reconstruction and 6-DoF Tracking with an Event Camera](https://www.doc.ic.ac.uk/~ajd/Publications/kim_etal_eccv2016.pdf)**<br> Hanme Kim (Imperial College London); Stefan Leutenegger (Imperial College London); Andrew J. Davison (Imperial College London) |
| 2014 | 1. **[Large-Scale Object Classification using Label Relation Graphs](https://www.cs.princeton.edu/~jiadeng/paper/deng2014large.pdf)**<br> Jia Deng (University of Michigan); Nan Ding (Google); Yangqing Jia (Google); Andrea Frome (Google); Kevin Murphy (Google); Samy Bengio (Google); Yuan Li (Google); Hartmut Neven (Google); Hartwig Adam (Google) <br>2. **[Scene Chronology](http://www.cs.cornell.edu/~snavely/publications/matzen_eccv2014.pdf)**<br> Kevin Matzen (Cornell University); Noah Snavely (Cornell University) |
---

<a id="acl"></a>
## ACL

The full list of ACL best papers is presented on [this website](https://aclweb.org/aclwiki/Best_paper_awards#ACL).

| Year | Paper |
|:-:|:-|
| 2020 | **[Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/pdf/2005.04118.pdf)**<br>Marco Tulio Ribeiro (Microsoft Research), Tongshuang Wu (University of Washington), Carlos Guestrin (University of Washington), Sameer Singh (University of Washington) |
| 2019 | **[Bridging the Gap between Training and Inference for Neural Machine Translation](https://arxiv.org/pdf/1906.02448.pdf)**<br>Wen Zhang (University of Chinese Academy of Sciences); Yang Feng (University of Chinese Academy of Sciences); Fandong Meng (WeChat AI); Di You (Worcester Polytechnic Institute); Qun Liu (Huawei Noah’s Ark Lab) |
| 2018 | **[Finding syntax in human encephalography with beam search](https://arxiv.org/pdf/1806.04127.pdf)**<br>John Hale (Cornell University); Chris Dyer (DeepMind); Adhiguna Kuncoro (University of Oxford); Jonathan R. Brennan (University of Michigan) |
| 2017 | **[Probabilistic Typology: Deep Generative Models of Vowel Inventories](https://arxiv.org/pdf/1705.01684.pdf)**<br>Ryan Cotterell (Johns Hopkins University); Jason Eisner (Johns Hopkins University) |
| 2016 | **[Finding Non-Arbitrary Form-Meaning Systematicity Using String-Metric Learning for Kernel Regression](https://www.aclweb.org/anthology/P16-1225.pdf)**<br>E. Darío Gutiérrez (University of California Berkeley); Roger Levy (Massachusetts Institute of Technology); Benjamin K. Bergen (University of California San Diego) |
| 2015 | 1. **[Improving Evaluation of Machine Translation Quality Estimation](https://www.aclweb.org/anthology/P15-1174.pdf)**<br>Yvette Graham (Trinity College Dublin)<br>2. **[Learning Dynamic Feature Selection for Fast Sequential Prediction](https://www.aclweb.org/anthology/P15-1015.pdf)**<br>Emma Strubell (University of Massachusetts Amherst); Luke Vilnis (University of Massachusetts Amherst); Kate Silverstein (University of Massachusetts Amherst); Andrew McCallum (University of Massachusetts Amherst) |
| 2014 | **[Fast and Robust Neural Network Joint Models for Statistical Machine Translation](https://www.aclweb.org/anthology/P14-1129.pdf)**<br>Jacob Devlin (Raytheon BBN Technologies); Rabih Zbib (Raytheon BBN Technologies); Zhongqiang Huang (Raytheon BBN Technologies); Thomas Lamar (Raytheon BBN Technologies); Richard Schwartz (Raytheon BBN Technologies); John Makhoul (Raytheon BBN Technologies) |
| 2013 | **[Grounded Language Learning from Video Described with Sentences](https://www.aclweb.org/anthology/P13-1006.pdf)**<br>Haonan Yu (Purdue University); Jeffrey Mark Siskind (Purdue University) |

<a id="emnlp"></a>
## EMNLP

The full list of EMNLP best papers is presented on [this website](https://aclweb.org/aclwiki/Best_paper_awards#EMNLP).

| Year | Paper |
|:-:|:-|
| 2020 | [**Digital voicing of Silent Speech**](https://arxiv.org/pdf/2010.02960.pdf)<br>David Gaddy (University of California, Berkeley); Dan Klein (University of California, Berkeley) |
| 2019 | **[Specializing Word Embeddings (for Parsing) by Information Bottleneck](https://arxiv.org/pdf/1910.00163.pdf)**<br> Xiang Lisa Li (Johns Hopkins University); Jason Eisner (Johns Hopkins University) |
| 2018 | **[Linguistically-Informed Self-Attention for Semantic Role Labeling](https://arxiv.org/pdf/1804.08199.pdf)**<br> Emma Strubell (University of Massachusetts Amherst); Patrick Verga (University of Massachusetts Amherst); Daniel Andor (Google AI Language); David Weiss (Google AI Language); Andrew McCallum (University of Massachusetts Amherst) |
| 2017 | 1. **[Depression and Self-Harm Risk Assessment in Online Forums](https://arxiv.org/pdf/1709.01848.pdf)**<br> Andrew Yates (†Max Planck Institute for Informatics); Arman Cohan (Georgetown University); Nazli Goharian (Georgetown University) <br>2. **[Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints](https://arxiv.org/pdf/1707.09457.pdf)**<br> Jieyu Zhao (University of Virginia); Tianlu Wang (University of Virginia); Mark Yatskar (University of Washington); Vicente Ordonez (University of Virginia); Kai-Wei Chang (University of Virginia) |
| 2016 | 1. **[Global Neural CCG Parsing with Optimality Guarantees](https://arxiv.org/pdf/1607.01432.pdf)**<br> Kenton Lee (University of Washington); Mike Lewis (University of Washington); Luke Zettlemoyer (University of Washington) <br>2. **[Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning](https://arxiv.org/pdf/1603.07954.pdf)**<br> Karthik Narasimhan (Massachusetts Institute of Technology); Adam Yala (Massachusetts Institute of Technology); Regina Barzilay (Massachusetts Institute of Technology) |
| 2015 | 1. **[Broad-coverage CCG Semantic Parsing with AMR](https://www.aclweb.org/anthology/D15-1198.pdf)**<br> Yoav Artzi (Cornell University); Kenton Lee (University of Washington); Luke Zettlemoyer (University of Washington) <br>2. **[Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems](https://arxiv.org/pdf/1508.01745.pdf)**<br> Tsung-Hsien Wen (Cambridge University); Milica Gasic (Cambridge University); Nikola Mrkši´c (Cambridge University); Pei-Hao Su (Cambridge University); David Vandyke (Cambridge University); Steve Young (Cambridge University) |
| 2014 | **[Modeling Biological Processes for Reading Comprehension](https://www.aclweb.org/anthology/D14-1159v2.pdf)**<br> Vivek Srikumar (Stanford University); Pei-Chun Chen (Stanford University); Abby Vander Linden (Stanford University); Brittany Harding (Stanford University); Brad Huang (Stanford University); Peter Clark (Stanford University); Christopher D. Manning (Stanford University) |
| 2013 | **[Breaking Out of Local Optima with Count Transforms and Model Recombination: A Study in Grammar Induction](https://www.aclweb.org/anthology/D13-1204.pdf)**<br> Valentin Spitkovsky (Stanford University); Hiyan Alshawi (Stanford University); Daniel Jurafsky (Stanford University) |

<a id="naacl"></a>
## NAACL

The full list of NAACL best papers is presented on [this website](https://aclweb.org/aclwiki/Best_paper_awards#NAACL).

| Year | Paper |
|:-:|:-|
| 2019 | **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423.pdf)**<br> Jacob Devlin (Google AI Language); Ming-Wei Chang (Google AI Language); Kenton Lee (Google AI Language); Kristina Toutanova (Google AI Language) |
| 2018 | **[Deep contextualized word representations](https://www.aclweb.org/anthology/N18-1202.pdf)**<br> Matthew E. Peters (Allen Institute for Artificial Intelligence); Mark Neumann (Allen Institute for Artificial Intelligence); Mohit Iyyer (Allen Institute for Artificial Intelligence); Matt Gardner (Allen Institute for Artificial Intelligence); Christopher Clark (University of Washington); Kenton Lee (University of Washington); Luke Zettlemoyer (Allen Institute for Artificial Intelligence) |
| 2016 | 1. **[Feuding Families and Former Friends; Unsupervised Learning for Dynamic Fictional Relationships](https://www.aclweb.org/anthology/N16-1180.pdf)**<br> Mohit Iyyer (University of Maryland); Anupam Guha (University of Maryland); Snigdha Chaturvedi (University of Maryland); Jordan Boyd-Graber (University of Colorado); Hal Daumé III (University of Maryland) <br>2. **[Learning to Compose Neural Networks for Question Answering](https://www.aclweb.org/anthology/N16-1181.pdf)** Jacob Andreas (University of California, Berkeley); Marcus Rohrbach (University of California, Berkeley); Trevor Darrell (University of California, Berkeley); Dan Klein (University of California, Berkeley) |
| 2015 | **[Unsupervised Morphology Induction Using Word Embeddings](https://www.aclweb.org/anthology/N15-1186.pdf)**<br> Radu Soricut (Google); Franz Och (Human Longevity) |
