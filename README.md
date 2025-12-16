<img src="logo/logo.png" width="500" title="HGX logo">

:page_facing_up: **[Paper](https://arxiv.org/pdf/2303.15356.pdf)** | :paperclip:
*[Docs](https://hypergraphx.readthedocs.io/en/latest/#)* | :computer: 
*[Tutorials](https://github.com/HGX-Team/hypergraphx/tree/master/tutorials)* | :floppy_disk: 
*[Data](https://github.com/HGX-Team/data)* | :bug: [Report bug](https://github.com/HGX-Team/hypergraphx/issues)
| :email: **[Reach us](mailto:quintino.lotito@unitn.it)**
-----


**HGX** is a Python library for the analysis of real-world complex systems with **group interactions** and provides a
comprehensive suite of tools and algorithms for constructing, visualizing, and analyzing **hypergraphs**. <br> The
library is designed to be user-friendly and accessible, with a wide range of functionalities that can be applied to a
diverse set of applications and use cases.

## Disclaimer

* We welcome early feedback, discussions, ideas and contributions.

## News

* 2023-03-28: HGX is now available!

## Table of contents

- [What are higher-order networks?](#what-are-higher-order-networks)
- [What is hypergraphx?](#what-is-hypergraphx)
- [Library highlights](#library-highlights)
- [Quick start](#quick-start)
    * [Installation](#installation)
- [Tutorials](#tutorials)
- [Data](#data)
- [The HGX team](#the-hgx-team)
    * [Project coordinators](#project-coordinators)
    * [Core members](#core-members)
    * [Collaborators](#collaborators)
- [Contributing](#contributing)
- [Citing HGX](#citing-hgx)
- [License](#license)
- [Other resources](#other-resources)

## What are higher-order networks?

In the last few decades, networks have emerged as the natural tool to model a wide variety of natural, social and
man-made systems.
Networks, collections of nodes and links connecting pairs of them, are able to capture dyadic interactions only.
However, in many real-world systems units interact in groups of three or more. Systems with non-dyadic interactions are
ubiquitous, with examples ranging from cellular networks, drug recombination, structural and functional brain networks,
human and animal face-to-face interactions, and collaboration networks. These higher-order interactions can be naturally
described by alternative mathematical structures such as hypergraphs, where hyperedges connect groups of nodes of
arbitrary size.

<img src="images/hypergraph.png" width="350" title="Hypergraph example">

## What is hypergraphx?
HGX aims to provide, as a single source, a comprehensive suite of tools and algorithms for constructing, storing, analysing and visualizing systems with
higher-order interactions.
These include different ways to convert data across distinct higher-order representations, a large variety of measures
of higher-order organization at the local and the mesoscale, statistical filters to sparsify higher-order data, a wide
array of static and dynamic generative models, an implementation of different dynamical processes, from epidemics to
diffusion and synchronization, with higher-order interactions, and more.
Our computational framework is general, and allows to analyse hypergraphs with weighted, directed, signed, temporal and
multiplex group interactions.
Beyond experts in the field, we hope that our library will make higher-order network analysis accessible to everyone
interested in exploring the higher-order dimension of relational data.

## Library highlights

* HGX allows to store **higher-order data** as **hypergraphs** and to **convert** them to bipartite networks, maximal
  simplicial complexes, higher-order line graphs, dual hypergraphs, and clique-expansion graphs.
* It provides simple tools to characterize basic **node and hyperedge statistics**, such as hyperdegree distributions,
  correlations and assortativity, at the level of the whole higher-order network and of the individual nodes.
* Our library provides a variety of **higher-order centrality measures** for nodes and hyperedges, based on
  participation in different subhypergraphs, on spectral approaches, and on shortest paths and betweenness flows.
* HGX implements **higher-order motif analysis**. It also provides an approximated algorithm for motif analysis based on
  hyperedge sampling, able to speed up computations by orders of magnitudes with only a minimal compromise in accuracy.
* Our library provides spectral method to recover **hard communities**, generative models to extract **overlapping
  communities** and to **infer hyperedges**, methods to capture **assortative and disassortative community structure**
  and **core-periphery organization** in higher-order systems.
* We implement a variety of different tools to **filter** the most informative higher-order interactions, based on
  extracting statistically validated hypergraphs and identifying significant maximally interacting node groups.
* HGX offers a **synthetic hypergraph samplers library**, implementing various models such as Erdős-Rényi, scale-free,
  configuration and community-based models. It also includes a higher-order activity-driven model for temporal group
  interactions.
* We provide functions to simulate and analyze several **dynamical process on higher-order networks**, including
  synchronization, social contagion and random walks.
* HGX is highly flexible. It allows to store and analyze hypergraphs with a **rich set of features associated with
  hyperedges**, including interactions of different intensity, directions, sign, that vary in time or belong to
  different layers of a multiplex system.
* Our library provides different **visualization tools** to gain visual insights into the higher-order organization of
  real-world systems.

## Quick start

### Installation

```
pip install hypergraphx
```

or, if you really want the latest updates

```
pip install hypergraphx@git+https://github.com/HGX-Team/hypergraphx
```

## Tutorials

You can find tutorials covering a variety of use
cases [here](https://github.com/HGX-Team/hypergraphx/tree/master/tutorials).

## Data

Higher-order datasets are available in our [data repository](https://github.com/HGX-Team/data).

## Citing HGX

If you use HGX or related data in your paper, please cite:

```
@article{lotito2023hypergraphx,
    author = {Lotito, Quintino Francesco and Contisciani, Martina and De Bacco, Caterina and Di Gaetano, Leonardo and Gallo, Luca and Montresor, Alberto and Musciotto, Federico and Ruggeri, Nicolò and Battiston, Federico},
    title = "{Hypergraphx: a library for higher-order network analysis}",
    journal = {Journal of Complex Networks},
    volume = {11},
    number = {3},
    year = {2023},
    month = {05},
    issn = {2051-1329},
    doi = {10.1093/comnet/cnad019},
    url = {https://doi.org/10.1093/comnet/cnad019},
    note = {cnad019},
    eprint = {https://academic.oup.com/comnet/article-pdf/11/3/cnad019/50461094/cnad019.pdf},
}
```

## The HGX team

### Project coordinators

* Quintino Francesco Lotito (lead developer) -
  quintino.lotito@unitn.it - <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/FraLotito"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/FraLotito"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Federico Battiston (project coordinator) -
  battistonf@ceu.edu <a title="Icons8, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://people.ceu.edu/federico_battiston"><img width="16" alt="Icons8 flat link" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Icons8_flat_link.svg/512px-Icons8_flat_link.svg.png"></a> <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/fede7j"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/fede7j"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>

### Core members

* Martina Contisciani -
  martina.contisciani@tuebingen.mpg.de <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/mcontisc"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/mcontisc"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Caterina De Bacco -
  caterina.debacco@tuebingen.mpg.de <a title="Icons8, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://www.cdebacco.com/"><img width="16" alt="Icons8 flat link" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Icons8_flat_link.svg/512px-Icons8_flat_link.svg.png"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/cdebacco"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Leonardo Di Gaetano -
  leonardo.digaetano.96@gmail.com <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/leodigaetano"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/LeonardoDiGaetano"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Luca Gallo -
  luca.gallo@uni-corvinus.hu - <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/l_gajo"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/lgajo"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Alberto Montresor -
  alberto.montresor@unitn.it <a title="Icons8, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="http://cricca.disi.unitn.it/montresor/"><img width="16" alt="Icons8 flat link" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Icons8_flat_link.svg/512px-Icons8_flat_link.svg.png"></a>
* Federico Musciotto -
  federico.musciotto@unipa.it <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/musci8"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/musci8"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Nicolò Ruggeri -
  nicolo.ruggeri@tuebingen.mpg.de <a title="Icons8, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://nickruggeri.github.io/"><img width="16" alt="Icons8 flat link" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Icons8_flat_link.svg/512px-Icons8_flat_link.svg.png"></a> <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/NikRuggeri"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/nickruggeri"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>

### Collaborators

* Lorenzo
  Betti <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://x.com/LoreBetti"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/Loreb92"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Alberto
  Ceria <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://twitter.com/cerialbo"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a>
* Davide Colosimo
* Helcio
  Felippe <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://x.com/juniorfelippe"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a>
* Alec
  Kirkley <a title="Icons8, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://aleckirkley.com/"><img width="16" alt="Icons8 flat link" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Icons8_flat_link.svg/512px-Icons8_flat_link.svg.png"></a> <a title="Twitter, Apache License 2.0 &lt;http://www.apache.org/licenses/LICENSE-2.0&gt;, via Wikimedia Commons" href="https://x.com/captainkirk1041"><img width="16" alt="Twitter-logo" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"></a> <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/aleckirkley"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Berné
  Nortier <a title="GitHub, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://github.com/joanne-b-nortier"><img width="16" alt="Octicons-mark-github" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/512px-Octicons-mark-github.svg.png"></a>
* Alberto Vendramini

## Contributing

HGX is a collaborative project and we welcome suggestions and contributions. If you are interested in contributing to
HGX or have any questions about our project, please do not hesitate to reach out to us.

:running: **I only have 1 minute**

- Tweet about our library and spread the voice!
- Give the project a star on GitHub :star:!

:hourglass_flowing_sand: **I've got 10 minutes**

- Are you interested in higher-order motif analysis or community detection in hypergraphs? Try out
  our [tutorials](https://github.com/HGX-Team/hypergraphx/tree/master/tutorials)!
- [Suggest](https://github.com/HGX-Team/hypergraphx/issues) ideas and engage in discussions
- [Help](https://github.com/HGX-Team/hypergraphx/issues) someone with a problem
- [Report a bug](https://github.com/HGX-Team/hypergraphx/issues) someone with a problem

:computer: **I've got a few hours to work on this**

- Create new tools for the community
- Help solving bugs reported in the [issues](https://github.com/HGX-Team/hypergraphx/issues)
- Please read the more detailed [contributing guidelines](CONTRIBUTING.md)

:tada: **I want to help grow the community**

- Spread the voice!
- Talk about how HGX has been useful for your research problem
- Engage in a discussion with the core members of the library

## License

Released under the 3-Clause BSD license (
see [LICENSE.md](https://github.com/HGX-Team/hypergraphx/blob/master/LICENSE.md))

HGX contains copied or modified code from third sources. The licenses of such code sources can be found in
our [license file](https://github.com/HGX-Team/hypergraphx/blob/master/LICENSE.md)

## Other resources

#### Python

- [ASH](https://github.com/GiulioRossetti/ASH)
- [HyperNetX](https://github.com/pnnl/HyperNetX)
- [Reticula](https://github.com/reticula-network/reticula)
- [XGI](https://github.com/ComplexGroupInteractions/xgi)

#### Julia

- [HyperGraphs.jl](https://github.com/lpmdiaz/HyperGraphs.jl)
- [SimpleHypergraphs.jl](https://github.com/pszufe/SimpleHypergraphs.jl)

#### R

- [hyperG](https://cran.r-project.org/web/packages/HyperG/index.html)

## Acknowledgments

This project is supported by the Air Force Office of Scientific Research under award number FA8655-22-1-7025.



