<h2 align="center">
  <img src="https://raw.githubusercontent.com/pyt-team/TopoEmbedX/main/resources/logo.png" height="250">
</h2>

<h3 align="center">
    Representation Learning on Topological Domains
</h3>

<p align="center">
  <a href="#%EF%B8%8F-main-features">Main Features</a> •
  <a href="#-installing-topoembedx">Installing TopoEmbedX</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-references">References</a>
</p>


<div align="center">
   
[![Test](https://github.com/pyt-team/TopoEmbedX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoEmbedX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoEmbedX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoEmbedX/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoEmbedX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoEmbedX)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://pyt-team.github.io/topoembedx/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/pyt-team/TopoNetX?color=green)](https://github.com/pyt-team/TopoNetX/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/pyt-teamworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)

[![DOI](https://zenodo.org/badge/609414708.svg)](https://zenodo.org/badge/latestdoi/609414708)

</div>



![topoembedx](https://user-images.githubusercontent.com/8267869/234074436-402ac931-2dc9-43da-a056-6c927f613242.png)

Many natural systems as diverse as social networks and proteins are characterized by _relational structure_. This is the structure of interactions between components in the system, such as social interactions between individuals or electrostatic interactions between atoms.

How can we conveniently represent data defined on such relational systems?

`TopoEmbedX` (TEX) is a package for representation learning on topological domains, the mathematical structures of relational systems.


## 🛠️ Main Features

Support of higher order representation learning algorithms such as:
- DeepCell,
- Cell2Vec,
- Higher Order Laplacian Eigenmaps, and
- Higher Order Geometric Laplacian Eigenmaps

for the topological domains supported in [TopoNetX](https://github.com/pyt-team/TopoNetX).


## 🤖 Installing TopoEmbedX


We recommend using Python 3.11, which is a python version used to run the unit-tests.

For example, create a conda environment:
   ```bash
   conda create -n tex python=3.11.3
   conda activate tex
   ```

Then:

1. Clone a copy of `TopoEmbedX` from source:
```bash
git clone https://github.com/pyt-team/TopoEmbedX
cd TopoEmbedX
```
2. If you have already cloned `TopoEmbedX` from source, update it:
```bash
git pull
```
3. Install `TopoEmbedX` in editable mode:
```bash
pip install -e '.[all]'
```
4. Install pre-commit hooks:
```bash
pre-commit install
```


## 🦾 Getting Started

```ruby
import topoembedx as tex
import toponetx as tnx

# create a cell complex object with a few cells
cc = tnx.classes.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]],ranks=2)

# create a model

model = tex.Cell2Vec()

# fit the model

model.fit(cc, neighborhood_type="adj", neighborhood_dim={"rank": 1, "via_rank": -1})
# here neighborhood_dim={"rank": 1, "via_rank": -1} specifies the dimension for
# which the cell embeddings are going to be computed.
# rank=1 means that the embeddings will be computed for the first dimension.
# The integer 'via_rank' is ignored and only considered
# when the input complex is a combinatorial complex or colored hypergraph.


# get the embeddings:

embeddings = model.get_embedding()

```

## 🔍 References ##

To learn more about topological representation learning.

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub. [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606).
```
@misc{hajij2023topological,
      title={Topological Deep Learning: Going Beyond Graph Data},
      author={Mustafa Hajij and Ghada Zamzmi and Theodore Papamarkou and Nina Miolane and Aldo Guzmán-Sáenz and Karthikeyan Natesan Ramamurthy and Tolga Birdal and Tamal K. Dey and Soham Mukherjee and Shreyas N. Samaga and Neal Livesay and Robin Walters and Paul Rosen and Michael T. Schaub},
      year={2023},
      eprint={2206.00606},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Figure from:
- Mathilde Papillon, Sophia Sanborn, Mustafa Hajij, Nina Miolane. [Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.](https://arxiv.org/pdf/2304.10031.pdf)
```
@misc{papillon2023architectures,
      title={Architectures of Topological Deep Learning: A Survey on Topological Neural Networks},
      author={Mathilde Papillon and Sophia Sanborn and Mustafa Hajij and Nina Miolane},
      year={2023},
      eprint={2304.10031},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Funding

<img align="right" width="200" src="https://raw.githubusercontent.com/pyt-team/TopoNetX/main/resources/erc_logo.png">

Partially funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

Partially funded by the National Science Foundation (DMS-2134231, DMS-2134241).

