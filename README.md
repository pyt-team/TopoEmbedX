
[![Lint](https://github.com/pyt-team/TopoEmbedX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoEmbedX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoEmbedX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoEmbedX/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoEmbedX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoEmbedX)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/609414708.svg)](https://zenodo.org/badge/latestdoi/609414708)


![pyt](https://github.com/mhajij/shrec_16/blob/main/logo.png)

![toponetx](https://user-images.githubusercontent.com/8267869/234068354-af9480f1-1d18-4914-92f1-916d9093e44d.png)

# 🌐 TopoEmbedX (TEX) 🍩
# Representation Learning on Topological Domains

![topoembedx](https://user-images.githubusercontent.com/8267869/234074436-402ac931-2dc9-43da-a056-6c927f613242.png)

Many natural systems as diverse as social networks and proteins are characterized by _relational structure_. This is the structure of interactions between components in the system, such as social interactions between individuals or electrostatic interactions between atoms.

How can we conveniently represent data defined on such relational systems?

`TopoEmbedX` (TEX) is a package for representation learning on topological domains, the mathematical structures of relational systems.


## 🛠️ Main Features in Version 1.0

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
pip install -e .[all]
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
cc = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]],ranks=2)

# create a model

model = tex.Cell2Vec()

# fit the model

model.fit(cc, neighborhood_type="adj", neighborhood_dim={"r": 1, "k": -1})
# here neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
# which the cell embeddings are going to be computed.
# r=1 means that the embeddings will be computed for the first dimension.
# The integer 'k' is ignored and only considered
# when the input complex is a combinatorial complex.


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
