
TopoEmbed
=========


TopoEmbed is a package for studying higher order repreesnation learning on simplicial, cellular and combinatorial complexes




New Features of Version 1.0
---------------------------

1.Support of higher order represenation learning algorithsm such as DeepCell, Cell2Vec, Higher Order Laplacian Eigenmaps and Higher Order Geometric Laplacian Eigenmaps for various complexes supported in TopoNetX (simplicial, cellular, combinatorial).



# Installing TopoEmbedX

1. Clone a copy of `TopoEmbedX` from source:
```bash
git clone https://github.com/pyt-team/TopoEmbedX
cd topoembedc
```
2. If you have already cloned `TopoEmbedX` from source, update it:
```bash
git pull
```
3. Install `TopoEmbedX` in editable mode:
```bash
pip install -e ".[dev,full]"
```
4. Install pre-commit hooks:
```bash
pre-commit install
```



## Getting Started : simplicia/cellular/combinatorial representation learning

```ruby
import topoembedx as tex
import toponetx as tnx

# create a cell complex object with a few cells
cx = tnx.CellComplex([[1, 2, 3, 4], [3,4,5,6,7,8]],ranks=2)

# create a model

model = tex.Cell2Vec()

# fit the model

model.fit(cx,neighborhood_type="adj", neighborhood_dim={"r": 1, "k": -1})
# here neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
# which the cell embeddings are going to be computed.
# r=1 means that the embeddings will be computed for the first dimension.
# The integer 'k' is ignored and only considered
# when the input complex is a combinatorial complex.


# get the embeddings:

embeddings = model.get_embedding()

```



