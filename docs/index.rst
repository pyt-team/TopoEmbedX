🌐 TopoEmbedX (TEX) 🍩
=======================

`TopoEmbedX` (TEX) is a Python package for representation Learning on Topological Domains. Topological domains are the natural mathematical structures representing relations between the components of a dataset.

.. figure:: https://user-images.githubusercontent.com/8267869/234074436-402ac931-2dc9-43da-a056-6c927f613242.png
   :alt: topoembedx
   :class: with-shadow
   :width: 1000px

Many natural systems as diverse as social networks and proteins are characterized by *relational structure*. This is the structure of interactions between components in the system, such as social interactions between individuals or electrostatic interactions between atoms. How can we conveniently represent data defined on such relational systems? `TopoEmbedX` (TEX) is a package for representation learning on topological domains, the mathematical structures of relational systems.


🛠️ Main Features
----------------

Support of higher order representation learning algorithms such as:

- DeepCell,
- Cell2Vec,
- Higher Order Laplacian Eigenmaps, and
- Higher Order Geometric Laplacian Eigenmaps,

for the topological domains supported in `TopoNetX <https://github.com/pyt-team/TopoNetX>`__.

🔍 References
-------------

To learn more about topological representation learning:

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub. `Topological Deep Learning: Going Beyond Graph Data <https://arxiv.org/abs/2206.00606>`__.

.. code-block:: BibTeX

   @misc{hajij2023topological,
         title={Topological Deep Learning: Going Beyond Graph Data},
         author={Mustafa Hajij and Ghada Zamzmi and Theodore Papamarkou and Nina Miolane and Aldo Guzmán-Sáenz and Karthikeyan Natesan Ramamurthy and Tolga Birdal and Tamal K. Dey and Soham Mukherjee and Shreyas N. Samaga and Neal Livesay and Robin Walters and Paul Rosen and Michael T. Schaub},
         year={2023},
         eprint={2206.00606},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }

For a literature review of topological deep learning:

- Mathilde Papillon, Sophia Sanborn, Mustafa Hajij, Nina Miolane. `Architectures of Topological Deep Learning: A Survey on Topological Neural Networks. <https://arxiv.org/pdf/2304.10031.pdf>`__

.. code-block:: BibTeX

   @misc{papillon2023architectures,
         title={Architectures of Topological Deep Learning: A Survey on Topological Neural Networks},
         author={Mathilde Papillon and Sophia Sanborn and Mustafa Hajij and Nina Miolane},
         year={2023},
         eprint={2304.10031},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }

.. toctree::
   :maxdepth: 1
   :hidden:

   api/index
   tutorials/index
   contributing/index

