# H-Space similarity explorer

[click here to open the tool](https://jonasloos.github.io/h-space-similarity-explorer/)

This is a simple tool to explore the similarity between the representations of different concepts at different spatial positions in the h-space of diffusion models.

![image](https://github.com/JonasLoos/h-space-similarity-explorer/assets/33965649/bf895921-2616-43c7-86f4-39d5d570d21c)

## Generating custom images for the similarity explorer

Just edit the prompts in the [generate-reprs.ipynb](generate-reprs.ipynb) notebook and run it. A folder called `representations` will be generated with the h-space representations of the concepts you chose.

Then start a simple local webserver, e.g. with `python -m http.server` and open [localhost:8000](http://localhost:8000) in your browser.

The representations used on the website are stored in a [separate repo](https://github.com/JonasLoos/h-space-similarity-explorer-data).
