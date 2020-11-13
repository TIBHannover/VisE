# Ontology-driven Event Type Classification in Images

This is the official GitHub page for the paper:

> Eric Müller-Budack, Matthias Springstein, Sherzod Hakimov, Kevin Mrutzek, and Ralph Ewerth:
"Ontology-driven Event Type Classification in Images".
In: *IEEE Winter Conference on Applications of Computer Vision (WACV)*, IEEE, 2021.

The paper is available on *arXiv*: https://arxiv.org/pdf/2011.04714.pdf

Further information can be found on the **EventKG** website: http://eventkg.l3s.uni-hannover.de/VisE


## Content

- [Ontology-driven Event Type Classification in Images](#ontology-driven-event-type-classification-in-images)
  - [Content](#content)
  - [Setup](#setup)
    - [Setup with Singularity (for Reproducibility)](#setup-with-singularity-for-reproducibility)
    - [Setup with Virtual Environment](#setup-with-virtual-environment)
    - [Setup with Docker](#setup-with-docker)
  - [Download Ontology, Dataset and Models](#download-ontology-dataset-and-models)
  - [Models](#models)
  - [Inference](#inference)
  - [Test](#test)
  - [VisE-D: Visual Event Classification Dataset](#vise-d-visual-event-classification-dataset)
  - [VisE-O: Visual Event Ontology](#vise-o-visual-event-ontology)
  - [Benchmark Ontologies](#benchmark-ontologies)
  - [Supplemental Material](#supplemental-material)
  - [LICENSE](#license)


## Setup

We provide three different ways to setup the project. The results can be reproduced using the [setup with singularity](#setup-with-singularity-for-reproducibility). The singularity image is built with an optimized pytorch implementation on arch linux, which we used for training and testing.
                                                                                                                            
While the other two setups using a [virtual environment](#setup-with-virtual-environment) or [docker](#setup-with-docker) produce the same result on our testsets, they slightly differ from the results reported in the paper (deviation around 0.1%).


### Setup with Singularity (for Reproducibility)

To install singularity please follow the instructions on: https://sylabs.io/guides/3.6/admin-guide/installation.html

Download our singularity image from: [link](https://tib.eu/cloud/s/fPMwLMWo3wCmMRy/download)

To run code using sinularity, please run: 

```shell script
singularity exec \
  -B </PATH/TO/REPOSITORY>:/src \
  --nv </PATH/TO/SINGULARITY/IMAGE>.sif \
  bash

cd /src
```


### Setup with Virtual Environment

Please run the following command to setup the project in your (virtual) environment:

```shell script
pip install -r requirements.txt
```

**NOTE:** This setup produces slightly different results (deviation around 0.1%) while [testing](#test). To fully reproduce our results we have provided a [singularity image](#setup-with-singularity-for-reproducibility), which is a copy of our training and testing environment and uses a highly optimized pytorch implementation.


### Setup with Docker

We have provided a Docker container to execute our code. You can build the container with:

```shell script
docker build <PATH/TO/REPOSITORY> -t <DOCKER_NAME>
```

To run the container please use:

```shell script
docker run \
  --volume <PATH/TO/REPOSITORY>:/src \
  --shm-size=256m \
  -u $(id -u):$(id -g) \
  -it <DOCKER_NAME> bash 

cd /src
```

**NOTE:** This setup produces slightly different results (deviation around 0.1%) while [testing](#test). To fully reproduce our results we have provided a [singularity image](#setup-with-singularity-for-reproducibility), which is a copy of our training and testing environment and uses a highly optimized pytorch implementation.


## Download Ontology, Dataset and Models

You can automatically download the files (ontologies, models, etc.) that are required for [inference](#inference) and [test](#test) with the following command:

```shell script
python download_resources.py
```

The files will be stored in a folder called ```resources/``` relative to the repository path.


## Models

We provide the trained models for the following approaches:

- Classification baseline (denoted as ```C```): [link](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/727c3ee1-4107-4996-878d-1caf537730e8/download/vise_c.tar.gz)
- Best ontology driven approach using the cross-entropy loss (denoted as ```CO_cel```): [link](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/7c672f2b-f45e-40aa-b6bb-01fb2e9bf5e7/download/vise_co_cel.tar.gz)
- Best ontology driven approach using the cross-entropy loss (denoted as ```CO_cos```): [link](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/b105c1aa-3bc4-4233-8103-8f4616948d85/download/vise_co_cos.tar.gz)

The performance of these models regarding the top-k accuracy, jaccard similarity coefficient (JSC), and cosine similarity (CS) on the *VisE-Bing* and *VisE-Wiki* testsets is listed below using the provided [singularity image](#setup-with-singularity-for-reproducibility):

\
**VisE-Bing**
| Model  |  Top-1   |  Top-3   |  Top-5   |   JSC    |    CS    |
| :----- | :------: | :------: | :------: | :------: | :------: |
| C      |   77.4   |   89.8   |   93.6   |   84.7   |   87.7   |
| CO_cel |   81.5   | **91.8** | **94.3** |   87.5   |   90.0   |
| CO_cos | **81.9** |   90.8   |   93.2   | **87.9** | **90.4** |

\
**VisE-Wiki**
| Model  |  Top-1   |  Top-3   |  Top-5   |   JSC    |    CS    |
| :----- | :------: | :------: | :------: | :------: | :------: |
| C      |   61.7   |   74.6   | **79.2** |   72.7   |   77.8   |
| CO_cel |   63.4   | **74.7** |   78.8   |   73.9   |   78.7   |
| CO_cos | **63.5** |   74.3   |   78.8   | **74.1** | **79.0** |


## Inference

In order to apply our [models](#models) on an image or a list of images, please execute the following command:

```shell script
python infer.py -c </path/to/model.yml> -i </path/to/image(s)>
```

If you followed the instructions in [Download Ontology, Dataset and Models](#download-ontology-dataset-and-models) the model config is placed in ```resources/VisE-D/models/<modelname>.yml``` relative to the repository path.

**Optional parameters:**
As standard parameters the batch size is set to 32, the top-5 predictions will be shown, and the multiplied values of the leaf node probability and subgraph cosine similarity are used to convert the subgraph vector to a leaf node vector (details are presented in Section 4.2.3 of the paper).


```--batch_size <int>``` specifies the batch size (default ```16```)

```--num_predictions <int>``` sets the number of top predictions printed on the console (default ```3```) 

```--s2l_strategy [leafprob, cossim, leafprob*cossim]``` specifies the strategies to retrieve the leaf node vector from a subgraph vector (default ```leafprob*cossim```) 


## Test

This step requires to download the test images in the *VisE-Bing* or *VisE-Wiki* dataset. You can run the following command to automatically download the images:

```shell script
python download_images.py -d </path/to/dataset.jsonl> -o </path/to/output/root_directory/>
```

If you followed the instructions in [Download Ontology, Dataset and Models](#download-ontology-dataset-and-models) the dataset is placed in ```resources/VisE-D/<datasetname>.jsonl``` and model config is placed in ```resources/VisE-D/models/<modelname>.yml``` relative to the repository path.


**Optional parameters:**

```-t <int>``` sets the number of parallel threads (default ```32```)

```-r <int>``` sets the number of retries to download an image (default ```5```)

```--max_img_dim <int>``` sets the dimension of the longer image dimension (default ```512```)

**NOTE:** This step also allows to download the training and validation images in case you want to build your own models.


\
After downloading the test images you can calculate the results using the following command:

```shell script
python test.py \
    -c </path/to/model.yml> \
    -i </path/to/image/root_directory> \
    -t </path/to/testset.jsonl>
    -o </path/to/output.json>
```

**Optional parameters:**
As standard parameters the batch size is set to 32 and the multiplied values of the leaf node probability and subgraph cosine similarity are used to convert the subgraph vector to a leaf node vector (details are presented in Section 4.2.3 of the paper).

```--batch_size <int>``` specifies the batch size (default ```16```)

```--s2l_strategy [leafprob, cossim, leafprob*cossim]``` specifies the strategies to retrieve the leaf node vector from a subgraph vector (default ```leafprob*cossim```) 


## VisE-D: Visual Event Classification Dataset

The *Visual Event Classification Dataset (VisE-D)* is available on: https://data.uni-hannover.de/de/dataset/vise

You can automatically download the dataset by following the instructions in [Download Ontology, Dataset and Models](#download-ontology-dataset-and-models). To download the images from the provided URLs, please run the following command:

```shell script
python download_images.py -d </path/to/dataset.jsonl> -o </path/to/output/root_directory/>
```

Optional parameters:

```-t <int>``` sets the number of parallel threads (default ```32```)

```-r <int>``` sets the number of retries to download an image (default ```5```)

```--max_img_dim <int>``` sets the dimension of the longer image dimension (default ```512```)


## VisE-O: Visual Event Ontology

In Section 3.2 of the paper, we have presented several methods to create an *Ontology* for newsworthy event types. Statistics are presented in Table 1 of the paper. 

Different versions of the *Visual Event Ontology (VisE-O)* can be downloaded here: [link](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/99ce7e4d-df5b-40f6-afb4-16085dbf697d/download/vise-d.tar.gz)

Furthermore you can explore the *Ontologies* using the following links:

- **Initial Ontology** (result of Section 3.2.2): [explore](https://tibhannover.github.io/VisE/VisE-O_initial/index.html)
- **Disambiguated Ontology** (result of Section 3.2.3): [explore](https://tibhannover.github.io/VisE/VisE-O_disambiguated/index.html)
- **Refined Ontology** (result of Section 3.2.4): [explore](https://tibhannover.github.io/VisE/VisE-O_refined/index.html)

**USAGE:** After opening an *Ontology*, the *Leaf Event Nodes* (blue), *Branch Event Nodes* (orange), and *Root Node* (yellow) as well as their *Relations* are displayed. By clicking on a specific *Event Node* additional information such as the *Wikidata ID* and related child (*Incoming*) and parent (*Outgoing*) nodes are shown. In addition, the search bar can be used to directly access a specific *Event Node*.


## Benchmark Ontologies

In order to evaluate the presented ontology-diven approach on other benchmark datasets, we have manually linked classes of the *Web Images for Event Recognition (WIDER)*, *Social Event Dataset (SocEID)*, and the *Rare Event Dataset (RED)* to the *Wikidata* knowledge base according to Section 5.3.3. The resulting *Ontologies* for these datasets can be downloaded and explored here:
- **WIDER Ontology**: [download](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/b1c2f92b-4b69-46fc-9282-16acc7a1c9aa/download/wider.tar.gz) | [explore](https://tibhannover.github.io/VisE/WIDER/index.html)
- **SocEID Ontology**: [download](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/a8373c98-32a8-408c-b8e9-51e6b1e01777/download/soceid.tar.gz) | [explore](https://tibhannover.github.io/VisE/SocEID/index.html)
- **RED Ontology**: [download](https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/d0f5cd8b-7c3e-4055-9810-f9cba2b69a33/download/red.tar.gz) | [explore](https://tibhannover.github.io/VisE/RED/index.html)

## Supplemental Material

Detailed information on the sampling strategy to gather event images, statistics for the training and testing datasets presented in Section 3.3, and results using different inference strategies (Section 4.2.3) are available in the [vise_supplemental.pdf](vise_supplemental.pdf). 


## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the LICENSE file in the repository.