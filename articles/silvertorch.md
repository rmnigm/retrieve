[^1]
[^2]

<div class="CCSXML">

\<ccs2012\>
\<concept\>
\<concept_id\>10002951.10003317.10003338.10010403\</concept_id\>
\<concept_desc\>Information systems Novelty in information retrieval\</concept_desc\>
\<concept_significance\>500\</concept_significance\>
\</concept\>
\</ccs2012\>

</div>

# Introduction

Serving embedding-based Deep Learning recommendation models (DLRM ) at scale is challenging since it is impossible to rank all items during inference. A multi-stage design is widely adopted. First, the retrieval stage narrows the item candidates to thousand scale by formulating the task as an Approximate Nearest Neighbor (ANN) search problem in vector space, identifying relevant items based on vector similarities. This is commonly built using libraries like Faiss , RAFT  or a dedicated vector database system like Milvus . Meanwhile, the retrieval stage applies feature filtering to match user attributes in multiple aspects. These filters are implemented using inverted-index mechanism. Retrieval therefore relies on the indexing and filtering services during online serving. Finally, the retrieved items are passed to downstream ranking models to generate recommendation results.

Despite wide adoption, the way of indexing and filtering in retrieval has drawbacks. First, authoring divergence of different serving components in retrieval slows down the end-to-end development, experiments and deployment. Second, isolated optimization in each component lacks co-design. Each system needs to implement redundant logic such as versioning, scheduling and batching, which makes the overall optimization fragmented . Third, the client needs to compose multiple requests for different services and orchestrate intermediate results. This increases retrieval latency due to unnecessary data movement and transformation.
In this paper, we propose SilverTorch, a model-based retrieval system built on PyTorch. Instead of building standalone indexing services, SilverTorch defines all serving components such as ANN search and feature filtering as layers of the served model itself. Based on the unified stack, we co-design the ANN search with the feature filtering and propose an index algorithm. SilverTorch’s in-model design also provides unified interface that simplifies orchestration. Client sends a single retrieval request to the model runtime that serves a SilverTorch model.

Another major challenge for serving large-scale recommendation models is compute scalability. As candidate pool grows, retrieval models need to build larger index. Meanwhile, model architectures are becoming more complex.
Recent retrieval approaches adopt complex learned structures to replace the dot-product to represent similarities  . There are also studies incorporating raw historical interaction data using transformers . Unfortunately, existing solutions are struggling to satisfy these increasing computational demand. Existing systems adopt CPU-based ANN indexing and feature filtering . To the best of our knowledge, no existing work studies feature filtering on GPUs. For ANN search, there are libraries implement ANN algorithms on GPUs . However, they only support limited topk and difficult to customize for recommendation. CPU-based ANN search and filtering can scale out by adding more CPU servers to partition the index, but the cost increases linearly which quickly becomes inefficient.
SilverTorch fully utilizes GPUs, introducing model-based ANN search and bloom index filtering designed for retrieval. Both ANN search and feature filtering processes during inference are unified as tensor computation, consistent with other model serving components. We propose a co-designed indexing for ANN search and feature filtering on GPUs to further reduce GPU memory utilization and eliminate unnecessary computation. We demonstrate that SilverTorch’s GPU solution is more cost-efficient.

Benefiting from SilverTorch’s in-model design, we extend retrieval models by introducing an OverArch scoring layer to further learn user-item interactions. The initial returned items from ANN and filtering are re-ranked by the OverArch. For the OverArch, we pre-compute item embeddings and cache them into GPU memory to reduce online computation cost. Additionally, we enable multi-task retrieval with a aggregation layer (referred to as Value Model) to compute the overall score across multi-tasks. These initiatives enable retrieval model to pre-rank more items in the retrieval stage and improve the model consistency between retrieval and ranking stages. For ANN, we leverage Int8 quantization to save compute for pre-ranking more items in OverArch.
We summarize our contributions as follows:

- We propose SilverTorch, a unified model-based system for serving large-scale recommendation models. It unifies the serving development within PyTorch, re-defines the standalone ANN and feature filtering services with in-model tensor operators, simplifies client-side orchestration.

- We introduce a novel GPU-based bloom index algorithm for feature filtering and build a fused Int8 ANN kernel on GPUs. We further propose a co-designed index algorithm combining ANN search with feature filtering. This largely reduces memory utilization and eliminate unnecessary computation. To the best of our knowledge, bloom index is the first attempt to conduct feature filtering on GPUs and the first study applying it to recommendation systems.

- We extend existing retrieval models by introducing an OverArch scoring layer as learned similarities, as well as a multi-task retrieval with Value Model in SilverTorch. The item embeddings used for the OverArch are pre-computed and cached into SilverTorch model. These functionalities improve recommendation accuracy and enable more advanced retrieval model architectures.

- We evaluate SilverTorch on industry-scale datasets with a 10-million items and a 80-million items respectively. The results show SilverTorch improves serving throughput by up to $`23.7\times`$ compared to the state-of-the-art baselines. By introducing OverArch scoring layers and multi-task retrieval, SilverTorch achieves over 5.6% recall improvement and is $`13.35\times`$ cost efficient.

# Background and Related Work

<figure id="fig:st_prototype">
<img src="figures/background/st_prototype.png" style="width:50.0%" />
<figcaption>A simplified SilverTorch program in PyTorch.</figcaption>
</figure>

Modern recommendation systems widely adopt the two-tower architecture  to represent user and item features in vector space. Features for recommendation models can be categorized into dense (numerical) and sparse (categorical). User and item features contain both types, with contextual features (e.g., timestamp) bundled with user features. The embedding in our paper refers to the vector representation generated from either the user-side layers(User Tower) or the item-side layers(Item Tower). The interaction layer in retrieval is commonly simplified as dot product due to serving efficiency. The development of recommendation models primarily relies on platforms such as PyTorch , TorchRec , TensorFlow , and Jax  which provide user-friendly Python interface for building and training neural networks. They convert models into a graph of kernel operators on GPU during execution. SilverTorch leverages PyTorch to construct all components for recommendation serving, as shown in Figure <a href="#fig:st_prototype" data-reference-type="ref" data-reference="fig:st_prototype">1</a>.

To efficiently retrieve items given a user embedding, a critical problem is vector search, which is a trending topic in domains like natural language processing , computer vision , search , and recommendation systems . Existing solutions are either using libraries like Faiss  as dedicated services  or building a vector database . Libraries like Faiss-GPU , Milvus  and CAGRA  provide GPU implementations for ANN search. Unfortunately, they only support topk and probes up to 2048  and topk up to 1024 . As retrieval models for recommendation evolve, tens of thousands items returned from ANN search are fed into interaction layers . SilverTorch provides a PyTorch native implementation of nearest neighbor search running on GPUs and is highly efficient and customizable. SilverTorch’s ANN module can efficiently return tens of thousands items. For feature filtering, traditional search engines  employ inverted index which can be also applied to recommendation. These systems are CPU-based solutions and introduce overhead and cost when serving recommendation models. SilverTorch’s bloom index idea is motivated by Bitfunnel , which tries to replace the inverted index in text search with bloom-filter-based signatures. However, Bitfunnel is a CPU solution and not specifically designed for recommendation. In recommendation, the cardinality of the feature attributes for each item is orders of magnitude smaller.
Both the existing ANN search index and inverted index are originally designed for other applications, requiring customizations for recommendation.
In contrast, SilverTorch’s model-based approach defines the ANN search and the feature filtering within PyTorch model layers running on GPUs. The unified authoring allows us to co-design the ANN search, the feature filtering and the OverArch.

Recommendation models are becoming more complex in terms of compute, model complexity and data . Serving these models is challenging but is not well studied. SilverTorch aims to support serving evolving recommendation model architectures.   extends the dot-product retrieval with the idea of mixture of logits and h-indexer, and shows the advantages using such complex structures.   proposes multi-task multi-head approach for item-to-item-based retrieval. But these works do not discuss their infrastructures to serve such advanced model architectures. LiNR  proposes model-based retrieval on GPUs. Unlike SilverTorch, LiNR does not provide co-designed index and does not propose a GPU solution for filtering. It also does not discuss the cost efficiency of introducing GPUs. Merlin  discusses the possibility of using GPU to accelerate the whole stack of recommendation from data processing, training to inference, but it is a combination of different libraries using GPUs.

# SilverTorch Overview

We propose SilverTorch, a model-based approach to unify recommendation. Instead of building standalone indexing systems, SilverTorch defines all serving components as model layers. The retrieval flow is no different from a forward function execution of tensor operators. Figure <a href="#fig:st_arch" data-reference-type="ref" data-reference="fig:st_arch">2</a> illustrates the overall workflow.

After training, a publish flow is executed to compose a SilverTorch model. We initially load trained models. The embedding evaluator utilizes both item features to calculate the item embeddings. We leverage GPUs to calculate the item embeddings. The item embeddings are then processed by the ANN index builder. It quantizes the embeddings to Int8 precision and adopts KMeans++-based  training on GPUs. For feature filtering, SilverTorch introduces a novel signature-based GPU index(referred as Bloom Index). The bloom index transforms search index matching to bit operations. Both the ANN index and bloom index are represented as GPU tensors that serve as parameters of the served model.
Meanwhile, User Tower, OverArch layers, and Value Model are quantized to BFloat16 and constructed as model parameters by corresponding builders. The model composer combines all the components into a SilverTorch model. Finally, the model optimizer compiles the eager-mode SilverTorch model into a graph and applies lowering and scripting.
The output from publish is a model snapshot containing all weights tensors and index tensors that can be served in a pure C++ runtime(referred to as predictor).

During online serving, the predictor runtime is simplified as a forward function execution of a SilverTorch model.
The SilverTorch retrieval model extracts user features and the filtering query from recommendation request, executes a sequence of kernel operators on GPUs. First, the User Tower computes the user embedding. Subsequently, the Bloom index layer further filters out irrelevant items and generate a mask tensor. The user embedding and the mask tensor are then fed into model’s ANN layer and get O(10,000) item ids as pre-filter results. The OverArch layer fetches corresponding item embeddings from embedding cache, and re-rank items by calculating scores and aggregate using the value model across multi-tasks. Final retrieval result returns O(1,000) ids. Final retrieval results are sent to ranking models.

<figure id="fig:st_arch" data-latex-placement="h">
<img src="figures/overview/st_overall.png" style="width:80.0%" />
<figcaption>Overall workflow of SilverTorch model publish and serving flow.</figcaption>
</figure>

# SilverTorch Model Design

The idea of SilverTorch is to define the recommendation serving components as model, which provides a unified interface and enables the co-design between components. This section first introduces key abstractions in SilverTorch, and discuss in-model design containing the bloom index based feature filtering and a fused Int8 ANN search. Finally, we discuss a co-designed ANN search and feature filtering algorithm.

<u>Index as Model</u>. Both ANN search and feature filtering are required to efficiently serve online retrieval requests within a latency budget. A recommendation query can be defined as follows:

    ANN_Index(user_emb) AND 
    (feature1=value1 OR feature1=value2 OR ...) AND
    (feature3=value3 OR feature4=value4 OR ...) AND..

A concrete example is below:

    ANN_Index(user_emb) AND item_country = "US"
    AND (item_lang = "EN" OR item_lang = "ES")

The user embedding (user_emb) is computed from User Tower, while item attributes such as item_country and item_lang are defined during publish. Query parameters include user-specific features (user_emb, user_country, user_lang1, user_lang2).
To support the ANN search sub-query, SilverTorch provides a fused Int8 ANN kernel leveraging the IVF(inverted file indexing) algorithm. It first probes a subset of clusters close to the query. Then it calculates topk items within each cluster and generates the global topk result.
For feature filtering, we propose bloom index, which leverage efficient bit-wise computation on GPUs. We optimize the feature filtering with ANN search by proposing a co-designed index.
During online serving, the retrieval model is executed in a simplified predictor, which eliminates the communication between standalone indexing services and fully utilizes GPU resources.

<u>Embedding and Feature Cache</u>. For the OverArch, to reduce online inference cost, SilverTorch pre-computes item embeddings during publish and populates results on GPUs as embedding cache. The in-model cache look-up removes the dependency of caching services and keeps the computation inside the GPUs. Similarly, the static features are cached on GPUs.

## Bloom Index

A recommendation query contains feature filtering to match item features with user attributes, represented as nested logical expressions. A common approach is inverted index, which maps each feature value to a posting list of matching items.
However, inverted index is not well suited to GPUs. First, unlike text terms in web search that follow a Zipf distribution with many short postings, recommendation features are typically broad and dense. The lack of sparsity eliminates the efficiency gains that inverted indexes provide in needle-in-the-haystack scenarios. Second, the list-based structure of inverted indexes is inherently sequential and misaligned with GPU parallelism. These limitations motivate our design of a more efficient feature filtering index for recommendation.

We revisit feature filtering problem with two key observations.
First, the query structure is known in advance, enabling optimized data layouts.
Second, recommendation items contain few feature values per item compared to high-cardinality text search.
Forward index offers stateless query evaluation, enabling parallel processing across all items—a natural fit for GPUs.
Each thread evaluates a partition of items and matches local results, improving efficiency in high-recall scenarios compared to inverted index’s multi-way merge. We can represent forward index with three tensors: feature_ids for identifiers, feature_values tensor for grouped values per (item, feature) pair, and feature_offsets tensor for indexing feature_values, defining value boundaries for each feature. The
feature_values for a given (item, feature_id) pair are sorted.
Despite enabling parallelism, forward index has limitations. Offset lookups for each (item, feature) pair create irregular and non-contiguous memory access patterns, limiting throughput when multiple filter conditions are matched. Since the GPU memory is expensive and largely impact the system throughput, forward index requires a large memory footprint because it stores the feature values using int64. To address these limitations, we propose the Bloom Index, a bloom-filter-based GPU indexing structure. It addresses the warp divergence of forward index, ensures contiguous memory access through bitwise operations, and significantly reduces memory consumption by representing each item with compact signature bits.

<figure id="fig:bloom_all" data-latex-placement="h">
<embed src="figures/system-design/bloom_all.pdf" style="width:100.0%" />
<figcaption>Explanations of Bloom Index Algorithm for Feature Filtering</figcaption>
</figure>

We formulate the feature filtering problem as follows, given a list of items V, where each item is represented as a list of features:
``` math
\begin{equation}
    V = \{ V_i = [f_1, f_2, ... f_t] \}
\end{equation}
```
Given a query Q, which can also be represented as a list of features:
``` math
\begin{equation}
    Q = [f_1, f_2, ... f_t]
\end{equation}
```
The goal is to find a list of items that meet following criteria:
``` math
\begin{equation}
    R = \{ V_i \in V \mid Q \cap V_i == Q \}
\end{equation}
```
Bloom filter is a hash-based structure used to check whether an element exists in a set. In bloom index design, we construct a M-bit bloom filter for each item, denoted as VB_i. For each feature, we apply K hash functions to compute hash_i(feature) % N and set the corresponding bits in bloom filter to 1. Similarly, we generate a M-bit bloom filter for the filtering query, denoted as QB, and apply previous k hash functions to mark corresponding bits to 1. By applying this to (3), we obtain:
``` math
\begin{equation}
    R = \{ V_i \in V,  V_i = VB_i \mid QB  \&  VB_i == QB \}
\end{equation}
```

To implement (4), each thread iterates through the bits of QB and then iterates each item in its partition to compute the matches. Figure <a href="#fig:bloom_all" data-reference-type="ref" data-reference="fig:bloom_all">3</a>(a) illustrates this process using a real example. While transforming filtering into bit manipulation of matrix, it has two drawbacks. First, each GPU thread processes one item at a time. Second, items corresponding to 0 bits in QB are also evaluated. We optimize it by only examining the bits set to 1 in QB. If all the corresponding bits in VB_i are also 1, it is a match. By rotating the matrix, we isolate the rows containing 1 bits in the query and skip the rest. Such transpose allowing multiple items to be matched simultaneously while eliminating the bit masks necessary to perform Boolean computation.
We only calculate rows containing the 1 bit from QB by performing a bit-wise AND operation between all the matched rows. Any 0 bits indicate a non-match, while 1 bits indicate a match. Figure <a href="#fig:bloom_all" data-reference-type="ref" data-reference="fig:bloom_all">3</a>(b) shows the process how our bloom index works. QB is a 8-bit bloom filter, and V2 and V6 are matches.
We use a single instruction to execute a 64-bit AND operation (PTX: and.b64), meaning each thread processes 64 items with a single instruction. Comparing to forward index, which matches item-by-item iterative inside a partition and requires a Boolean to store the result for each item, bloom index process 64 items simultaneously and stores the results of 64 items using one Int64. For a case of 40 million items, the data can be splitted into 625,000 partitions. Each thread processes $`\frac{625,000}{\text{\#Threads}}`$ partitions.
Bloom index leverages bloom filter, which may introduce false positives due to hash collision. By tuning the bloom filter size M and the number of hash functions K based on the number of features N, we could keep the rate very low. Additionally, rare false positive items generated by bloom index in retrieval can always be eliminated by subsequent ranking stages.

## Fused Int8 ANN Search

Existing ANN systems such as Faiss and Milvus are general-purpose libraries requiring customization for recommendation use cases. Although both support GPU execution, they impose hard limitations on top-k size. Modern retrieval models use ANN search as a pre-filter returning O(10,000) items, followed by an OverArch model for re-ranking. Consequently, most production systems still rely on CPU-based ANN search due to its scalability and maturity.
To overcome these limitations, we propose a Int8 ANN search kernel in SilverTorch using tensors as index containers. This tensor-native design integrates directly into arbitrary ML serving stack and is inherently parallel-friendly on GPUs. We adopt the clustering-based IVF algorithm with three steps: (1) compute dot products between query and centroid embeddings, (2) search top item embeddings within selected clusters, and (3) identify global top-k across clusters.
We identified the bottleneck in the index selection step, which generates a large temporary tensor to retrieve embeddings for top-k computation. To address this, we introduce a fused index-matmul operator that streams item embeddings directly from the embedding table and computes dot products with batched queries on-the-fly, avoiding intermediate tensor construction. By assigning each warp to process a contiguous tile of items, this operator fully exploits GPU parallelism with coalesced memory access.
We further observe that even precise nearest neighbor results may not yield perfect retrieval accuracy, as dot-product similarity oversimplifies the retrieval model. Additionally, storing complete embedding tensors on GPU becomes a memory bottleneck as candidate pools grow. Based on these observations, we propose Int8 quantized fused ANN search. By representing embeddings with 8-bit integers and leveraging GPU’s dp4a instruction (computing four multiply-adds in one instruction), we achieve higher throughput and halve the memory footprint. Quantization occurs during model publish: we compute global min/max values across all embeddings, scale them to \[-128, 127\], and assign integer representations accordingly.
Our Int8 quantized ANN search shows limited quality loss while significantly improving serving performance. It frees headroom for the OverArch layers to rank more items and improves the end-to-end retrieval accuracy. The algorithm supports large top-k and probe counts; in practice, we observe no recall loss with 64 probes and top-2048.

## ANN and Filtering Co-design

SilverTorch unifies ANN search and feature filtering operators within the PyTorch stack, with all indexes stored as GPU tensors in the same runtime. This unified design provides opportunities to co-design these operators. We observe that although full bloom filtering is accurate, much of the computation is wasted—items passing the filter are often excluded by ANN probing, which only scores items within selected clusters. This insight motivates our co-designed approach that reverses the traditional order of operations: we first identify which clusters to probe, then apply filtering only to items within those clusters. The results remain the same. Our implementation leverages several GPU-friendly optimizations. The ANN search operator accepts a bit mask from the bloom index using 1-bit per item instead of PyTorch’s native 8-bit boolean, conserving memory and reducing global memory bandwidth. For complex filter queries containing multiple logical expressions (AND, OR, NOT), we parse queries and pre-compute each feature’s bloom result, storing operators and results in an operation array. During evaluation, we process this array sequentially using a stack to push temporary results and pop for logical computation. Since a single batch may contain hundreds of sub-queries requiring extensive bloom computation, and GPU memory bounds model throughput, reducing the filtering scope becomes critical.

Algorithm <a href="#alg:partial_bloom" data-reference-type="ref" data-reference="alg:partial_bloom">[alg:partial_bloom]</a> presents our co-designed index. In Phase 1, we compute query-centroid distances and select the top-$`n_p`$ clusters, identifying which items will actually be scored. In Phase 2, we compute bloom filter results only for items within selected clusters, generating compact bit masks $`\mathbf{M}_c`$ for each cluster. Phase 3 computes embedding similarity scores only for items passing the partial bloom filter, combining filtering and scoring in a single GPU kernel pass. Finally, Phase 4 aggregates scores across all probed clusters and returns the top-$`k`$ item IDs and scores. For an index with 81 million items across 9,000 clusters, using 256 probes processes only 2.3 million items (2.8), achieving a 30$`\times`$ reduction in both filtering computation and GPU scratch memory.

<div class="algorithm">

<div class="algorithmic">

Query embedding $`\mathbf{q}`$, filter predicates $`\mathcal{F}`$, item embeddings $`\mathbf{E}`$, cluster centroids $`\mathbf{C}`$, cluster offsets $`\mathbf{O}`$, cluster lengths $`\mathbf{L}`$, Bloom index $`\mathbf{B}`$, number of probes $`n_p`$, $`k`$
Top-$`k`$ item IDs and scores
**// Phase 1: ANN Probing**
$`\mathbf{D} \gets \mathrm{Distance}(\mathbf{q}, \mathbf{C})`$
$`\mathcal{P} \gets \mathrm{TopK}(\mathbf{D}, n_p)`$
**// Phase 2: Partial Filtering on selected probes**
**for all** cluster $`c \in \mathcal{P}`$:
$`s_c \gets \mathbf{O}[c]`$, $`l_c \gets \mathbf{L}[c]`$
$`\mathbf{M}_c \gets \mathrm{BloomSearch}(\mathbf{B}, \mathcal{F}, s_c, l_c)`$
**// Phase 3: Fused Scoring with Partial Masks**
$`\mathbf{S} \gets \emptyset`$, $`\mathbf{I} \gets \emptyset`$
**for all** cluster $`c \in \mathcal{P}`$, item $`d \in c`$ where $`\mathbf{M}_c[d] = 1`$:
$`\mathbf{S} \gets \mathbf{S} \cup \{\langle \mathbf{q}, \mathbf{E}[d] \rangle\}`$, $`\mathbf{I} \gets \mathbf{I} \cup \{d\}`$
**// Phase 4: Global Top-K**
$`\mathbf{R} \gets \mathrm{ArgTopK}(\mathbf{S}, k)`$
$`\mathbf{S}^* \gets \mathbf{S}[\mathbf{R}]`$, $`\mathbf{I}^* \gets \mathbf{I}[\mathbf{R}]`$
$`(\mathbf{I}^*, \mathbf{S}^*)`$

</div>

</div>

# Extensibility

This section explores two extensions on top of canonical retrieval models, and discuss how we scale out SilverTorch to multi-GPUs.

## OverArch Scoring

In the two-tower model architecture, user-item similarities are computed using dot product. Unfortunately, this approach has no trainable parameters and oversimplifies user-item interactions. Without SilverTorch’s GPU-based approach, meeting latency requirements while adding complex interaction layers is challenging. SilverTorch substantially reduces inference cost, enabling retrieval models to incorporate scoring layers beyond nearest neighbor search.
Building on the ANN search and feature filtering queries defined in Section <a href="#sec:4" data-reference-type="ref" data-reference="sec:4">4</a>, a SilverTorch model includes additional scoring layers (referred to as the OverArch layer). The retrieval process operates in two steps. First, it executes the query to pre-filter $`K_0`$ items (on the order of $`O(10{,}000)`$ to $`O(100{,}000)`$) from original candidate pool, where dot product is still adopted for distance calculation within ANN search. Second, the OverArch employs neural network modules to rank the $`K_0`$ user-item pairs and returns the final retrieval results. We find that introducing OverArch achieves better recall for retrieval than enhancing ANN search accuracy alone.
An OverArch layer can be defined as a Multi-layer Perceptron (MLP), or a multiple stacked self-attention layers to capture the correlation to understand user’s interest, with the capability of looking at items in an entire session. Recent study proposes more structured interaction layer using Mixture of logits (MoL)   that defines similarity as adaptive composition of elementary functions. Benefiting from SilverTorch’s caching design, item embeddings and cross-features used in the OverArch can be directly extracted from GPU memory during publish. SilverTorch supports complex OverArch architectures that could improve model quality of retrieval through the full fidelity of training and serving consistency.

## Multi-Task Retrieval with Value Model

To learn multiple aspects of users, recommendation should predict multiple objectives to capture diverse user behaviors such as content—like, share, or comment. Multi-task learning addresses this by training a unified model that exploits commonalities and differences across tasks, improving prediction accuracy through knowledge sharing. However, previous retrieval systems largely avoided multi-task approaches due to prohibitive latency costs.

A straightforward idea is to apply different user embeddings and item embeddings to represent each task. However, item embeddings are learned to be the semantics representation so they should be shared across tasks. Therefore, in multi-task retrieval, the user tower shares a lookup table but applies task-specific dense layers to generate user embeddings per task, while item embeddings are shared across tasks.
At serving time, ANN search handles multi-task queries and returns task-specific item lists. CPU-based solutions require replicating the ANN index for each task to avoid latency increases, causing linear cost growth. In contrast, SilverTorch leverages GPU parallelism to batch requests from multiple tasks within a single index copy without latency regression, making multi-task retrieval cost-efficient.
After ANN search and filtering, a merge operation combines results across tasks before the OverArch layer predicts per-task scores. To aggregate these predictions into a unified engagement score, we introduce a Value Model (VM) that transforms the model predictions to business objectives. The VM combines predictions (likes, shares, comments) through user-defined formulas expressed in a JSON-like format with pre-defined conditions. SilverTorch implements a GPU VM kernel that parses formulas into an abstract syntax tree and process multiple items in parallel. Applying value-based aggregation at retrieval provides better consistency between retrieval and ranking.

## Scale Out

To handle larger candidate pools and models, SilverTorch scales out to multiple GPUs. The ANN index and Bloom index are sharded across GPU cards, with each GPU processing a partition of items, while OverArch parameters are replicated on each GPU. During serving, each GPU independently computes local pre-filtered results through ANN search and feature filtering. Item embeddings are then gathered to a single GPU to compute the final retrieval results. Since each GPU maintains a copy of the OverArch and Value Model, request batches can be evenly distributed across GPUs in parallel, preventing any single GPU from becoming a bottleneck.

GPU memory is the primary factor determining the number of GPUs required for serving. It depends on multiple factors: ANN and Bloom index size and weights of the OverArch. As candidate pool grows, both of Bloom and ANN index scale accordingly. ANN index size further depends on quantization precision and embedding dimensions. SilverTorch’s Int8 ANN and compact Bloom index designs significantly reduce GPU memory footprint. The complexity of the OverArch also constrains how many items a single GPU can serve before reaching limits. For user embedding tables, we leverage CPU-based parameter servers for distributed inference.
Adopting the unified design facilitates co-design across recommendation services, enabling future flexible disaggregation of serving components based on serving characteristics (e.g., compute-bound vs. memory-bound) rather than pre-defined boundaries. Additionally, real traffic is bursty, requiring automatic GPU scaling based on offline capacity estimation. We support QPS-based scaling that automatically scales GPU instances up or down within minutes. For extreme bursts exceeding capacity, excess traffic is throttled.

# Evaluation

We demonstrate that SilverTorch offers performance improvement with better cost-efficiency. We also show that SilverTorch enables more complex architectures that improves retrieval recalls. Our experiments use the real-world dataset sampled from production. The dataset consists of a candidate pool of 80 million items(referred to as 80M-dataset) and a candidate pool of 10 million items(referred to as 10M-dataset).
We set the embedding dimensions to 128 for experiments. We reuse the model architecture to evaluate performance and use A100 40G GPUs. The replayed requests are generated from real traffic containing 5000 recommendation requests. For the end-to-end experiments, we measure Query per Second(QPS) from client side given 200 ms P99 latency budget. We observe that given more small latency budget, the throughput number of the CPU baselines is too small to report. The client concurrently sends requests to predictor and indexing servers. For each test, we gradually increase sending QPS to avoid server-side congestion and report the maximum throughput it could reach. Each experiment is running 5 times and we report the average. To evaluate the OverArch and in-model Value Model, we measure recalls at different scales, together with QPS.

<u>Baseline-Retrieval</u> is a service-based retrieval system. The client first sends a request to predictor that serves User Tower on 1 GPU and gets the user embedding. The client sends user embedding with filtering queries to indexing servers. The indexing servers contain ANN search index built based on the Faiss-CPU(IVF) and inverted-index(CPU) for feature filtering. Each inverted-index server builds its index on a partition of items and the filtering queries are running in a scatter-gather manner. <u>Baseline-Retrieval-GPU</u> is a service-based retrieval baseline on GPUs. User Tower is served on 1 GPU. The ANN search index is built based on Faiss-GPU(IVF) and the filtering index is using GPU-based forward index discussed in <a href="#sec:4.1" data-reference-type="ref" data-reference="sec:4.1">4.1</a>. The ANN index and the forward index are served in multiple GPUs. Each GPU builds its index on a partition of items, the ANN search and filtering are running in a scatter-gather manner. 1 means no sharding. <u>SilverTorch-Retrieval</u> is the SilverTorch retrieval without OverArch and Value Model layers. Client sends a single request to GPU predictor. The server computes the user embedding followed by SilverTorch’s co-designed ANN search and bloom index, returning topk item ids. For the end-to-end experiments, we shard it to 2 GPUs for 80M-dataset and use a single GPU for 10M-dataset. <u>SilverTorch-Retrieval-OverArch</u> is SilverTorch retrieval with OverArch scoring layer and the in-model Value Model. This is the multi-task setup. We compare SilverTorch’s ANN with Faiss-CPU(IVF), Faiss-GPU(IVF) and HNSW. We compare Bloom Index with the GPU-based forward index and the CPU-based inverted index.

<figure id="fig:retrieval-e2e" data-latex-placement="h">
<img src="figures/exp/experiment_e2e_final2.png" style="width:100.0%" />
<figcaption>End-to-end performance Results for Retrieval.</figcaption>
</figure>

## End-to-end Evaluation

### Throughput

We build the state-of-the-art CPU-based baseline adopting the same model architecture. It is a multi-task model with 12 embedding heads per user, performing 12-way top-k ANN search. Filtering queries combine AND/OR/NOT operators across 6 features with 7 conditions on average, using Faiss-CPU for ANN and our internal inverted index implementation. For the 80M dataset, indexes are sharded across 2 CPU servers (Faiss) and 4 servers (inverted index)—the minimal configuration for meaningful QPS. Requests fan out to shards and merge at an aggregator. The 10M dataset runs un-sharded.
Figure <a href="#fig:retrieval-e2e" data-reference-type="ref" data-reference="fig:retrieval-e2e">4</a> shows end-to-end performance varying ANN probes with top-k fixed at 1024. Faiss-CPU uses 64 OpenMP threads. GPU baselines are labeled FaissX-ForwardY, indicating X Faiss shards and Y forward index shards.
For the 80M dataset (40GB memory requiring 2 Faiss-GPU shards) at 24 probes(production setting), SilverTorch achieves 1210 QPS—$`23.7\times`$ over CPU baseline and $`3.5\times`$–$`6.7\times`$ over GPU baselines. Although GPU baseline performance improves with more shards, cost increases linearly.
For the 10M dataset without sharding, SilverTorch achieves 3802 QPS—$`165.3\times`$ over CPU and $`20.8\times`$ over GPU baselines. Notably, SilverTorch QPS scales $`3.1\times`$ when reducing pool size from 80M to 10M, while baseline QPS remains similar since per-server item count is unchanged.

### Cost Efficiency Analysis

To illustrate the cost efficiency of SilverTorch, we estimate Total Cost of Ownership (TCO) reduction using QPS results from the 80M-dataset experiments. Our CPU server has 256GB memory and 40 cores, while GPU servers have 48 CPU cores, 384GB memory, and varying numbers of A100 40GB cards. We map these to similar AWS instance types to estimate TCO.
For the CPU server, we use r6i.8xlarge (32 vCPUs, 256GB memory) at 2.24/hour  as a lower-bound estimate. For GPU servers, AWS offers A100 40GB exclusively in p4d.24xlarge instances with 8 GPU cards, 96 vCPUs, and 1.15TB memory at 32.77/hour. To estimate the cost of a single A100 40GB card, we subtract the CPU-equivalent cost (x2idn.24xlarge at 13.01/hour ) and divide by 8, yielding 2.47/hour per GPU. Thus, the baseline’s user tower with one GPU costs approximately 15.47/hour.
Using the QPS results, we evaluate cost efficiency as follows. Baseline-Retrieval uses one GPU for user tower, 2 CPU servers for ANN search, and 4 CPU servers for filtering, totaling 28.92/hour with 51 QPS. The GPU baselines Baseline-Retrieval-GPU-Faiss2-Forward4 and Baseline-Retrieval-GPU-Faiss2-Forward6 cost 30.29 and 32.77/hour, achieving 184 and 340 QPS respectively. SilverTorch-Retrieval uses one GPU server with 2 A100 40GB cards at 32.77/hour , achieving 1210 QPS.
Table <a href="#tab:TCO-analysis" data-reference-type="ref" data-reference="tab:TCO-analysis">[tab:TCO-analysis]</a> shows the TCO breakdown for serving traffic at 1000 QPS. SilverTorch achieves $`20.9\times`$ cost-efficiency improvement over CPU baseline and $`3.56\times`$ over the best GPU baseline. Furthermore, when including computational overhead of OverArch and Value Model, SilverTorch’s QPS decreases to 771, which still delivers $`13.35\times`$ improvement over CPU solution and $`2.27\times`$ over GPU solution.

## Breakdown Analysis

In this section, we evaluate the performance of each component in SilverTorch. We focus on a single GPU experiments.

### Evaluation on ANN Search

<figure id="fig:ann_latency_recall">
<img src="figures/exp/ann_latency_recall2.png" style="width:50.0%" />
<figcaption>Latency/Recall results of different ANN methods.</figcaption>
</figure>

We compare SilverTorch’s ANN search against both CPU and GPU baselines. For Faiss, we evaluate the most efficient IVFFlat index with float32 on both CPU and GPU, as well as HNSW. CAGRA(GPU) limits top-k to 1024, making it unsuitable for recommendation scenarios for comparison.
Since HNSW is graph-based without a probes parameter, we compare average latency at equivalent recall levels. The dataset contains 20 million item embeddings with 128 dimensions, with query embeddings generated from the User Tower. We focus on single-server and single-GPU performance, running 50 warm-up batches followed by 100 test batches. While in practice we use top-k around 10,000, we test with 2048 and 4096 since Faiss-GPU only supports up to 2048.
Figure <a href="#fig:ann_latency_recall" data-reference-type="ref" data-reference="fig:ann_latency_recall">5</a> shows average latency across different recalls at batch size 16. SilverTorch-INT8 achieves the lowest latency in all cases. At top-k=2048, SilverTorch achieves $`2.2\times`$–$`14.7\times`$ lower latency than Faiss-GPU across recalls from 0.35 to 0.92. Due to INT8 quantization, SilverTorch cannot reach 0.95 recall. However, achieving this recall requires Faiss-CPU to use 1024 probes and Faiss-GPU to use 512 probes, which significantly degrades its performance. The latency savings from SilverTorch’s ANN search can fund additional OverArch model computation.
At top-k=4096, Faiss-GPU cannot support it. SilverTorch-INT8 achieves $`31.3\times`$–$`51\times`$ lower latency than HNSW and $`4.6\times`$–$`49.2\times`$ lower latency than Faiss-CPU. Additionally, tuning cluster-based ANN is simpler(only probe parameter) comparing to HNSW’s three interdependent parameters.

<div class="table*">

<table>
<thead>
<tr>
<th style="text-align: left;"><strong>Task</strong></th>
<th style="text-align: left;"><strong>Method</strong></th>
<th style="text-align: center;"><strong>Recall@20</strong></th>
<th style="text-align: center;"><strong>Recall@100</strong></th>
<th style="text-align: center;"><strong>Recall@200</strong></th>
<th style="text-align: center;"><strong>Recall@500</strong></th>
<th style="text-align: center;"><strong>Recall@1000</strong></th>
<th style="text-align: center;"><strong>QPS</strong></th>
<th style="text-align: center;"></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: left;">E-Task</td>
<td style="text-align: left;">Baseline</td>
<td style="text-align: center;">0.08239</td>
<td style="text-align: center;">0.19179</td>
<td style="text-align: center;">0.29131</td>
<td style="text-align: center;">0.4295</td>
<td style="text-align: center;">0.44127</td>
<td style="text-align: center;">51</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">SilverTorch</td>
<td style="text-align: center;">0.07163</td>
<td style="text-align: center;">0.20306</td>
<td style="text-align: center;">0.28923</td>
<td style="text-align: center;">0.4237</td>
<td style="text-align: center;">0.44651</td>
<td style="text-align: center;">1210</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">SilverTorch-OverArch</td>
<td style="text-align: center;"><strong>0.09181 (+28.2%)</strong></td>
<td style="text-align: center;"><strong>0.24189 (+19.1%)</strong></td>
<td style="text-align: center;"><strong>0.33148 (+14.6%)</strong></td>
<td style="text-align: center;"><strong>0.44758 (+5.6%)</strong></td>
<td style="text-align: center;"><strong>0.45727 (+2.4%)</strong></td>
<td style="text-align: center;">771</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">C-Task</td>
<td style="text-align: left;">Baseline</td>
<td style="text-align: center;">0.09642</td>
<td style="text-align: center;">0.25217</td>
<td style="text-align: center;">0.3551</td>
<td style="text-align: center;">0.4971</td>
<td style="text-align: center;">0.5162</td>
<td style="text-align: center;">51</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">SilverTorch</td>
<td style="text-align: center;">0.09652</td>
<td style="text-align: center;">0.25291</td>
<td style="text-align: center;">0.352</td>
<td style="text-align: center;">0.4969</td>
<td style="text-align: center;">0.51973</td>
<td style="text-align: center;">1210</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">SilverTorch-OverArch</td>
<td style="text-align: center;"><strong>0.0971 (+0.6%)</strong></td>
<td style="text-align: center;"><strong>0.25733 (+1.7%)</strong></td>
<td style="text-align: center;"><strong>0.36011 (+2.3%)</strong></td>
<td style="text-align: center;"><strong>0.50747 (+2.2%)</strong></td>
<td style="text-align: center;"><strong>0.52559 (+1.12%)</strong></td>
<td style="text-align: center;">771</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

### Evaluation on Bloom Index

We compare Bloom index performance against a production CPU inverted-index baseline and the forward index baseline on GPU. We evaluate on a single server with an index built on 40 million items, using 5,000 real filtering queries. Each item contains 6 features with 10 feature values on average. The Bloom index uses 5 hash functions.
Figure <a href="#fig:bloom-all" data-reference-type="ref" data-reference="fig:bloom-all">6</a>(a) shows average latency per batch at different batch sizes. Bloom index supports batching more effectively, achieving $`291\times`$–$`523\times`$ speedup over inverted index and $`12.6\times`$–$`42.7\times`$ over forward index. Bloom index latency remains constant regardless of bit size (512 to 1024 bits), since queries involve a fixed number of hash functions and bitwise memory accesses that are efficiently parallelized via GPU warp-level execution. In contrast, inverted index latency varies significantly depending on posting list lengths.
Figure <a href="#fig:bloom-all" data-reference-type="ref" data-reference="fig:bloom-all">6</a>(b) shows false positive rates at different bit sizes. Using 512 bits per item (1.2 GB total) yields 6.98 false positive rate, dropping to 0.067 at 1024 bits. The 512, 768, 1024, and 2048-bit configurations require 1.2, 1.8, 2.4, and 4.7 GB respectively. Inverted index requires 19.8 GB—$`8.25\times`$ larger than the 1024-bit Bloom index.
A heuristic for estimating optimal bit count is: maxfeaturevalues $`\times`$ hashfunctions $`\times`$ collisionbuffer. In our query set, with maximum 120 feature values per item, 5 hash functions, and buffer of 3, this yields 1800 bits per item.

<figure id="fig:bloom-all" data-latex-placement="h">
<img src="figures/exp/bloom-all.png" style="width:50.0%" />
<figcaption>Performance of Bloom Index.</figcaption>
</figure>

<figure id="fig:bloom-joint">
<img src="figures/exp/co-designed.png" style="width:50.0%" />
<figcaption>Latency and memory utilization comparison between co-designed ANN+Bloom and original ANN+Bloom.</figcaption>
</figure>

### Evaluation on ANN and filtering co-designed Index

We evaluate co-designed index performance using a 20 million item dataset with 128-dimensional embeddings, comparing against a baseline that runs Bloom index separately and passes mask results to ANN. Figure <a href="#fig:bloom-joint" data-reference-type="ref" data-reference="fig:bloom-joint">7</a> shows GPU memory utilization and latency across probe counts. At probe=32, the baseline requires 35.6MB scratch memory (17.4MB for ANN, 18.2MB for Bloom index). Co-design reduces Bloom index scratch memory to 0.14MB, lowering total memory to 18.2MB while reducing latency from 1.55ms to 0.72ms. On average, co-design achieves $`1.79\times`$–$`2.15\times`$ latency improvement. As discussed, memory savings in this memory-bound scenario enable larger request batching and higher QPS.

### Evaluation on OverArch Scoring and Value Model

We evaluate OverArch and Value Model benefits by comparing recalls at different sizes, with ground truth from user-item interaction behaviors and probes set to 32. We report recalls for a major engagement event (E-Task) and consumption event (C-Task). The OverArch implements Mixture of Logits (MoL), while the Value Model applies rules validated through online A/B testing.
As shown in Table <a href="#tab:recall-overarch" data-reference-type="ref" data-reference="tab:recall-overarch">[tab:recall-overarch]</a>, adding scoring layers improves E-Task recall by 2.4–28.2 and C-Task recall by 0.6–1.12 across different sizes. While QPS only decreases from 1210 to 771, SilverTorch still achieves $`15.11\times`$ QPS speedup compared to service-based solutions.

# Discussion and Future Work

SilverTorch is designed for large-scale recommendation with hundreds of millions of items. However, it also provides benefits of unified development and serving at smaller scales. Instead of maintaining separate systems across multiple CPU servers, we deploy a single GPU server to support thousands of QPS. For low-traffic scenarios, the remaining capacity accommodates future business growth. For extremely small-scale cases, users can switch to a CPU execution with a configuration change to run the SilverTorch model.

SilverTorch supports item freshness and builds an update service on top of offline publish, which updates the index in a streaming way. One approach is to use a pre-allocated memory space and a watermark to accept insertions in an append-only fashion . Unfortunately, such solution has notable scalability limitations. The fresh part of the index requires a linear scan to retrieve results, which makes it inefficient during serving. In contrast, we create the concept of fresh index, which has exact same layout of the main index and is self-contained and independently updated. During Serving, a item streaming enabled SilverTorch model merges the pre-filtered items from main index with fresh index before the OverArch. We leave the details of fresh index for future work.

# Conclusion

SilverTorch, a model-based recommendation serving system on GPUs, simplifies client-side logic and eliminates dependencies on standalone services. We propose a fused Int8 ANN index, Bloom index for feature filtering as model layers, and a co-designed index. SilverTorch extends retrieval models with OverArch and in-model Value Model, enhancing recall while maintaining latency budget, employs item embedding caching to reduce online computation. Our experiments demonstrate scaling to millions of items across multiple GPUs with $`23.7\times`$ throughput improvement and $`13.35\times`$ better cost-efficiency compared to state-of-the-art systems.

<div class="thebibliography">

41

Martı́n Abadi, Paul
Barham, Jianmin Chen, Zhifeng Chen,
Andy Davis, Jeffrey Dean,
Matthieu Devin, Sanjay Ghemawat,
Geoffrey Irving, Michael Isard,
et al. 2016.
$`\{`$TensorFlow$`\}`$: a system for
$`\{`$Large-Scale$`\}`$ machine learning. In *12th
USENIX symposium on operating systems design and implementation (OSDI 16)*.
265–283.

Newsha Ardalani,
Carole-Jean Wu, Zeliang Chen,
Bhargav Bhushanam, and Adnan Aziz.
2022.
Understanding scaling laws for recommendation
models.
*arXiv preprint arXiv:2208.08489*
(2022).

David Arthur and Sergei
Vassilvitskii. 2006.
*k-means++: The advantages of careful
seeding*.
echnical Report.
Stanford.

Paul Baltescu, Haoyu
Chen, Nikil Pancha, Andrew Zhai,
Jure Leskovec, and Charles Rosenberg.
2022.
Itemsage: Learning product embeddings for shopping
recommendations at pinterest. In *Proceedings of
the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.
2703–2711.

Fedor Borisyuk, Qingquan
Song, Mingzhou Zhou, Ganesh
Parameswaran, Madhu Arun, Siva Popuri,
Tugrul Bingol, Zhuotao Pei,
Kuang-Hsuan Lee, Lu Zheng,
et al. 2024.
LiNR: Model Based Neural Retrieval on GPUs at
LinkedIn. In *Proceedings of the 33rd ACM
International Conference on Information and Knowledge Management*.
4366–4373.

Sergey Brin and Lawrence
Page. 1998.
The anatomy of a large-scale hypertextual web
search engine.
*Computer networks and ISDN systems*
30, 1-7 (1998),
107–117.

B Barla Cambazoglu and
Ricardo Baeza-Yates. 2016.
Scalability and efficiency challenges in
large-scale web search engines. In *Proceedings of
the 39th International ACM SIGIR conference on Research and Development in
Information Retrieval*. 1223–1226.

Paul Covington, Jay
Adams, and Emre Sargin.
2016.
Deep neural networks for youtube recommendations.
In *Proceedings of the 10th ACM conference on
recommender systems*. 191–198.

Bailu Ding and Jiaqi
Zhai. 2025.
Retrieval with Learned Similarities. In
*Proceedings of the ACM on Web Conference 2025*.
1626–1637.

Matthijs Douze, Alexandr
Guzhva, Chengqi Deng, Jeff Johnson,
Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini,
and Hervé Jégou. 2024.
The Faiss library.
(2024).
arXiv:2401.08281 \[cs.LG\]

Roy Frostig, Matthew James
Johnson, and Chris Leary.
2019.
Compiling machine learning programs via high-level
tracing. In *SysML conference 2018*.

github 2023.
Faiss on the GPU Limitations.
<https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#limitations>

Bob Goodwin, Michael
Hopcroft, Dan Luu, Alex Clemmer,
Mihaela Curmei, Sameh Elnikety, and
Yuxiong He. 2017.
Bitfunnel: Revisiting signatures for search. In
*Proceedings of the 40th International ACM SIGIR
Conference on Research and Development in Information Retrieval*.
605–614.

Jui-Ting Huang, Ashish
Sharma, Shuying Sun, Li Xia,
David Zhang, Philip Pronin,
Janani Padmanabhan, Giuseppe Ottaviano,
and Linjun Yang. 2020.
Embedding-based retrieval in facebook search. In
*Proceedings of the 26th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining*.
2553–2561.

Dmytro Ivchenko, Dennis
Van Der Staay, Colin Taylor, Xing Liu,
Will Feng, Rahul Kindi,
Anirudh Sudarshan, and Shahin Sefati.
2022.
Torchrec: a pytorch domain library for
recommendation systems. In *Proceedings of the 16th
ACM Conference on Recommender Systems*. 482–483.

Suhas Jayaram Subramanya,
Fnu Devvrit, Harsha Vardhan Simhadri,
Ravishankar Krishnawamy, and Rohan
Kadekodi. 2019.
Diskann: Fast accurate billion-point nearest
neighbor search on a single node.
*Advances in neural information processing
Systems* 32 (2019).

Jeff Johnson, Matthijs
Douze, and Hervé Jégou.
2019.
Billion-scale similarity search with GPUs.
*IEEE Transactions on Big Data*
7, 3 (2019),
535–547.

Patrick Lewis, Ethan
Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis,
Wen-tau Yih, Tim Rocktäschel,
et al. 2020.
Retrieval-augmented generation for
knowledge-intensive nlp tasks.
*Advances in neural information processing
systems* 33 (2020),
9459–9474.

Milvus 2023.
Milvus GPU Limitations.
<https://milvus.io/docs/gpu_index.md>.

Philipp Moritz, Robert
Nishihara, Stephanie Wang, Alexey
Tumanov, Richard Liaw, Eric Liang,
Melih Elibol, Zongheng Yang,
William Paul, Michael I Jordan,
et al. 2018.
Ray: A distributed framework for emerging
$`\{`$AI$`\}`$ applications. In *13th USENIX symposium
on operating systems design and implementation (OSDI 18)*.
561–577.

Dheevatsa Mudigere, Yuchen
Hao, Jianyu Huang, Zhihao Jia,
Andrew Tulloch, Srinivas Sridharan,
Xing Liu, Mustafa Ozdal,
Jade Nie, Jongsoo Park, et al.
2022.
Software-hardware co-design for fast and scalable
training of deep learning recommendation models. In
*Proceedings of the 49th Annual International
Symposium on Computer Architecture*. 993–1011.

Maxim Naumov, Dheevatsa
Mudigere, Hao-Jun Michael Shi, Jianyu
Huang, Narayanan Sundaraman, Jongsoo
Park, Xiaodong Wang, Udit Gupta,
Carole-Jean Wu, Alisson G Azzolini,
et al. 2019.
Deep learning recommendation model for
personalization and recommendation systems.
*arXiv preprint arXiv:1906.00091*
(2019).

Even Oldridge, Julio
Perez, Ben Frederickson, Nicolas
Koumchatzky, Minseok Lee, Zehuan Wang,
Lei Wu, Fan Yu, Rick
Zamora, Onur Yilmaz, et al.
2020.
Merlin: a gpu accelerated recommendation
framework. In *Proceedings of IRS*.

Hiroyuki Ootomo, Akira
Naruse, Corey Nolet, Ray Wang,
Tamas Feher, and Yong Wang.
2024.
Cagra: Highly parallel graph construction and
approximate nearest neighbor search for gpus. In
*2024 IEEE 40th International Conference on Data
Engineering (ICDE)*. IEEE, 4236–4247.

Weston Pace, Chang She,
Lei Xu, Will Jones,
Albert Lockett, Jun Wang, and
Raunak Shah. 2025.
Lance: Efficient Random Access in Columnar Storage
through Adaptive Structural Encodings.
*arXiv preprint arXiv:2504.15247*
(2025).

A Paszke. 2019.
Pytorch: An imperative style, high-performance deep
learning library.
*arXiv preprint arXiv:1912.01703*
(2019).

Pinterest.
2023.
Manas HNSW Realtime: Powering Realtime
Embedding-Based Retrieval.
<https://medium.com/pinterest-engineering/manas-hnsw-realtime-powering-realtime-embedding-based-retrieval-dc71dfd6afdd>.

Rapidsai. 2022.
Rapidsai/raft: RAFT contains fundamental widely-used
algorithms and primitives for data science, Graph and machine learning.
<https://github.com/rapidsai/raft>

vantage. 2025a.
AWS p4d.24xlarge Instance Cost.
<https://instances.vantage.sh/aws/ec2/p4d.24xlarge?region=us-west-2>.

vantage. 2025b.
AWS r6i.8xlarge Instance Cost.
<https://instances.vantage.sh/aws/ec2/r6i.8xlarge?region=us-west-1>.

vantage. 2025c.
AWS x2idn.24xlarge Instance Cost.
<https://instances.vantage.sh/aws/ec2/x2idn.24xlarge?region=us-west-1>.

Jizhe Wang, Pipei Huang,
Huan Zhao, Zhibo Zhang,
Binqiang Zhao, and Dik Lun Lee.
2018.
Billion-scale commodity embedding for e-commerce
recommendation in alibaba. In *Proceedings of the
24th ACM SIGKDD international conference on knowledge discovery & data
mining*. 839–848.

Jianguo Wang, Xiaomeng
Yi, Rentong Guo, Hai Jin,
Peng Xu, Shengjun Li,
Xiangyu Wang, Xiangzhou Guo,
Chengming Li, Xiaohai Xu,
et al. 2021.
Milvus: A purpose-built vector data management
system. In *Proceedings of the 2021 International
Conference on Management of Data*. 2614–2627.

Xinyang Yi, Ji Yang,
Lichan Hong, Derek Zhiyuan Cheng,
Lukasz Heldt, Aditee Kumthekar,
Zhe Zhao, Li Wei, and
Ed Chi. 2019.
Sampling-bias-corrected neural modeling for large
corpus item recommendations. In *Proceedings of the
13th ACM conference on recommender systems*. 269–277.

Andrew Zhai, Hao-Yu Wu,
Eric Tzeng, Dong Huk Park, and
Charles Rosenberg. 2019.
Learning a unified embedding for visual search at
pinterest. In *Proceedings of the 25th ACM SIGKDD
international conference on knowledge discovery & data mining*.
2412–2420.

Jiaqi Zhai, Zhaojie Gong,
Yueming Wang, Xiao Sun,
Zheng Yan, Fu Li, and
Xing Liu. 2023.
Revisiting neural retrieval on accelerators. In
*Proceedings of the 29th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining*. 5520–5531.

Jiaqi Zhai, Lucy Liao,
Xing Liu, Yueming Wang,
Rui Li, Xuan Cao, Leon
Gao, Zhaojie Gong, Fangda Gu,
Michael He, et al.
2024.
Actions speak louder than words: Trillion-parameter
sequential transducers for generative recommendations.
*arXiv preprint arXiv:2402.17152*
(2024).

Buyun Zhang, Liang Luo,
Yuxin Chen, Jade Nie, Xi
Liu, Daifeng Guo, Yanli Zhao,
Shen Li, Yuchen Hao,
Yantao Yao, et al.
2024.
Wukong: Towards a scaling law for large-scale
recommendation.
*arXiv preprint arXiv:2403.02545*
(2024).

Jiang Zhang, Sumit Kumar,
Wei Chang, Yubo Wang,
Feng Zhang, Weize Mao,
Hanchao Yu, Aashu Singh,
Min Li, and Qifan Wang.
2025.
Optimizing Recall or Relevance? A Multi-Task
Multi-Head Approach for Item-to-Item Retrieval in Recommendation. In
*Proceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V. 2*. 5194–5204.

Xiangyu Zhao, Maolin
Wang, Xinjian Zhao, Jiansheng Li,
Shucheng Zhou, Dawei Yin,
Qing Li, Jiliang Tang, and
Ruocheng Guo. 2023.
Embedding in recommender systems: A survey.
*arXiv preprint arXiv:2310.18608*
(2023).

Guorui Zhou, Xiaoqiang
Zhu, Chenru Song, Ying Fan,
Han Zhu, Xiao Ma,
Yanghui Yan, Junqi Jin,
Han Li, and Kun Gai.
2018.
Deep interest network for click-through rate
prediction. In *Proceedings of the 24th ACM SIGKDD
international conference on knowledge discovery & data mining*.
1059–1068.

</div>

[^1]: $`^{\ast}`$Equal Contribution

[^2]: $`^{\dagger}`$Work done while at Meta Platforms
