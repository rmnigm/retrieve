# Introduction

Modern recommender systems have become a cornerstone of the digital ecosystem, driving the success of music streaming services and short-video platforms. Their ability to personalize content directly impacts user engagement and monetization of services. However, the efficacy of these systems is critically dependent on the quality and scale of training data. Historically, recommendation algorithms have evolved from classical collaborative filtering methods to neural architectures such as DSSM and DCN , followed by sequential models based on LSTM and GRU frameworks (e.g., GRU4Rec ). A breakthrough emerged with the adoption of transformers: approaches such as BERT4Rec and SASRec established new accuracy benchmarks, demonstrating the ability to capture long-term dependencies in user sessions.

Empirical studies in adjacent domains, such as computer vision (ViT ) and natural language processing (GPT-3 , Chinchilla ), confirm that scaling data and model parameters is a key driver of performance. For instance, formalized scaling laws, showing that NLP model performance improves polynomially with increased data and computational resources. Similar trends are observed in recommender systems , where industrial solutions routinely leverage terabytes of data—unavailable to the academic community.

A critical barrier to research remains the limited availability of representative datasets. Commercial platforms rarely release raw data due to its strategic value, forcing researchers to rely on outdated (e.g., Netflix Prize ) and relatively small benchmarks (e.g., MovieLens , Steam ). For example, the popular Spotify Million Playlist dataset contains only 1 million playlists, orders of magnitude smaller than real-world industrial scenarios. This creates a gap between academic experiments and practical requirements: models trained on small-scale data often lose efficacy when scaled.

To address this challenge, we introduce Yambda[^1] (**YA**ndex **M**usic **B**illion-interactions **DA**taset) — one of the largest open datasets of music listening interactions, comprising 4.79 billion events (listening events, likes, dislikes, unlikes and undislikes) from 1 million anonymized users and 9.39 million tracks, collected over 11 months. Additionally, the dataset includes track metadata (duration, content embedding, artist, album) and timestamps, ensuring compatibility with modern sequential and context-aware recommendation approaches. The release of Yambda aims to democratize research in recommender systems: lowering barriers to experiment reproducibility, validating hypotheses about model scaling, and developing methods for extreme data sparsity.

The article is structured as follows: Sec. <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a> analyzes limitations of existing datasets, Sec. <a href="#sec:yambda" data-reference-type="ref" data-reference="sec:yambda">3</a> details Yambda’s collection and preprocessing methodology and presents dataset statistics, Sec. <a href="#sec:bench" data-reference-type="ref" data-reference="sec:bench">4</a> defines evaluation process of the baseline models and provide results, and Sec. <a href="#sec:future" data-reference-type="ref" data-reference="sec:future">5</a> discusses implications for research and industry.

# Related Work

This section examines prominent datasets in recommender systems research that capture user-item interactions, highlighting their characteristics and limitations.

<div id="dataset_comparison">

| Dataset                |      Users |      Items | Interactions |
|:-----------------------|-----------:|-----------:|-------------:|
| MovieLens-10M          |        72K |        10K |          10M |
| MovieLens-20M          |       138K |        27K |          20M |
| MovieLens-25M          |       162K |        62K |          25M |
| MovieLens-32M          |       201K |        88K |          32M |
| Steam                  |       2.6M |        32K |           8M |
| Netflix                |       480K |        18K |         100M |
| Amazon Reviews 2013    |       6.6M |       2.4M |          35M |
| Amazon Reviews 2014    |        21M |       9.9M |          83M |
| Amazon Reviews 2018    | <u>44M</u> |        15M |         233M |
| Amazon Reviews 2023    |    **55M** | <u>48M</u> |         572M |
| Criteo 1TB Click Logs  |        N/A |        N/A |           4B |
| Music4All-Onion        |       119K |       109K |         253M |
| LFM-1b                 |       120K |       3.1M |           1B |
| LFM-2b                 |       120K |    **51M** |           2B |
| MLHD                   |       583K |         7M |      **27B** |
|                        |            |            |              |
| Yambda-50M             |        10K |       0.9M |          48M |
| Yambda-500M            |       100K |         3M |         480M |
| Yambda-5B              |         1M |       9.4M |  <u>4.8B</u> |

Dataset Size Comparison

</div>

#### MovieLens 

Available in six variants (100K, 1M, 10M, 20M, 25M, 32M interactions ), this dataset was first released in 1998 and remains widely adopted in academic studies due to its longevity and accessibility.
It records user-provided movie ratings on a 1–5 scale, along with optional tags.
Each interaction includes precise timestamps, and movie metadata includes titles, release years, and genre annotations.
While its simplicity facilitates matrix factorization-based approaches, the limited item pool ( 10,000 movies) renders it unrepresentative of industrial-scale scenarios, where catalogs often exceed millions of items.

#### Steam

Derived from the gaming platform, this dataset contains user reviews of games, including binary recommendations (recommend/not recommend), review helpfulness scores, and daily-level timestamps.
While text reviews provide rich contextual signals, the dataset’s focus on explicit feedback (e.g., reviews posted post-consumption) limits its utility for modeling real-time sequential interactions, a critical requirement for platforms like music or short-video streaming services.

#### Netflix

The iconic dataset from the 2006–2009 Netflix Prize competition comprises anonymized 1–5 star ratings with associated dates.
Despite its historical significance, its sparse temporal granularity (date-only precision) and small item catalog ( 17,000 movies) restrict its applicability to modern sequential recommendation tasks requiring millisecond-level precision.

#### Amazon Reviews

Spanning four iterations since 2013, this large-scale e-commerce dataset includes product reviews, ratings, and extensive metadata (titles, descriptions, categories, prices, images).
Interactions feature millisecond-precision timestamps, crucial for modeling purchase sequences in recommendation scenarios. However, its emphasis on explicit feedback (reviews) may underrepresent implicit signals (e.g., clicks, dwell time), which dominate many real-world systems.

#### Criteo 1TB Click Logs

A terabyte-scale dataset for click-through rate (CTR) prediction, it aggregates ad impressions and clicks.
Although its sheer size (1 billion users, 500 million banners) aligns with industrial needs, the absence of official feature documentation, timestamps, and user/item identifiers hinders reproducibility and limits its utility for sequential or context-aware modeling.

#### Music4All-Onion

This datasets combine content-based features (lyrics, audio spectrogram analyses, video clip embeddings) with listening histories from Last.fm.
While it enables multimodal recommendation research, its limited interaction counts ($`\approx`$<!-- -->250M) and focus on content metadata rather than behavioral sequences reduce its suitability for large-scale sequential modeling.

#### LFM-1b and LFM-2b 

Containing 1 billion and 2 billion interactions respectively, these large-scale datasets include audio features, track metadata, and user demographics.
However, licensing restrictions currently block public access, severely limiting their academic utility.

#### Music Listening Histories Dataset 

This dataset encompasses a comprehensive collection of user listening histories from the Last.fm platform.
While it aggregates data from a substantial user base (583K) and an extensive catalog of tracks (\>7M), it currently faces accessibility limitations and remains unavailable for utilization in recommender systems research

To bridge the gap between academic research and industrial practice, we identify three critical dataset features:

- Large-Scale: Datasets must approach industrial scales (millions of users/items, billions of interactions) to validate scalability and robustness of algorithms like graph neural networks or transformer-based architectures;

- Diverse Feedback Types: Coexistence of implicit (clicks, skips) and explicit (ratings, likes) feedback is essential for modeling complex user behavior, particularly in domains like music streaming where rapid feedback loops are predominant;

- Global Temporal Split: To properly evaluate recommendation algorithms datasets should provide timestamp for each event or global ordering of all events.

Current datasets often lack one or more of these attributes. For instance, MovieLens and Netflix are relatively small and contains only explicit signals, while Criteo’s lack of explicit user and item identifiers makes sequential analysis difficult. By addressing these gaps, our proposed dataset, Yambda, aims to provide a comprehensive resource for the research of next-generation recommender systems.

# The Yambda Dataset

## Dataset Content

<div id="tab:is_organic_distr">

| Event     |         Total | Recommendation |  Ratio |
|:----------|--------------:|---------------:|-------:|
| Listen    | 4,649,567,411 |  2,266,400,808 | 48.74% |
| Like      |    89,334,605 |     37,789,576 | 42.30% |
| Dislike   |    11,579,143 |      5,612,434 | 48.47% |
| Unlike    |    32,944,520 |      1,651,117 |  5.01% |
| Undislike |     2,434,208 |        239,135 |  9.82% |

Recommendation-Driven Events Ratio

</div>

Yandex.Music is a streaming music service that employs recommendation technologies to select relevant tracks in real time and curate personalized daily playlists. The primary user action is track listening (implicit feedback), which includes track length and listening percentage. Additionally, users can express explicit feedback by liking or disliking tracks, as well as canceling these actions (unlike and undislike). Thus, the Yambda dataset consists of five types of user-item interactions: Listen, Like, Dislike, Unlike, and Undislike.

User actions within Yandex.Music can be categorized as organic (where a user independently discovers and interacts with a track) or recommendation-driven (where interactions occur via recommendations from the service’s algorithm). The dataset includes an `is_organic` flag for each event, enabling the distinction between organic and recommendation-driven actions. This feature is a critical aspect of Yambda, as it allows researchers to disentangle the platform’s logging policies from purely recommendation-influenced behavior. Tab. <a href="#tab:is_organic_distr" data-reference-type="ref" data-reference="tab:is_organic_distr">2</a> shows the ratio between organic and recommendation-driven events in the dataset.

Furthermore, Yambda provided neural embeddings derived from convolutional neural network trained in contrastive manner . These embeddings facilitate refined content-based recommendations for musical compositions and enable the application of modern approaches leveraging Semantic IDs .

## Data Availability

<div class="table*">

| Dataset     |     Users |     Items |       Listens |      Likes |   Dislikes |
|:------------|----------:|----------:|--------------:|-----------:|-----------:|
| Yambda-50M  |    10,000 |   934,057 |    46,467,212 |    881,456 |    107,776 |
| Yambda-500M |   100,000 | 3,004,578 |   466,512,103 |  9,033,960 |  1,128,113 |
| Yambda-5B   | 1,000,000 | 9,390,623 | 4,649,567,411 | 89,334,605 | 11,579,143 |

</div>

To address accessibility of the dataset, we released two subsampled variants along with the original one: Yambda-5B (full scale), Yambda-500M (1/10 subsample), and Yambda-50M (1/100 subsample), created by randomly sampling users at corresponding ratios. Tab. <a href="#tab:yambda_sizes" data-reference-type="ref" data-reference="tab:yambda_sizes">[tab:yambda_sizes]</a> summarizes their key statistics.

Each variant is available in two formats:

- *Flat*: events are stored as tuples `(uid, item_id, timestamp, is_organic, ...)` in tabular form, with separate tables per interaction type;

- *Sequential*: user histories are aggregated into timestamp-sorted lists for each interaction type, facilitating sequence-based modeling.

Moreover, for ease of use of the dataset for evaluation, we combined all event types into one `multi_event` file.

Data are distributed using well-proven Apache Parquet format, ensuring compatibility with distributed processing frameworks (e.g., Hadoop, Spark) and modern analytical tools (e.g., Polars, Pandas). Tab. <a href="#tab:file_structure" data-reference-type="ref" data-reference="tab:file_structure">[tab:file_structure]</a> reflects the structure and contents of the files in the dataset.

<div class="table*">

| File | Schema |
|:---|:---|
| `artist_item_mapping.parquet` | <span style="color: violet">artist_id</span>, <span style="color: cyan">item_id</span> |
| `album_item_mapping.parquet` | <span style="color: orange">album_id</span>, <span style="color: cyan">item_id</span> |
| `embeddings.parquet` | <span style="color: cyan">item_id</span>, embed, normalized_embed |
| `{size}/listens.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic, played_ratio_pct, track_length_seconds |
| `{size}/likes.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic |
| `{size}/dislikes.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic |
| `{size}/unlikes.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic |
| `{size}/undislikes.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic |
| `{size}/multi_event.parquet` | <span style="color: red">uid</span>, <span style="color: cyan">item_id</span>, timestamp, is_organic, played_ratio_pct, track_length_seconds, event_type |

</div>

## Acquisition And Processing

To construct the dataset, we defined an approximately 11-month observation period and identified users who performed at least 10 target actions during the first 10 months and at least one action in the subsequent period. Then we randomly sampled 1 million users meeting these criteria and collect their interaction history for the specified period. In compliance with platform policies, all user and track data were anonymized. The dataset contains exclusively numerical identifiers for users, tracks, albums, and artists, along with their relational mappings. Event timestamps were consistently transformed as follows:
``` math
T'_{event} = \left[ \frac{T_{event} - T_{start}}{5} \right] \times 5
```

where $`T_{start}`$ denotes timestamp of the first event in the dataset. This transformation preserves temporal ordering with 5-second precision. Similarly, track length is rounded to 5-seconds precision, and listening percentage available at 1-percent granularity.

## Analysis And Statistics

<figure id="fig:user_hist_len" data-latex-placement="h">
<img src="history_len_2.png" />
<figcaption>User History Length Distribution</figcaption>
</figure>

Our dataset emphasizes user personalization through the analysis of interaction history with the service. Figure <a href="#fig:user_hist_len" data-reference-type="ref" data-reference="fig:user_hist_len">1</a> illustrates the distribution of interaction history lengths across all event types for users during the dataset collection period. The median history length is comparable to the context window of modern large language models (LLMs), such as GPT-3. In addition, Tab. <a href="#tab:event_history_len" data-reference-type="ref" data-reference="tab:event_history_len">3</a> shows that most of the user history consists of implicit feedback (listens).

<figure id="fig:item_hist_len" data-latex-placement="h">
<img src="item_hist_len.png" />
<figcaption>Item History Length Distribution</figcaption>
</figure>

The event distribution across items (Figure <a href="#fig:item_hist_len" data-reference-type="ref" data-reference="fig:item_hist_len">2</a>) exhibits a pronounced imbalance, with a small subset of highly popular platform tracks being distinctly prominent, alongside a substantial long-tail segment of tracks exhibiting minimal (1–2) interactions.

<div id="tab:event_history_len">

| Event     | Median |    p90 |    p95 |
|:----------|-------:|-------:|-------:|
| Listen    |  3,076 | 12,956 | 17,030 |
| Like      |     45 |    269 |    409 |
| Dislike   |      4 |     30 |     60 |
| Unlike    |     15 |    111 |    201 |
| Undislike |      3 |     14 |     22 |

User History Length per Event Type

</div>

# Benchmarking

<div class="table*">

| Dataset | Model | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Coverage@10 | Coverage@100 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Yambda-50M | Random | 0.0000 | 0.0000 | 0.0000 | 0.0002 | **0.1363** | **0.7692** |
|  | MostPop | 0.0186 | 0.0249 | 0.0064 | 0.0321 | 0.0000 | 0.0002 |
|  | DecayPop | 0.0260 | 0.0323 | 0.0122 | 0.0479 | 0.0000 | 0.0002 |
|  | ItemKNN | **0.0781** | **0.0934** | **0.0373** | **0.1297** | 0.0180 | 0.0874 |
|  | iALS | 0.0407 | 0.0642 | 0.0128 | 0.0808 | 0.0044 | 0.0156 |
|  | BPR | 0.0389 | 0.0641 | 0.0139 | 0.0836 | <u>0.0257</u> | <u>0.0999</u> |
|  | SANSA | 0.0069 | 0.0095 | 0.0031 | 0.0137 | 0.0114 | 0.0730 |
|  | SASRec | <u>0.0748</u> | <u>0.0764</u> | <u>0.0325</u> | <u>0.1026</u> | 0.0130 | 0.0310 |
|  |  |  |  |  |  |  |  |
| Yambda-500M | Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.3896** | **0.9928** |
|  | MostPop | 0.0173 | 0.0237 | 0.0060 | 0.0309 | 0.0000 | 0.0001 |
|  | DecayPop | 0.0267 | 0.0340 | 0.0127 | 0.0505 | 0.0000 | 0.0001 |
|  | ItemKNN | <u>0.0708</u> | <u>0.0771</u> | <u>0.0320</u> | <u>0.1049</u> | 0.0257 | <u>0.0917</u> |
|  | iALS | 0.0384 | 0.0621 | 0.0131 | 0.0811 | 0.0022 | 0.0067 |
|  | BPR | 0.0400 | 0.0652 | 0.0137 | 0.0861 | 0.0244 | 0.0758 |
|  | SANSA | — | — | — | — | — | — |
|  | SASRec | **0.0754** | **0.0884** | **0.0336** | **0.1240** | <u>0.0347</u> | 0.0824 |
|  |  |  |  |  |  |  |  |
| Yambda-5B | Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.8207** | **1.0000** |
|  | MostPop | 0.0175 | 0.0239 | 0.0062 | 0.0313 | 0.0000 | 0.0000 |
|  | DecayPop | 0.0271 | 0.0348 | 0.0130 | 0.0509 | 0.0000 | 0.0000 |
|  | ItemKNN | — | — | — | — | — | — |
|  | iALS | 0.0388 | 0.0628 | 0.0132 | 0.0816 | 0.0010 | 0.0027 |
|  | BPR | <u>0.0408</u> | <u>0.0664</u> | <u>0.0141</u> | <u>0.0870</u> | 0.0179 | <u>0.0508</u> |
|  | SANSA | — | — | — | — | — | — |
|  | SASRec | **0.0647** | **0.0847** | **0.0289** | **0.1214** | <u>0.0212</u> | 0.0456 |

</div>

<div class="table*">

| Dataset | Model | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Coverage@10 | Coverage@100 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Yambda-50M | Random | 0.0000 | 0.0000 | 0.0000 | 0.0003 | **0.3677** | **0.9901** |
|  | MostPop | 0.0046 | 0.0097 | 0.0083 | 0.0222 | 0.0001 | 0.0006 |
|  | DecayPop | **0.0180** | **0.0269** | **0.0333** | **0.0651** | 0.0001 | 0.0006 |
|  | ItemKNN | <u>0.0125</u> | <u>0.0251</u> | <u>0.0199</u> | <u>0.0648</u> | <u>0.1050</u> | <u>0.3922</u> |
|  | iALS | 0.0078 | 0.0224 | 0.0103 | 0.0568 | 0.0110 | 0.0554 |
|  | BPR | 0.0056 | 0.0187 | 0.0073 | 0.0526 | 0.0464 | 0.1449 |
|  | SANSA | 0.0068 | 0.0203 | 0.0105 | 0.0616 | 0.0194 | 0.1601 |
|  | SASRec | 0.0100 | 0.0208 | 0.0175 | 0.0603 | 0.0211 | 0.0712 |
|  |  |  |  |  |  |  |  |
| Yambda-500M | Random | 0.0000 | 0.0000 | 0.0000 | 0.0002 | **0.7272** | **1.0000** |
|  | MostPop | 0.0028 | 0.0087 | 0.0033 | 0.0215 | 0.0000 | 0.0002 |
|  | DecayPop | **0.0174** | **0.0279** | **0.0285** | <u>0.0736</u> | 0.0000 | 0.0002 |
|  | ItemKNN | 0.0101 | <u>0.0258</u> | 0.0151 | 0.0701 | <u>0.1112</u> | <u>0.3182</u> |
|  | iALS | 0.0050 | 0.0209 | 0.0064 | 0.0596 | 0.0034 | 0.0131 |
|  | BPR | 0.0071 | 0.0234 | 0.0101 | 0.0655 | 0.0628 | 0.1659 |
|  | SANSA | 0.0077 | 0.0252 | 0.0090 | 0.0681 | 0.0213 | 0.1367 |
|  | SASRec | <u>0.0125</u> | 0.0237 | <u>0.0200</u> | **0.0890** | 0.0392 | 0.1047 |
|  |  |  |  |  |  |  |  |
| Yambda-5B | Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.9838** | **1.0000** |
|  | MostPop | 0.0026 | 0.0084 | 0.0035 | 0.0238 | 0.0000 | 0.0000 |
|  | DecayPop | **0.0165** | <u>0.0267</u> | **0.0281** | <u>0.0733</u> | 0.0000 | 0.0000 |
|  | ItemKNN | — | — | — | — | — | — |
|  | iALS | 0.0052 | 0.0207 | 0.0076 | 0.0626 | 0.0012 | 0.0039 |
|  | BPR | 0.0071 | 0.0245 | 0.0106 | 0.0726 | <u>0.0463</u> | <u>0.1141</u> |
|  | SANSA | — | — | — | — | — | — |
|  | SASRec | <u>0.0136</u> | **0.0348** | <u>0.0217</u> | **0.1026** | 0.0224 | 0.0547 |

</div>

To establish a foundation for utilizing this dataset as a benchmark for novel approaches in recommender systems, we implemented several baseline algorithms and proposed an offline evaluation procedure closely aligned with industrial recommendation system practices. The code is available in our GitHub repository.

## Evaluation Process

Typically, academic studies report offline evaluations due to the lack of access to online experiments. Currently, two offline evaluation schemes are most prevalent:

- Leave-One-Out (LOO), where the last positive interaction of each user is excluded from the training set and used for prediction. A key limitation of this approach is the violation of temporal dependencies, as the model may inadvertently utilize more recent interactions during training than those present in the test set;

- Global Temporal Split (GTS), where the entire dataset is partitioned into training and test sets based on timestamps. While this preserves temporal consistency, it risks including users absent from the training data, whose interaction histories are unknown at the start of the test period.

To better approximate real-world production environments, we adopted a modified GTS scheme:

- *Train*:  300 days

- *Gap*: 30 minutes

- *Test*: 1 day

A 30-minute gap between training and test sets was introduced to exclude interactions used neither for training nor evaluation. This mimics the latency between model training and deployment in industrial systems. Additionally, to ensure consistency, users with empty interaction histories at the start of the test period were discarded.

All model parameters and user states were frozen at the beginning of the test period. A trade-off exists here: true real-time recommendation systems update user states with delays of tens of seconds, but replicating this would significantly complicate evaluation and increase computational costs. By limiting the test period to one day and freezing user states, we approximate systems with daily offline updates for model and user state refreshes—a reasonable compromise for practical benchmarking.

## Baselines

We selected well-established algorithms as baselines:

- *MostPop*: Recommendations based on overall item popularity.

- *DecayPop* : A time-decayed variant of *MostPop*, better utilizing emergent popularity.

- *ItemKNN* is a neighborhood-based collaborative filtering approach that generates recommendations by identifying items similar to those a user has previously interacted with. It employs similarity metrics (e.g., cosine or Pearson correlation) to compute item-item affinity, leveraging the assumption that users prefer items analogous to their historical preferences.

- *iALS* is a matrix factorization method optimized for implicit feedback datasets. It decomposes the user-item interaction matrix into latent user and item factors, regularized to prevent overfitting. The alternating least squares solver ensures scalability and efficiency, making it suitable for large-scale recommendation tasks.

- *BPR* is a pairwise ranking optimization framework designed for implicit feedback. It learns personalized rankings by maximizing the posterior probability that a user prefers observed items over unobserved ones. BPR employs a triplet loss to model user preferences through stochastic gradient descent, emphasizing personalized ordinal relationships.

- *SANSA* is a scalable variant of EASE framework, designed to address computational bottlenecks in large-item datasets. By relaxing the symmetry constraint of EASE’s item-item similarity matrix and introducing approximate training strategies, SANSA reduces time and memory complexity while retaining high recommendation accuracy, making it practical for industrial-scale applications.

- *SASRec* is a transformer-based sequential model that leverages self-attention to model user behavior sequences. It dynamically assigns attention weights to past interactions, capturing temporal dependencies and highlighting influential items. SASRec excels in scenarios requiring precise modeling of evolving user preferences over time.

## Results

For the analysis of the algorithms, we employed well-established metrics, including NDCG@$`k`$ ($`k\in\{10,100\}`$) to assess ranking quality, Recall@$`k`$ to evaluate candidate generation performance, and Coverage@$`k`$ to quantify the recommendation system’s comprehensiveness in representing the item collection. For our evaluation Coverage is calculated using the following formula:
``` math
\text{Coverage@}k = \frac{|\bigcup_{u \in U}{R(u, k))}|}{|I|},
```
where $`U`$ and $`I`$ denote sets of users and items respectively, $`R(\cdot, \cdot)`$ is a ranking function which returns top relevant items for the specified user.

We evaluated algorithm performance under two feedback setups: positive listening events, or Listen$`_+`$ (implicit feedback) and Like (explicit feedback). To produce Listen$`_+`$ we used 50% of the track duration as the listening threshold.
Additionally, we tested the algorithms on smaller subsets of the dataset. Hyperparameters for DecayPop, ItemKNN, iALS, BPR, and SANSA were tuned using the GTS scheme. For SASRec, hyperparameter optimization was deferred due to computational constraints. To find optimal hyperparameters, we reserved one day from the training set for validation, maintaining the 30-minute gap for consistency. The NDCG metric, widely used for ranking evaluation, guided the optimization process via the OPTUNA  framework.

Final results, obtained by training models on the full training set with tuned hyperparameters, are summarized in Tab. <a href="#tab:eval_le" data-reference-type="ref" data-reference="tab:eval_le">[tab:eval_le]</a> and Tab. <a href="#tab:eval_like" data-reference-type="ref" data-reference="tab:eval_like">[tab:eval_like]</a>. Metrics for ItemKNN and SANSA are unavailable at larger dataset scales due to their computational intractability within practical time constraints. We observed that the top-2 algorithms by ranking metrics in the Listen$`_+`$ scenario consistently included ItemKNN and SASRec across all dataset scales. Conversely, the DecayPop algorithm demonstrated superior performance in the Like scenario, establishing itself as the most effective ranking method despite its simplicity.

# Conclusions And Future Work

A large-scale dataset containing 4.79 billion user-item interactions has been publicly released to advance research in recommender systems. This resource is distinguished by three critical features: (1) high-fidelity audio embeddings derived from spectral analysis of tracks, enabling content-aware recommendation approaches; (2) a binary is_organic flag annotating interactions to differentiate organic user behavior from algorithmically influenced actions; and (3) timestamp granularity aligned with the Global Temporal Split evaluation protocol, which was rigorously implemented to prevent temporal data leakage during baseline model benchmarking.

Methodologically, the dataset’s design addresses longstanding limitations in academic research by mirroring industrial-scale conditions. The evaluation framework employed in this work demonstrates that conventional collaborative filtering methods (e.g., Matrix Factorization) exhibit degraded performance when applied to scenarios requiring real-time interaction processing, highlighting the necessity of sequence-aware architectures like Transformers.
Future Directions

Three primary research avenues are proposed to extend this work:

#### Enhanced Temporal Evaluation Protocol

The Global Temporal Split scheme will be refined to better emulate industrial recommender systems through:

- Incremental user state updates via streaming data pipelines;

- Periodic retraining of lightweight algorithms (e.g., iALS, BPR) at fixed intervals;

- Simulation of cold-start scenarios through dynamic user/item inclusion.

#### Multi-Modal Recommendation Paradigms

The dataset’s multi-modal nature—combining behavioral sequences, audio embeddings, and track metadata—will be leveraged to investigate:

- Cross-modal fusion techniques for hybrid recommender architectures;

- Graph neural networks exploiting artist-album-track relationships;

- Contrastive learning frameworks aligning audio features with user preferences.

#### Organic vs. Algorithmic Interaction Analysis

A comprehensive analysis of behavioral differences between organic and recommendation-driven interactions will be conducted, focusing on:

- Temporal patterns in user engagement decay post-recommendation;

- Bias propagation in feedback loops induced by algorithmic curation;

- Metrics quantifying the diversity and serendipity of organic discovery.

This dataset is positioned to serve as a foundational resource for bridging the gap between academic experimentation and industrial deployment. By democratizing access to web-scale interaction data while preserving privacy through rigorous anonymization, the work aims to catalyze innovation in sequential recommendation, fairness-aware algorithms, and multi-modal personalization. All artifacts, including preprocessing code and baseline implementations, have been standardized to ensure reproducibility across experimental setups.

<div class="thebibliography">

37

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*. 2623–2631.

Newsha Ardalani, Carole-Jean Wu, Zeliang Chen, Bhargav Bhushanam, and Adnan Aziz. 2022.
Understanding scaling laws for recommendation models.
*arXiv preprint arXiv:2208.08489* (2022).

James Bennett and Stan Lanning. 2007.
The netflix prize.
(2007).

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.
Language models are few-shot learners.
*Advances in neural information processing systems* 33 (2020), 1877–1901.

Ching-Wei Chen, Paul Lamere, Markus Schedl, and Hamed Zamani. 2018.
Recsys challenge 2018: Automatic music playlist continuation. In *Proceedings of the 12th ACM Conference on Recommender Systems*. 527–528.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014.
Empirical evaluation of gated recurrent neural networks on sequence modeling.
*arXiv preprint arXiv:1412.3555* (2014).

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020.
An image is worth 16x16 words: Transformers for image recognition at scale.
*arXiv preprint arXiv:2010.11929* (2020).

GroupLens. \[n. d.\].
MovieLens.
<https://grouplens.org/datasets/movielens/>.
Accessed: (2025-05-28).

F Maxwell Harper and Joseph A Konstan. 2015.
The movielens datasets: History and context.
*Acm transactions on interactive intelligent systems (tiis)* 5, 4 (2015), 1–19.

Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015.
Session-based recommendations with recurrent neural networks.
*arXiv preprint arXiv:1511.06939* (2015).

Sepp Hochreiter and Jürgen Schmidhuber. 1997.
Long short-term memory.
*Neural computation* 9, 8 (1997), 1735–1780.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022.
Training compute-optimal large language models.
*arXiv preprint arXiv:2203.15556* (2022).

Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024.
Bridging Language and Items for Retrieval and Recommendation.
*arXiv preprint arXiv:2403.03952* (2024).

Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008.
Collaborative filtering for implicit feedback datasets. In *2008 Eighth IEEE international conference on data mining*. Ieee, 263–272.

Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013.
Learning deep structured semantic models for web search using clickthrough data. In *Proceedings of the 22nd ACM international conference on Information & Knowledge Management*. 2333–2338.

Yitong Ji, Aixin Sun, Jie Zhang, and Chenliang Li. 2020.
A re-visit of the popularity baseline in recommender systems. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 1749–1752.

Wang-Cheng Kang and Julian McAuley. 2018.
Self-attentive sequential recommendation. In *2018 IEEE international conference on data mining (ICDM)*. IEEE, 197–206.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.
Scaling laws for neural language models.
*arXiv preprint arXiv:2001.08361* (2020).

Criteo AI Lab. \[n. d.\].
Criteo 1TB Click Logs Dataset.
<https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset>.
Accessed: (2025-05-12).

Julian McAuley and Jure Leskovec. 2013.
Hidden factors and hidden topics: understanding rating dimensions with review text. In *Proceedings of the 7th ACM conference on Recommender systems*. 165–172.

Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015.
Image-based recommendations on styles and substitutes. In *Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval*. 43–52.

Marta Moscati, Emilia Parada-Cabaleiro, Yashar Deldjoo, Eva Zangerle, and Markus Schedl. 2022.
Music4All-Onion–A Large-Scale Multi-faceted Content-Centric Music Recommendation Dataset. In *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*. 4339–4343.

Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019.
Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)*. 188–197.

Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023.
Recommender systems with generative retrieval.
*Advances in Neural Information Processing Systems* 36 (2023), 10299–10315.

Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2012.
BPR: Bayesian personalized ranking from implicit feedback.
*arXiv preprint arXiv:1205.2618* (2012).

Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001.
Item-based collaborative filtering recommendation algorithms. In *Proceedings of the 10th international conference on World Wide Web*. 285–295.

Markus Schedl. 2016.
The lfm-1b dataset for music retrieval and recommendation. In *Proceedings of the 2016 ACM on international conference on multimedia retrieval*. 103–110.

Markus Schedl, Stefan Brandl, Oleg Lesota, Emilia Parada-Cabaleiro, David Penz, and Navid Rekabsaz. 2022.
LFM-2b: A dataset of enriched music listening events for recommender systems research and fairness analysis. In *Proceedings of the 2022 Conference on Human Information Interaction and Retrieval*. 337–341.

Janne Spijkervet and John Ashley Burgoyne. 2021.
Contrastive learning of musical representations.
*arXiv preprint arXiv:2103.09410* (2021).

Martin Spišák, Radek Bartyzal, Antonı́n Hoskovec, Ladislav Peska, and Miroslav Tŭma. 2023.
Scalable approximate nonsymmetric autoencoder for collaborative filtering. In *Proceedings of the 17th ACM Conference on Recommender Systems*. 763–770.

Harald Steck. 2019.
Embarrassingly shallow autoencoders for sparse data. In *The World Wide Web Conference*. 3251–3257.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019.
BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 1441–1450.

Gabriel Vigliensoni and Ichiro Fujinaga. 2017.
The music listening histories dataset.. In *ISMIR*. 96–102.

Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017.
Deep & cross network for ad click predictions.
In *Proceedings of the ADKDD’17*. 1–7.

Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021.
Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In *Proceedings of the web conference 2021*. 1785–1797.

Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, et al. 2024b.
Wukong: Towards a scaling law for large-scale recommendation.
*arXiv preprint arXiv:2403.02545* (2024).

Gaowei Zhang, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. 2024a.
Scaling law of large sequential recommendation models. In *Proceedings of the 18th ACM Conference on Recommender Systems*. 444–453.

</div>

[^1]: <https://huggingface.co/datasets/yandex/yambda>
