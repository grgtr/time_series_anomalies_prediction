### Abstract

Time series data is ubiquitous across numerous domains, from financial markets and industrial monitoring to medical diagnostics and environmental sensing. The continuous nature and often complex temporal dependencies within these datasets make anomaly detection a particularly challenging yet crucial task. Anomalies, representing deviations from expected patterns, often signify critical events such as system malfunctions, fraudulent activities, or emergent phenomena, necessitating robust and timely identification. Traditional anomaly detection methods frequently struggle with the inherent complexity and high dimensionality of real-world time series, often failing to adequately capture intricate temporal dynamics or adapt to evolving patterns.

In this paper, we propose a novel deep learning framework for unsupervised time series anomaly detection, leveraging the strengths of deep temporal contrastive learning and the mathematical rigor of signature methods. Our approach, building upon the principles of Deep Temporal Contrastive Clustering (DTCC) [[arXiv:2212.14366v1](https://arxiv.org/abs/2212.14366v1)], aims to learn discriminative and clustering-friendly representations from time series data, where anomalous instances are implicitly driven apart from normal patterns in the learned embedding space. We explore various data preprocessing strategies, including direct time series windows and different formulations of path signatures, specifically focusing on the full signature and a novel "streamed" signature approach where the signature is computed up to each time step. These signature transformations, rooted in rough path theory [[arXiv:1603.03788v1](https://arxiv.org/abs/1603.03788v1)], capture essential geometric and algebraic properties of time series paths, providing a rich, high-dimensional representation that is invariant to time reparameterization. Our experimental evaluation, conducted on synthetic datasets exhibiting diverse stochastic processes and real-world benchmarks from the UCR Time Series Classification Archive, demonstrates the efficacy of our proposed methods in isolating anomalous time series windows. This work contributes to advancing unsupervised anomaly detection in complex time series environments by effectively combining state-of-the-art deep contrastive learning with powerful topological data representations.

### 1 Introduction

Time series analysis plays a pivotal role in a multitude of scientific and engineering disciplines, offering insights into dynamic systems and predicting future states. From monitoring critical infrastructure to analyzing complex biological signals, the ability to discern normal behavior from anomalous occurrences is paramount. Anomalies in time series data, often defined as patterns that do not conform to an expected normal behavior, can indicate significant events such as equipment failure, cyber intrusions, or abrupt changes in environmental conditions. The timely and accurate detection of these anomalies is crucial for maintaining system integrity, preventing catastrophic events, and enabling proactive decision-making.

The inherent characteristics of time series data—including high dimensionality, complex temporal dependencies, and non-stationary dynamics—present significant challenges for effective anomaly detection. Traditional statistical methods, while foundational, often rely on strong assumptions about data distribution or stationarity, which rarely hold true in real-world scenarios. Machine learning approaches, particularly deep learning, have shown promise in addressing these challenges by automatically learning intricate patterns from raw data. However, many existing deep learning models for anomaly detection are supervised or semi-supervised, requiring labeled anomaly data which is often scarce or expensive to obtain. Unsupervised methods, which learn from unlabeled data, are thus highly desirable for practical applications.

This paper addresses the problem of unsupervised time series anomaly detection by proposing a novel framework that integrates deep temporal contrastive learning with advanced feature engineering using signature methods. Our approach is inspired by the recent advancements in self-supervised learning, particularly contrastive learning, which has demonstrated remarkable success in learning robust representations without explicit labels. By extending the principles of Deep Temporal Contrastive Clustering (DTCC) [[arXiv:2212.14366v1](https://arxiv.org/abs/2212.14366v1)], we aim to leverage the inherent structure of time series data to learn embeddings where normal and anomalous patterns are naturally segregated.

Furthermore, we incorporate the theory of path signatures, a powerful mathematical tool from rough path theory [[arXiv:1603.03788v1](https://arxiv.org/abs/1603.03788v1)], to enhance the feature representation of time series. The signature of a path provides a concise, infinite-dimensional summary of its geometric and algebraic properties, offering a robust representation that is invariant to time reparameterization and sensitive to the order and interaction of features. By integrating signature features, we aim to capture richer and more discriminative information from time series segments, thereby improving the efficacy of the anomaly detection process.

The main contributions of this paper are:

*   **A novel deep learning framework for unsupervised time series anomaly detection** that combines deep temporal contrastive learning with advanced signature-based feature engineering.
*   **Adaptation of the Deep Temporal Contrastive Clustering (DTCC) model** to encourage the separation of anomalous and normal time series patterns in the learned embedding space.
*   **Exploration of various signature methods for time series preprocessing**, including classical full signatures and a new "streamed" signature approach where the signature is computed up to each time step, providing a richer temporal context.
*   **Comprehensive experimental evaluation** on synthetic datasets generated from diverse stochastic processes and real-world time series benchmarks, demonstrating the superiority of our proposed framework in identifying anomalies.

The remainder of this paper is organized as follows: Section 2 provides background on signature methods and reviews related work in time series anomaly detection. Section 3 details the proposed methodology, including the DTCC adaptation and signature-based feature extraction. Section 4 presents the experimental setup, datasets, and results. Finally, Section 5 concludes the paper and outlines future research directions.

---

### 2 Related Work

#### 2.1 Anomaly Detection in Time Series

Anomaly detection in time series has been a long-standing research area with diverse applications. Methods can broadly be categorized into statistical, machine learning (including deep learning), and signature-based approaches.

*   **Statistical Methods:** Traditional approaches like ARIMA, Exponential Smoothing, and Control Charts rely on statistical models to characterize normal behavior and flag deviations. While effective for simple, univariate series, they often struggle with multivariate, high-dimensional, and non-stationary data.
*   **Machine Learning Approaches:** These include methods like Isolation Forest, One-Class SVM, and clustering algorithms (e.g., K-Means). These models learn decision boundaries or clusters from normal data, identifying points far from these structures as anomalies. Deep learning has significantly advanced this field, with Autoencoders, LSTMs, and GANs being widely used. Autoencoders learn a compressed representation of normal data, and high reconstruction errors indicate anomalies. RNN-based models like LSTMs can capture temporal dependencies, predicting future values and flagging deviations as anomalies. Generative Adversarial Networks (GANs) can learn the data distribution to generate normal samples, with anomalies identified by their low probability under the learned distribution. More recently, self-supervised learning, particularly contrastive learning, has shown promise in learning robust representations for anomaly detection without explicit anomaly labels.
*   **Signature-based Methods:** The signature of a path, a concept from rough path theory, has gained traction in time series analysis due to its ability to capture the geometric and algebraic properties of paths in a way that is robust to time reparameterization. Signature features have been successfully applied in various machine learning tasks, including classification, regression, and to a lesser extent, anomaly detection. The inherent sensitivity of signatures to the "shape" of a path makes them particularly suitable for identifying subtle changes that might signify an anomaly.

#### 2.2 Signature Methods for Time Series Analysis

The signature method, originating from the theory of rough paths, provides a powerful non-parametric tool for feature extraction from sequential data. As detailed in the seminal work by Lyons [[arXiv:1603.03788v1](https://arxiv.org/abs/1603.03788v1)], the signature of a path is an infinite-dimensional series of iterated integrals that uniquely characterizes the path up to time reparameterization. For a path $X: [a, b] \rightarrow \mathbb{R}^d$, its signature $S(X)_{a,b}$ is a collection of real numbers given by:

$$
S(X)_{a,b} = (1, S(X)^1_{a,b}, \ldots, S(X)^d_{a,b}, S(X)^{1,1}_{a,b}, \ldots, S(X)^{d,d}_{a,b}, \ldots)
$$

where each term $S(X)^{i_1, \ldots, i_k}_{a,b}$ is a $k$-fold iterated integral:
$$
S(X)^{i_1, \ldots, i_k}_{a,b} = \int_{a < t_1 < \ldots < t_k < b} dX_{t_1}^{i_1} \ldots dX_{t_k}^{i_k}
$$
Here, $dX^i$ denotes the increment of the $i$-th coordinate of the path $X$.

**Key properties of the signature include:**

*   **Uniqueness:** The signature uniquely determines a path of bounded variation up to time reparameterization. This means that if two paths have the same signature, they are essentially the same path, just possibly traversed at different speeds.
*   **Time Re-parameterization Invariance:** The signature is invariant under reparameterizations of time. This is a crucial property for time series analysis, as it makes the features robust to variations in sampling rate or speed of events.
*   **Completeness:** The signature captures all information about the path, including its geometric and algebraic properties. Lower-level terms capture basic information like total displacement (first level), while higher-level terms capture more complex interactions and "Lévy area" (second level), which quantifies the signed area enclosed by the path and its chord.
*   **Algebraic Structure (Chen's Identity & Shuffle Product):** The signature possesses a rich algebraic structure. Chen's identity states that the signature of a concatenated path is the tensor product of the individual path signatures, providing a way to combine information from segments. The shuffle product provides a way to express products of signature terms as sums of other signature terms, highlighting redundancies and allowing for efficient computation.

In practical applications, the signature is typically truncated to a certain level, providing a finite-dimensional feature vector. The choice of truncation level depends on the complexity of the patterns one wishes to capture and computational constraints. The input data stream is first embedded into a continuous path, often by adding time as an extra dimension (e.g., $(t, X_t)$), and then its signature is computed. This transforms raw time series data into a set of features that can be used in various machine learning tasks.

---

### 3 Proposed Methodology

Our proposed framework for unsupervised time series anomaly detection integrates a deep temporal contrastive learning model with signature-based feature extraction. This section details the architecture, loss functions, and the various signature preprocessing strategies employed.

#### 3.1 Deep Temporal Contrastive Clustering (DTCC) for Anomaly Detection

We adapt the Deep Temporal Contrastive Clustering (DTCC) model [[arXiv:2212.14366v1](https://arxiv.org/abs/2212.14366v1)] as the core deep learning component. The DTCC model is designed to learn discriminative and clustering-friendly representations from time series data in a self-supervised manner. The key idea is to encourage instances belonging to the same (latent) cluster to have similar representations while pushing representations of different instances/clusters apart. In the context of anomaly detection, our objective is that normal time series segments form tight clusters in the learned embedding space, while anomalous segments, being inherently different, are either isolated or form distinct, smaller clusters.

The DTCC architecture consists of two identical temporal auto-encoders that process two parallel "views" of the input time series: an original view and an augmented view.

*   **Temporal Auto-encoders:** Each auto-encoder comprises a bidirectional multi-layer dilated Recurrent Neural Network (RNN) as the encoder and a single-layer RNN as the decoder. The encoder maps the input time series into a latent representation, capturing temporal dynamics and multi-scale characteristics. The decoder then reconstructs the input from this latent representation.
*   **Data Augmentation:** To generate the augmented view, we apply time-series-specific augmentations (e.g., jittering, scaling, permutation) to the original time series. This creates positive pairs (original, augmented) for contrastive learning.
*   **Clustering Layer:** A soft k-means objective is integrated to guide the learning of clustering-friendly representations. This objective encourages the latent space to form distinct clusters.
*   **Dual Contrastive Learning:** DTCC employs two levels of contrastive learning:
    *   **Instance-level Contrastive Loss:** This aims to maximize the similarity between the latent representations of an original sample and its augmented counterpart (positive pair) while minimizing similarity with all other samples in the batch (negative pairs). This pushes individual instances with similar underlying patterns closer in the latent space.
    *   **Cluster-level Contrastive Loss:** This component ensures consistency between the cluster assignments derived from the original and augmented views. It encourages the cluster representations (prototypes) to be similar for corresponding clusters across views.

**Loss Function:** The overall loss function for DTCC is a unified objective that combines:

1.  **Reconstruction Loss ($\mathcal{L}_{\text{reconstruction}}$):** Measures the fidelity of the auto-encoders in reconstructing the input time series from their latent representations. This helps preserve the essential information of the time series.
2.  **K-Means Objective Loss ($\mathcal{L}_{\text{km}}$):** A spectral clustering-inspired term that promotes learning of compact and well-separated clusters in the latent space.
3.  **Instance-level Contrastive Loss ($\mathcal{L}_{\text{instance}}$):** Enforces discriminative learning by pushing positive pairs closer and negative pairs further apart.
4.  **Cluster-level Contrastive Loss ($\mathcal{L}_{\text{cluster}}$):** Aligns cluster assignments across different views, promoting robust cluster structures.
5.  **Cluster Alignment Loss ($\mathcal{L}_{\text{cluster-alignment}}$):** Minimizes the discrepancy between cluster probability distributions for original and augmented representations, further enforcing consistent clustering.
6.  **Entropy Regularization ($\mathcal{L}_{\text{entropy-reg}}$):** Encourages a balanced distribution of samples across clusters, preventing degenerate solutions where all samples are assigned to a single cluster.

In the context of anomaly detection, the auto-encoder's ability to reconstruct normal patterns, combined with the contrastive learning objectives, drives the model to learn a compact and well-defined representation of "normalcy." Anomalous instances, by their nature, will either have high reconstruction errors, fall outside dense normal clusters in the latent space, or produce inconsistent cluster assignments across augmented views, thus signaling an anomaly.

#### 3.2 Signature-based Feature Extraction

To enrich the time series representation and provide a more robust input to the DTCC model, we explore various signature-based feature extraction methods. These methods transform raw time series data into features that capture crucial geometric and algebraic properties, offering invariance to time reparameterization and sensitivity to sequential interactions.

The time series $X = \{X_t\}_{t \in [0, T]}$ is first transformed into a path by augmenting it with time: $P_t = (t, X_t)$. This ensures that the path is non-degenerate and allows the signature to capture both the values of the series and their temporal ordering.

We investigate two main approaches for integrating signatures:

##### 3.2.1 Full Signature on Windows

In this approach, time series are first segmented into overlapping windows. For each window, the full signature up to a certain level is computed.

*   **Windowing:** The raw time series data is divided into fixed-size, overlapping segments or "windows" (e.g., using a sliding window approach with a defined window size and stride). Each window is treated as an independent time series instance.
*   **Signature Computation per Window:** For each window, the path $P_t$ (augmented with time) is constructed. The signature $S(P)_{a,b}$ is then computed up to a chosen truncation level $L$. This results in a fixed-length feature vector for each window.
*   **Normalization (Optional):** The signature features can be further normalized (e.g., using StandardScaler) to ensure consistent scaling across features.

This method provides a powerful, compact representation for each window, capturing its local dynamics and shape. The resulting signature features (e.g., $S(X)^{1,2}_{a,b}$ and $S(X)^{2,1}_{a,b}$ which relate to the Lévy area as discussed in [[arXiv:1603.03788v1](https://arxiv.org/abs/1603.03788v1)] Section 1.2.4) inherently encode complex interactions within the time series window.

##### 3.2.2 Streamed Signature (Full Path Signature at Each Time Step)

This novel approach aims to capture the evolving signature of the entire time series up to each time step, providing a rich, high-dimensional input for the recurrent auto-encoders. Instead of computing one signature per window, we compute a "stream" of signatures, where each element in the sequence of window corresponds to the signature of the path from the beginning of the time series up to the end of the current window.

*   **Path Construction for Full Series:** For each original time series $X$, a single path $P_t = (t, X_t)$ is constructed for its entire duration.
*   **Evolving Signature Feature:** For each window ending at time $t_{end}$, we compute the full signature $S(P)_{0, t_{end}}$ of the path from its start ($t=0$) up to $t_{end}$. This is repeated for every possible $t_{end}$ that aligns with the end of a window.
*   **Windowed Sequence of Signatures:** The input to the DTCC model for a given window is then a sequence of these "evolving signature" features. For a window spanning times $[t_{start}, t_{end}]$, the input would be $[S(P)_{0, t_1}, S(P)_{0, t_2}, \ldots, S(P)_{0, t_{end}}]$ where $t_i$ are the time points within the window. This provides a deep context of how the path has evolved up to that point.

This "streamed" signature approach is designed to capture not only the local dynamics within a window but also the global context of the time series' history up to that point. The sequence of signatures provides a more granular and temporally aware input, potentially allowing the recurrent auto-encoder to learn more sophisticated representations.

##### 3.2.3 Other Signature Variants

We also consider other variants of the signature:

*   **Log Signature:** The log signature, as described in [[arXiv:1603.03788v1](https://arxiv.org/abs/1603.03788v1)] Section 1.3.5, offers a more compact representation of path information. It is the logarithm of the signature in the algebra of formal power series and can capture essential geometric information, such as Lévy area, in its lower-order terms.
*   **Randomized Signature (R-Sig):** This method provides a randomized approximation of the full signature, offering a trade-off between computational cost and expressive power. It projects the path into a lower-dimensional space using random matrices, potentially enabling faster processing for very high-dimensional signatures.

These signature features are used as input to the DTCC encoder. The choice of signature level, normalization, and windowing parameters are crucial hyper-parameters that are tuned during experimentation.
