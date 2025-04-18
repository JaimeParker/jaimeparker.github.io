---
title: "Transformer: Attention is All You Need"
categories: tech
tags: [Large Language Models]
use_math: true
---

## 1. Overall Architecture

The Transformer is a sequence-to-sequence neural architecture based entirely on self-attention mechanisms, without recurrence or convolution. It consists of two main parts:

- **Encoder**: Generates contextualized representations of input sequences.
- **Decoder**: Autoregressively generates output sequences based on encoder output and previously generated tokens.

<img src="https://picx.zhimg.com/70/v2-7be8fe269991a236f000168291481c8b_1440w.avis?source=172ae18b&biz_tag=Post" 
     alt="Transformer Architecture" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Transformer Architecture</em></p>

<b>Step 1: Input Embedding</b>

Input takes word embedding and position embedding, and adds them together. The result is passed to the encoder.

$$
\text{Input} = \text{Embedding}(x) + \text{PositionEmbedding}(x)
$$

<img src="https://pic2.zhimg.com/v2-7dd39c44b0ae45d31a3ae7f39d3f883f_1440w.jpg" 
     alt="Transformer Input Embedding" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Transformer Input Embedding</em></p>

<b>Step 2: Encoder</b>

Input vector is represented as a matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is the sequence length and $d_{\text{model}}$ is the embedding dimension. The encoder consists of $N$ identical layers, each with two sub-layers:

1. Multi-head self-attention (MHA)
2. Feed-forward network (FFN)

<img src="https://pic1.zhimg.com/v2-45db05405cb96248aff98ee07a565baa_1440w.jpg" 
     alt="Encoder" 
     style="width: 50%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Encoder Process</em></p>

<b>Step 3: Decoder</b>

The decoder also consists of $N$ identical layers, each with three sub-layers:
1. Masked multi-head self-attention (MHA)
2. Multi-head self-attention (MHA) with encoder output as keys and values
3. Feed-forward network (FFN)

The use of Masked MHA ensures that the prediction for a given position depends only on the known outputs at previous positions.

<img src="https://picx.zhimg.com/v2-5367bd47a2319397317562c0da77e455_1440w.jpg" 
     alt="Decoder Prediction" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Decoder Prediction</em></p>

<b>Step 4: Output</b>

The decoder output is passed through a linear layer followed by a softmax function to produce the final output probabilities for the next token in the sequence.

$$
\text{Output} = \text{softmax}(\text{DecoderOutput} \times W_{\text{embedding}}^T)
$$

Where $W_{\text{embedding}}$ is the learned embedding matrix.
The softmax function converts the logits into probabilities for each token in the vocabulary.

## 2. Input of Transformer

<img src="https://pic2.zhimg.com/v2-b0a11f97ab22f5d9ebc396bc50fa9c3f_1440w.jpg" 
     alt="Transformer Input" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Transformer Input</em></p>

### 2.1 Word Embedding

- Input tokens are mapped to embedding vectors using learned embeddings.
- Embedding dimension: $d_{\text{model}}$ (typically 512).

### 2.2 Positional Embedding

- Added to embeddings to inject positional information.
- Fixed sinusoidal positional embeddings used:

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

## 3. Self-Attention

<figure class="align-center">
  <img src="/assets/images/multi_head_attentioin.png" alt="MHA" style="width: 100%;">
  <figcaption>(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.</figcaption>
</figure>

### 3.1 Architecture of Self-Attention

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$.

Self-attention maps queries (Q), keys (K), and values (V) to an output via scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q, K, V$: matrices of queries, keys, and values.
- $d_k$: dimension of queries and keys.

### 3.2 Calculating Q, K, V

<img src="https://pic1.zhimg.com/v2-4f4958704952dcf2c4b652a1cd38f32e_1440w.jpg" 
     alt="Calculating Q, K, V" 
     style="width: 50%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Calculating Q, K, V</em></p>

Computed via learned linear projections from input embeddings ($X$):

$$
Q = X W^Q,\quad K = X W^K,\quad V = X W^V
$$

Where $W^Q, W^K, W^V$ are learned parameter matrices.

First calculate the $Q K^T$:

<img src="https://pic4.zhimg.com/v2-9caab2c9a00f6872854fb89278f13ee1_1440w.jpg" 
     alt="Calculating QK^T" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Calculating $Q K^T$</em></p>

Then apply the softmax function to each row to get the attention weights:

<img src="https://pic3.zhimg.com/v2-96a3716cf7f112f7beabafb59e84f418_1440w.jpg" 
     alt="softmax" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Apply softmax on rows$</em></p>

Multiply the attention weights with the values to get the output:

<img src="https://pic2.zhimg.com/v2-7ac99bce83713d568d04e6ecfb31463b_1440w.jpg" 
     alt="softmax" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Apply softmax on rows$</em></p>

<img src="https://pic1.zhimg.com/v2-27822b2292cd6c38357803093bea5d0e_1440w.jpg" 
     alt="softmax" 
     style="width: 75%; max-width: 600px; display: block; margin: 1em auto;" />
<p style="text-align: center;"><em>Figure: Apply softmax on rows$</em></p>


### 3.3 Output of Self-Attention

- Weighted sum of values based on attention scores.
- Output dimension matches input embedding size ($d_{\text{model}}$).

### 3.4 Multi-Head Attention (MHA)

Multiple parallel attention heads run independently, then concatenate:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

Each head computes:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

- Typically, $h=8$, each head dimension $d_k = d_v = d_{\text{model}}/h$.

## 4. Architecture of Encoder

<figure class="align-center">
  <img src="/assets/images/transformer_arch.png" alt="Transformer Arch" style="width: 100%;">
  <figcaption>The Transformer - model architecture.</figcaption>
</figure>

### 4.1 Add and LayerNorm

Each encoder layer includes residual connections and layer normalization:

$$
x = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

- Applied after Multi-head Self-Attention and Feed-Forward layers.

### 4.2 Feed Forward

Position-wise fully-connected feed-forward network (FFN):

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

- Intermediate dimension typically $d_{ff}=2048$.

### 4.3 Input and Output Calculation of Encoder

Overall encoder-layer calculation per layer:

$$
x = \text{LayerNorm}(x + \text{MultiHeadSelfAttn}(x))
$$

$$
x = \text{LayerNorm}(x + \text{FFN}(x))
$$

- Encoder stack repeats this structure $N=6$ times.


## 5. Architecture of Decoder

Decoder consists of 6 identical layers, each with three sub-layers.

### 5.1 First MHA in Decoder Block

- Masked self-attention prevents tokens from attending to future positions.
- Computation is similar to encoder self-attention, but with a causal mask:

$$
x = \text{LayerNorm}(x + \text{MaskedMultiHeadSelfAttn}(x))
$$

### 5.2 Second MHA in Decoder Block (Encoder-Decoder Attention)

- Queries come from previous decoder sub-layer.
- Keys and values come from the final encoder output:

$$
x = \text{LayerNorm}(x + \text{MultiHeadAttn}(x, \text{EncoderOutput}, \text{EncoderOutput}))
$$

### 5.3 Softmax Prediction

- After decoder stack, the final decoder representation is projected to vocabulary logits:
  
$$
\text{logits} = \text{DecoderOutput} \times W_{\text{embedding}}^T
$$

- Softmax function predicts next token probabilities:

$$
P(y_i \mid y_{<i}, x) = \text{softmax}(\text{logits})
$$


## References

**Transformer Architecture** provides a powerful parallelizable structure for sequence modeling, relying entirely on attention mechanisms to encode global dependencies efficiently.

* [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
* [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
