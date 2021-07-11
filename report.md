<h1 style="text-align:center;">Plagiarism Detection</h1>
<p style="text-align:center;"><strong>Ashish Yadav, Divyanshu Singh, Naveen Mishra</strong> </p>

- [Abstract](#abstract)
- [Introduction](#introduction)
  - [Intrinsic Plagiarism Detection](#intrinsic-plagiarism-detection)
  - [Extrinsic Plagiarism Detection](#extrinsic-plagiarism-detection)
  - [Traditional Approaches](#traditional-approaches)
  - [Problems with traditional Approaches](#problems-with-traditional-approaches)
- [Literature Review](#literature-review)
  - [Transformer](#transformer)
    - [Architecture](#architecture)
    - [Attention](#attention)
    - [Self-Attention](#self-attention)
      - [Calculating the Self-Attention](#calculating-the-self-attention)
    - [Multi-Head Attention](#multi-head-attention)
  - [BERT Architecture](#bert-architecture)
    - [BERT Input](#bert-input)
    - [Tokenization](#tokenization)
      - [Subword Tokenization](#subword-tokenization)
      - [Byte-Pair Encoding (BPE)](#byte-pair-encoding-bpe)
      - [WordPiece](#wordpiece)
    - [Masked Language Model (MLM)](#masked-language-model-mlm)
    - [Fine-Tuning](#fine-tuning)
- [Dataset and Libraries](#dataset-and-libraries)
- [Proposed Methodology](#proposed-methodology)
- [Results](#results)

# Abstract
Plagiarism is possibly almost an inevitable problem for any type of Intellectual Property, and it is prevalent in literary and scientific works.The widespread use of computers and the advent of the Internet have made it easier to plagiarize the work of others. One of the challenges of detecting plagiarism is that plagiarism changes per the type of work and where the work is getting submitted or published. Several techniques have been adopted over the years for detecting plagiarism in documents, books, art, publications, code and so many other intellectual properties of an individual. Our work is relevant to plagiarism in text documents written by an author. We utilize the latest advancements in Deep Learning, Natural Language Processing (NLP), and Natural Language Understanding (NLU) to check the authorship of a suspect document. In recent years, the NLP community has been putting forward incredibly powerful components that we can freely download and use and fine-tune for our downstream tasks and pipelines. It’s been referred to as NLP’s ImageNet moment, referencing how years ago similar developments accelerated the development of machine learning in Computer Vision tasks. One of the latest milestones in this development is the release of BERT, an event described as marking the beginning of a new era in NLP. BERT is a model that broke several records for how well models can handle language-based tasks. We use the BERT model to extract features from the text document of an author and then try to assign authorship to the suspect text through a classification head on top of the BERT model. Our results have shown a big leap in the performance at correctly assigning the authorship to a text document. We used the freely available C50 dataset from UCI Machine Learning Dataset Repository to train and test our model.

# Introduction
One of the problems that publishers constantly face is - how to automatically and correctly check if a certain document is plagiarised and locating instances of plagiarism or copyright infringement within a work or document. Detection of plagiarism can be undertaken in a variety of ways. Human detection is the most traditional form of identifying plagiarism from written work. This can be a lengthy and time-consuming task for the reader and can also result in inconsistencies in how plagiarism is identified within an organization. For automated plagiarism detection mainly two classes of techniques are used - 
1. Intrinsic Plagiarism Detection 
2. Extrinsic Plagiarism Detection

## Intrinsic Plagiarism Detection
The goal of intrinsic plagiarism detection is to find passages within a document which appear to be significantly different from the rest of the document. In order to do so, we break the process down into three steps.

1. Atomization -- Deconstruct a document into passages.
2. Feature Extraction -- Quantify the style of each passage by extracting stylometric features based on linguistic properties of the text. Each passage is represented numerically as a vector of feature values.
3. Classification -- Compare the feature vectors of passages to one another; those passages that are significantly different will have higher confidences of plagiarism. Return a confidence that a passage was plagiarized.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/intrinsic.jpg">
<p style="color:gray; font-size:13px;">Figure-1 - Intrinsic plagiarism detection</p>
</div>

## Extrinsic Plagiarism Detection
Extrinsic plagiarism detection is given more information to work with: in addition to a suspicious document, we are also given a number of external documents or source documents to compare to the suspicious document. The extrinsic detection process can be broken into three steps:

1. Atomization -- Deconstruct a document into passages.
2. Fingerprinting -- Compress a passage of text into a fingerprint, a set of integers that represent the passage. The integers come from applying a hash function to some subset of n-grams of the passage.
3. Fingerprint Matching -- Passages are now represented by fingerprints, which are simply sets of integers. To compare a fingerprint from the suspicious document with source fingerprints, we can use set similarity measures. Fingerprints with high similarity indicate a high confidence of the presence of plagiarism.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/extrinsic.jpg">
<p style="color:gray; font-size:13px;">Figure-2 - Extrinsic plagiarism detection</p>
</div>

## Traditional Approaches
The figure below represents a classification of all detection approaches currently in use for computer-assisted content similarity detection. The approaches are characterized by the type of similarity assessment they undertake: global or local. Global similarity assessment approaches use the characteristics taken from larger parts of the text or the document as a whole to compute similarity, while local methods only examine pre-selected text segments as input.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/types.png">
<p style="color:gray; font-size:13px;">Figure-3 - Traditional approaches of plagiarism detection</p>
</div>

**1. Fingerprinting:** It is currently the most widely applied approach to content similarity detection. This method forms representative digests of documents by selecting a set of multiple substrings (n-grams) from them. The sets represent the fingerprints and their elements are called minutiae. A suspicious document is checked for plagiarism by computing its fingerprint and querying minutiae with a precomputed index of fingerprints for all documents of a reference collection. Minutiae matching with those of other documents indicate shared text segments and suggest potential plagiarism if they exceed a chosen similarity threshold. Computational resources and time are limiting factors to fingerprinting, which is why this method typically only compares a subset of minutiae to speed up the computation and allow for checks in a very large collection, such as the Internet.

**2. String matching:** String matching is a prevalent approach used in computer science. When applied to the problem of plagiarism detection, documents are compared for verbatim text overlaps. Numerous methods have been proposed to tackle this task, of which some have been adapted to external plagiarism detection. Checking a suspicious document in this setting requires the computation and storage of efficiently comparable representations for all documents in the reference collection to compare them pairwise. Generally, suffix document models, such as suffix trees or suffix vectors, have been used for this task. Nonetheless, substring matching remains computationally expensive, which makes it a non-viable solution for checking large collections of documents.

**3. Bag of words:** Bag of words analysis represents the adoption of vector space retrieval, a traditional IR concept, to the domain of content similarity detection. Documents are represented as one or multiple vectors, e.g. for different document parts, which are used for pair-wise similarity computations. Similarity computation may then rely on the traditional cosine similarity measure or on more sophisticated similarity measures.

**4. Citation analysis:** Citation-based plagiarism detection (CbPD) relies on citation analysis and is the only approach to plagiarism detection that does not rely on textual similarity. CbPD examines the citation and reference information in texts to identify similar patterns in the citation sequences. As such, this approach is suitable for scientific texts, or other academic documents that contain citations. Citation analysis to detect plagiarism is a relatively young concept. It has not been adopted by commercial software, but a first prototype of a citation-based plagiarism detection system exists. Similar order and proximity of citations in the examined documents are the main criteria used to compute citation pattern similarities. Citation patterns represent subsequences non-exclusively containing citations shared by the documents compared. Factors, including the absolute number or relative fraction of shared citations in the pattern, as well as the probability that citations co-occur in a document are also considered to quantify the patterns’ degree of similarity.

**5. Stylometry:** Stylometry subsumes statistical methods for quantifying an author’s unique writing style and is mainly used for authorship attribution or intrinsic plagiarism detection. Detecting plagiarism by authorship attribution requires checking whether the writing style of the suspicious document, which is written supposedly by a certain author, matches with that of a corpus of documents written by the same author. Intrinsic plagiarism detection, on the other hand, uncover plagiarism based on internal pieces of evidence in the suspicious document without comparing it with other documents. This is performed by constructing and comparing stylometric models for different text segments of the suspicious document, and passages that are stylistically different from others are marked as potentially plagiarized, Although they are simple to extract, character n-grams are proven to be among the best stylometric features for intrinsic plagiarism detection.

**6. Performance:** Comparative evaluations of content similarity detection systems indicate that their performance depends on the type of plagiarism present (see figure). Except for citation pattern analysis, all detection approaches rely on textual similarity. It is therefore symptomatic that detection accuracy decreases the more plagiarism cases are obfuscated.

## Problems with traditional Approaches
Traditional approaches have several problems, we have explained some of the most prominent ones here.
1. Preeminent problem with plagiarism detection using authorship is the sheer number of authors out of which the algorithm needs to decide the actual author of the suspect document.
2. In case of  Fingerprinting, if someone plagiarise some part of the document or modify the sementics of the text by either using synonyms or rearranging the text in some way then this method obtains a fairly different Digest for that document and the plagiarism goes undetected from the system and it can't be used as a reliable tool.
3. String Matching has the same issue as with the Fingerprinting, it does not detect any kind of semantic modifications in the text document whatsoever, thus this technique too can not be used reliably. 
4. Bag of words technique is used as a data preprocessing step for machine learning models, and often to extract stylometric features and other statistics from the generated matrix. Some of the authors used it along with tf-idf to account for rare and frequent words within documents. Problem with BOW is that it produces very large matrix and a lot of the features in the matrix are not useful for the task and thus obscure the decesion boundary. Even when PCA was used to reduce the dimensionality, it proved to be less effective because it does not account for the order of words in a sentence or document. Even if we randomly shuffle the words in a document, the downstream machine learning model would still classify it same as before.
5. Citation based methods do not work where the document does not contain citations or if the author choose not to include the document in the citation.
6. Stylometry is so far the most interesting method used for detecting plagiarism, but the methods used have not been able to extract features from the document in a way that enables a machine learning model to accurately predict its authorship.

# Literature Review
The year 2018 has been an inflection point for machine learning models handling text (or more accurately, Natural Language Processing or NLP for short). Our conceptual understanding of how best to represent words and sentences in a way that best captures underlying meanings and relationships is rapidly evolving. Moreover, the NLP community has been putting forward incredibly powerful components that we can freely download and use in your own models (It’s been referred to as NLP’s ImageNet moment, referencing how years ago similar developments accelerated the development of machine learning in Computer Vision tasks).
One of the latest milestones in this development is the release of BERT, an event described as marking the beginning of a new era in NLP which is based on the Transformer Architecture published in the popular paper **Attention is all you need**. BERT is a model that broke several records for how well models can handle language-based tasks. Soon after the release of the paper describing the model, the team also open-sourced the code of the model, and made available for download versions of the **model that were already pre-trained on massive datasets**. This is a momentous development since it enables anyone building a machine learning model involving language processing to use this powerhouse as a readily-available component – saving the time, energy, knowledge, and resources that would have gone to training a language-processing model from scratch.

Figure below describes how BERT was trained:

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/bert-transfer-learning.png">
<p style="color:gray; font-size:13px;">Figure-4: BERT Pretraining and Fine Tuning</p>
</div>
BERT builds on top of a number of clever ideas that have been bubbling up in the NLP community recently – including but not limited to Semi-supervised Sequence Learning (by Andrew Dai and Quoc Le), ELMo (by Matthew Peters and researchers from AI2 and UW CSE), ULMFiT (by fast.ai founder Jeremy Howard and Sebastian Ruder), the OpenAI transformer (by OpenAI researchers Radford, Narasimhan, Salimans, and Sutskever), and the Transformer (Vaswani et al).

There are a number of fairly complex concepts one needs to be aware of to properly wrap one’s head around what BERT is. Since the BERT is a special kind of transformer model so we'll first explain the transformer model and then BERT.

## Transformer
BERT is based on the Transformer Architecture introduced in <i style="color:tomato;">Attention is all you need </i> paper, transformer is - in a nutshell - an Encoder-Decoder model that uses the Attention Mechanism for language modelling an boosting the speed with which these massive attention-based models can be trained. Transformrs were originally designed to work on Machine Translation tasks.
In a conventional RNN model, it takes two inputs at each timestep - current state (hidden units) and previous state(next word in the sequence). To turn words into numbers we use word embeddings which captures the semantics and other information about the word. These embeddings can be either learned with the model or we can used pretrained embeddings like Word2Vec or GloVe which were trained in an unsupervised manner on a huge datasets with millions of words.
<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/embedding.png">
<p style="color:gray; font-size:13px;">Figure-7: Word Embeddings</p>
</div>

In a machine translation task using sequence-to-sequence model, RNN produces its output by taking into account its current input and previous hidden state. The final encoded output is called the **Context** which is then passed to the decoder which sequentially process the context to generate the target sequence.
<div style="width: 90%; margin: 0 auto; text-align:center;">
<image src="./plots/seq-to-seq.png">
<p style="color:gray; font-size:13px;">Figure-7: Sequence to Sequence Model</p>
</div>

The problem with this kind of model is that, the encoded vecotor proved to be a bottleneck and it can not encode a long sequence properly. A solution to this problem was offered by LSTM and GRU models but they did not resolve the problem completely.

Transformer model solves this problem by using Attention Mechanism and eliminating the RNN and LSTM completely from the encoder-decoder architecture.
### Architecture

The input to the transformer are word-embeddings followed by a positinal encoding which accounts for the order and position of words in a text, without it the model would not be able to distinguish the context in which a word is being used.
Let's see why the order matters through an example:
Even though she did <i style="color:tomato;">not</i> win the award, she was satisfied.
Even though she did win the award, she was <i style="color:tomato;">not</i> satisfied.
Without the positional encoding, the transformer can not tell the difference between the two sentences.
<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/pos-en.png">
<p style="color:gray; font-size:13px;">Figure-5: Positional Encoding</p>
</div>

**pos**: Position of a word in the sentence
**i**: index in Embedding dimension
**d**: Embedding dimensionality

The encoded text is then passed into the transformer and it passes throught encoder and decoder blocks to produces the text in the target language.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/tf-1.png">
<p style="color:gray; font-size:13px;">Figure-6: Abstract Transformer Architecture</p>
</div>

When we unravel the Encoders and Decoders block, they are themselves a stack of multiple encoders and decoders - stack of 6 encoders and decoders was used in the original paper.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/tf-2.png">
<p style="color:gray; font-size:13px;">Figure-7: Encoder-Decoder Stacks as a Transformer</p>
</div>

If we can understand the encoder, we would easily understand the decoder as it is almost similar to the decoder with some modification. The encoder consists of two moules - **Self Attention** and **Feed Forward**
Additional steps like dropout, layrenorm(or Add & Norm as stated in the diagram) are used for the regularizatio of the model.
Each of the submodule is connected with the previous module (encoded inputs for the first encoder) with a residual connections(developed by Kaiming and hist team at Microsoft that won the ImageNet challenge) similar to what we see in ResNet or several other CNN based architectures. Residual connection helps in resolving two major problems with a Very Deep Architecture, **Vanishing Gradients** and **Representation Bottleneck**. A residual connecion reinjects the previous representation into the downstream flwo of the data by adding the past output tensor to the later output tensor and thus preventing the information loss during the data flow.

The output of each encoder sublayer is **$LayerNorm\big(x + SubLayer\small(x\small)\big)$** where **$SubLayer\small(x)$** is the function implemented by the corresponding sub-layer. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension **$d_{model}$**.

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/tf-3.png">
<p style="color:gray; font-size:13px;">Figure-7: Encoder-Decoder Submodules</p>
</div>

Attention mechanism is the most important part of the transformer. We'll first explain the attention and then the self-attention.
### Attention 
Attention mechanism was first used in seq-to-seq models for machine translation. Before that seq-to-seq models used a combination of RNN, CNN, LSTM, and GRU and were very successful at this task, in 2016 Google started using them in Google Translate and other applications.
A solution to problems with seq-to-seq models using the recurrence was proposed in [Bahdanau et al., 2014 and Luong et al., 2015]. These papers introduced and refined a technique called “Attention”, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

An attention model differs from a classic sequence-to-sequence model in two main ways:
First, the encoder passes a lot more data to the decoder. Instead of simply passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder:
<div style="width: 100%; margin: 0 auto; text-align:center;">
<image src="./plots/sqs-1.png">
<p style="color:gray; font-size:13px;">Figure-7: Sequence to Sequence Model</p>
</div>

Second, an attention decoder does an extra step - calculate the attention on its input - before producing its output. In order to focus on the parts of the input that are relevant to this decoding at a certain time step.
The decoder does the following:
1. Look at the set of encoder hidden states it received
2. Give each hidden states a score as each hidden-state is most associated with certain word in the input sentence.
3. Take the softmax of these socres and use them as weights.
4. Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores
5. Take the weighted sum of the input hidden-states
6. The scoring is done at each time step.

Note that the model isn’t just mindless aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:

<div style="width: 40%; margin: 0 auto; text-align:center;">
<image src="./plots/tf-4.png">
<p style="color:gray; font-size:13px; padding-left:70px;">Figure-8: Attention</p>
</div>
<p style="color:gray; font-size:13px; padding-left:70px;">It's evident from the figure that the model paid attention correctly when outputing "European Economic Area". In French, the order of these words is reversed ("européenne économique zone") as compared to English. Every other word in the sentence is in similar order</p>

### Self-Attention
It's almost similar to how attention mechanism works. To get a high level overview of self-attention and to see why it works so well, let's say we want to translate a sentence from english to a target language-

"I was working on a project, and it turned out to be a fiasco."

If we were to translate this sentence, we know that the words <i style="color:tomato;">'it'</i> and <i style="color:tomato;">'fiasco'</i> are related to the word <i style="color:tomato;">'project'</i>. There are several such relationship between words in a sentence that we need to pay *attention* to while translation. Attention mechanism uses this simple idea of paying attention to relevent words while translating a sequence by assigning a score to other words of the input sentence. Model uses these scores(We'll explain how to calculate these in the next section) to decide the parts of the input to focus on.

We as humans can form these association very easily but it's not so easy for an algorithm. Self-attention allows the word <i style="color:tomato;">'it'</i> with <i style="color:tomato;">'project'</i>. *Self-attention also allows the encoder to capture both the left and right context in a sentence* which leads to better encodings for the sequence.

#### Calculating the Self-Attention
It involves three weight matrices (which are learned during teh training process).
1. Quries
2. Keys
3. Values

Steps to calculate self-attention -
1. Turn the input words into embeddings
2. Calculate the scores
3. Divide the socres by **$\sqrt d_k$**, where **$d_k$** is dimentionality of *key vectors*. This step leads to more stable gradients.
4. Apply Softmax to the scores
5. Take weighted sum of the value vectors.

The intuition for the final step is that we want to keep the values of the words which are relevent and discard/drown the values of the irrelevent words.

The final equation for calculating the self-attention:
**$$Attention(Q,K,V)=softmax(\cfrac{QK^T}{\sqrt{d_k}}).V$$**

<div style="width: 20%; margin: 0 auto; text-align:center;">
<image src="./plots/self-attention.png">
<p style="color:gray; font-size:13px;">Figure-9: Self-Attention</p>
</div>

The operations are performed in the matrix form for two reasons:
1. Process the entire input sequence at once
2. Using the matrix multiplication for faster parallel computation
### Multi-Head Attention
The attention mechanism significantly reduced the representational bottleneck posed by the RNN or LSTM seq-to-seq architecture but one of the problem with them still needed to tackeled with i.e. <i style="color:blue">*these models still learned in a sequential manner*</i> and therefore the learning process took a significant amount of time and resources. The transformer architecture address this problem with removing RNN or LSTM completely from the model architecture and using only self-attention for encoding and decoding tasks.

Instead of performing a single attention function with $d_{model}-dimensional$ keys, values and queries, it is beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$ ,$d_k$ and $d_v$*dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v-dimentional$ output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure-10. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

**$$\begin{array}{cc}
  MultiHead(Q, K, V) = Concat(head_1,....,head_h)W^o \\
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
\end{array}$$**
Where the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model}\times d_k}$, $W_i^K \in \mathbb{R}^{d_{model}\times d_k}$, $W_i^V \in \mathbb{R}^{d_{model}\times d_v}$, and $W_i^O \in \mathbb{R}^{hd_v\times d_{model}}$
<div style="width: 40%; margin: 0 auto; text-align:center;">
<image src="./plots/mh-attention.png">
<p style="color:gray; font-size:13px;">Figure-10: Multi-Head Self-Attention</p>
</div>

e.g, If $d_{model} = 512$ and $h = 8$ then self-attention layer will output 8-attention vectors of **$d_v = d_k = \cfrac{d_{model}}{h} = 64$**. Then all of the attention vectors are concatenated to form a $d_{model}-dimensional$ vector.

Multi-Head Attention enhances the model in following ways:
1. If z1 is attention of first word in the sentence, it contains encoding for every word in the sentence but it could be dominated by the word itself.
2. It increases the representational power of the encoded vectors as each attention head has its own set of querys, keys, and values weight matrices.
3. Multiple heads can be processed in parallel laveraging the compute capability of modern hardware.
4. Multi-Head attention allows the model to have large number of parameters and makes the learning process much more efficient and faster.

## BERT Architecture
The original paper from Google presents two model sizes for BERT:
BERT BASE – Comparable in size to the OpenAI Transformer in order to compare performance.
BERT LARGE – A ridiculously huge model which achieved the state of the art results reported in the paper.
<i style="color:blue">BERT is basically a pretrained Transformer Encoder stack</i>.The previous section already described the Transformer model – a foundational concept for BERT.
<div style="width: 100%; margin: 0 auto; text-align:center;">
<image src="./plots/bert.png">
<p style="color:gray; font-size:13px;">Figure-11: BERT High Level Architecture</p>
</div>
Both BERT model sizes have a large number of encoder layers (which the paper calls Transformer Blocks) – twelve for the Base version, and twenty four for the Large version. These also have larger feedforward-networks (768 and 1024 hidden units respectively), and more attention heads (12 and 16 respectively) than the default configuration in the reference implementation of the Transformer in the initial paper (6 encoder layers, 512 hidden units, and 8 attention heads).

### BERT Input
BERT is significantly different from the original transformer in how it preprocess the text data.
General Text Preprocessing Steps:
1. Standardization: this step involves removing special symbols, strip white space and convert into uniform encoding.
2. Tokenization: using a certain tokenization algorithm, the text is then converted into tokens which could be words, bytes, n-grams, or subwords. Different transformer models use different tokenization algorithm, BERT uses WordPiece and GPT-2 uses Byte-Level BytePairEncoding.
3. Vectorization: the final step of text preprocessing is to convert the tokens generated from tokenization step into numerical vectors to pass them into the model. This is done using embedding method which is described earlier.

In addition to these three steps, BERT adds some special tokens in the input sequence which are used later during the fine-tuning process.
Special tokens used by BERT:
1. [CLS] - The first input token is supplied with a special [CLS]. CLS here stands for Classification. When we want to fine-tune the BERT for the downstream task of text-classification, we use the hidden states for [CLS] token as features and then train a classification head on top of these features.
2. [SEP] - This token is used to specify the sentence seperation, an is useful when working on NLG(Natural Language Generation) task.
3. [UNK] - Used when the token symbol is not present is the tokenizer vocabulary.
4. [MASK] - BERT model is pretrained as Masked Language Model, i.e. during pretraining, 15% of the words in text are replaced with [MASK] and the goal of the model is to predict these tokens.

### Tokenization
Tokenization is the process of splitting a sentence into tokens, and there are multiple ways to do so. For example,
<i style="color:#6670FF">"Don't you like Transformers? We sure do."</i>
One way to perform tokenization is to simply split the sentence at white space:
[ <i style="color:#6670FF">"Don't", "you", "like", "Transformers?", "We", "sure", "do."</i> ]

If we look at the tokens "Transformers?" and "do.", we notice that the punctuation is attached to the words "Transformer" and "do", which is suboptimal. We should take the punctuation into account so that a model does not have to learn a different representation of a word and every possible punctuation symbol that could follow it, which would explode the number of representations the model has to learn. 
Taking punctuation into account, tokenizing our exemplary text would give:
[ <i style="color:#6670FF">"Don", "'", "t", "you", "like", "Transformers", "?", "We", "sure", "do", "." </i>]

Even though this is better, but it is still disadvantageous, how the tokenization deal with the word "Don't". "Don't" stands for "do not", so it would be better tokenized as ["Do", "n't"]. This is where things start getting complicated, and part of the reason each model has its own tokenizer type. Depending on the rules we apply for tokenizing a text, a different tokenized output is generated for the same text. A pretrained model only performs properly if you feed it an input that was tokenized with the same rules that were used to tokenize its training data.

#### Subword Tokenization
Transformer models use a hybrid between word-level and character-level tokenization called subword tokenization that rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords. For instance "annoyingly" might be considered a rare word and could be decomposed into "annoying" and "ly". Both "annoying" and "ly" as stand-alone subwords would appear more frequently while at the same time the meaning of "annoyingly" is kept by the composite meaning of "annoying" and "ly". This is especially useful in agglutinative languages such as Turkish, where you can form (almost) arbitrarily long complex words by stringing together subwords.

Subword tokenization allows the model to have a reasonable vocabulary size while being able to learn meaningful context-independent representations. In addition, subword tokenization enables the model to process words it has never seen before, by decomposing them into known subwords. For instance, the BERT-Tokenizer tokenizes "I have a new GPU!" as follows:
[ <i style="color:#6670FF">"i", "have", "a", "new", "gp", "##u", "!"</i>]

The tokenizer splits "gpu" into known subwords: [ "gp" and "##u" ]. "##" means that the rest of the token should be attached to the previous one, without space (for decoding or reversal of the tokenization).

#### Byte-Pair Encoding (BPE)
Byte-Pair Encoding (BPE) was introduced in Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015). BPE relies on a pre-tokenizer that splits the training data into words. Pretokenization can be as simple as space tokenization, e.g. GPT-2, Roberta. More advanced pre-tokenization include rule-based tokenization, e.g. XLM, FlauBERT which uses Moses for most languages.

After pre-tokenization, a set of unique words has been created and the frequency of each word it occurred in the training data has been determined. Next, BPE creates a base vocabulary consisting of all symbols that occur in the set of unique words and <i style="color:tomato">learns merge rules to form a new symbol from two symbols of the base vocabulary</i>. It does so until the vocabulary has attained the desired vocabulary size. Note that the desired vocabulary size is a hyperparameter to define before training the tokenizer.

As an example, let’s assume that after pre-tokenization, the following set of words including their frequency has been determined:

(<i style="color:#6670FF">"hug", 10</i>), (<i style="color:#6670FF">"pug", 5</i>), (<i style="color:#6670FF">"pun", 12</i>), (<i style="color:#6670FF">"bun", 4</i>), (<i style="color:#6670FF">"hugs", 5</i>)
Consequently, the base vocabulary is [  <i style="color:#6670FF">"b", "g", "h", "n", "p", "s", "u" </i>]. Splitting all words into symbols of the base vocabulary, we obtain:

(<i style="color:#6670FF">"h" "u" "g", 10</i>), (<i style="color:#6670FF">"p" "u" "g", 5</i>), (<i style="color:#6670FF">"p" "u" "n", 12</i>), (<i style="color:#6670FF">"b" "u" "n", 4</i>), (<i style="color:#6670FF">"h" "u" "g" "s", 5</i>)
BPE then counts the frequency of each possible symbol pair and picks the symbol pair that occurs most frequently. In the example above "h" followed by "u" is present 10 + 5 = 15 times (10 times in the 10 occurrences of "hug", 5 times in the 5 occurrences of “hugs”). However, the most frequent symbol pair is "u" followed by “g”, occurring 10 + 5 + 5 = 20 times in total. Thus, the first merge rule the tokenizer learns is to group all "u" symbols followed by a "g" symbol together. Next, “ug” is added to the vocabulary. The set of words then becomes

(<i style="color:#6670FF">"h" "ug", 10</i>), (<i style="color:#6670FF">"p" "ug", 5</i>), (<i style="color:#6670FF">"p" "u" "n", 12</i>), (<i style="color:#6670FF">"b" "u" "n", 4)</i>, (<i style="color:#6670FF">"h" "ug" "s", 5</i>)
BPE then identifies the next most common symbol pair. It’s "u" followed by "n", which occurs 16 times. "u", "n" is merged to "un" and added to the vocabulary. The next most frequent symbol pair is "h" followed by "ug", occurring 15 times. Again the pair is merged and "hug" can be added to the vocabulary.

At this stage, the vocabulary is [ <i style="color:#6670FF">"b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug" </i>] and our set of unique words is represented as

(<i style="color:#6670FF">"hug", 10</i>), (<i style="color:#6670FF">"p" "ug", 5</i>), (<i style="color:#6670FF">"p" "un", 12</i>), (<i style="color:#6670FF">"b" "un", 4</i>), (<i style="color:#6670FF">"hug" "s", 5</i>)

Assuming, that the Byte-Pair Encoding training would stop at this point, the learned merge rules would then be applied to new words (as long as those new words do not include symbols that were not in the base vocabulary). For instance, the word "bug" would be tokenized to ["b", "ug"] but "mug" would be tokenized as ["<unk>", "ug"] since the symbol "m" is not in the base vocabulary. In general, single letters such as "m" are not replaced by the "<unk>" symbol because the training data usually includes at least one occurrence of each letter, but it is likely to happen for very special characters like emojis.

As mentioned earlier, the vocabulary size, i.e. the base vocabulary size + the number of merges, is a hyperparameter to choose. For instance GPT has a vocabulary size of 40,478 since they have 478 base characters and chose to stop training after 40,000 merges.

#### WordPiece
BERT uses WordPiece algorithm for tokenization of the text, which was introduced in Japanese and Korean Voice Search (Schuster et al., 2012). <i style="color:tomato">WordPiece is the subword tokenization algorithm used for BERT</i>, DistilBERT. The algorithm is very similar to BPE. WordPiece first initializes the vocabulary to include every character present in the training data and progressively learns a given number of merge rules. In contrast to BPE, WordPiece does not choose the most frequent symbol pair, but the one that maximizes the likelihood of the training data once added to the vocabulary.

Referring to the previous example, maximizing the likelihood of the training data is equivalent to finding the symbol pair, whose probability divided by the probabilities of its first symbol followed by its second symbol is the greatest among all symbol pairs. E.g. "u", followed by "g" would have only been merged if the probability of "ug" divided by "u", "g" would have been greater than for any other symbol pair. Intuitively, WordPiece is slightly different to BPE in that it evaluates what it loses by merging two symbols to make ensure it’s worth it.
### Masked Language Model (MLM)
BERT is pretrained using massive amounts of data and computational power as Masked Language model. Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. In technical terms, the prediction of the output words requires:
1. Adding a classification layer on top of the encoder output.
2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
3. Calculating the probability of each word in the vocabulary with softmax.

### Fine-Tuning
Fine-tuning is considered as Transfer learning (TL) which is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited. From the practical standpoint, reusing or transferring information from previously learned tasks for the learning of new tasks has the potential to significantly improve the sample efficiency of a reinforcement learning agent. Fine-Tuning has proved to be very successful in the Computer-Vision tasks, where models like "EfficientNet", "ResNet", "Inception", "Exception" are being constantly used for fine-tuning on a downstream taks.

**How to fine-tune BERT**
Using BERT for a specific task is relatively straightforward:
BERT can be used for a wide variety of language tasks, while only adding a small layer to the core model:
1. Classification tasks such as sentiment analysis are done similarly to Next Sentence classification, by adding a classification layer on top of the Transformer output for the [CLS] token.
2. In Question Answering tasks (e.g. SQuAD v1.1), the software receives a question regarding a text sequence and is required to mark the answer in the sequence. Using BERT, a Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.
3. In Named Entity Recognition (NER), the software receives a text sequence and is required to mark the various types of entities (Person, Organization, Date, etc) that appear in the text. Using BERT, a NER model can be trained by feeding the output vector of each token into a classification layer that predicts the NER label.

In our project, we are only concerned with Classification Fine-Tuning.

# Dataset and Libraries
In our project, we have used two datasets -
**1. C50 Dataset:** The dataset is the subset of RCV1. These corpus has already been used in author identification experiments. In the top 50 authors (with respect to total size of articles) were selected. 50 authors of texts labeled with at least one subtopic of the class CCAT(corporate/industrial) were selected.That way, it is attempted to minimize the topic factor in distinguishing among the texts. The training corpus consists of 2,500 texts (50 per author) and the test corpus includes other 2,500 texts (50 per author) non-overlapping with the training texts.
**2.PAN 2014:**  Dataset consists of training corpus that comprises a set of author verification problems in several languages/genres. Each problem consists of some (up to five) known documents by a single person and exactly one questioned document. All documents within a single problem instance will be in the same language and best efforts are applied to assure that within-problem documents are matched for genre, register, theme, and date of writing. The document lengths vary from a few hundred to a few thousand words.
For our project, we have only used the documents in English Language.

**Libraries Used**:
1. We have used <i style="color:red">Huggingface's transformers</i> library for pretrained Transformer based models and all the models in "transformers" libary are written either in PyTorch or tensorflow and thus they can be used natively with these libraries.
2. We used tensorflow models from transformers and fine-tuned them natively in tensorflow for out classification task.
3. Matplotlib is used for plotting
4. Numpy - Used for handling matrices and other mathematical operations.
5. sklearn - for validation
# Proposed Methodology

# Results