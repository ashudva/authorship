<h1 style="text-align:center;">Plagiarism Detection</h1>
<p style="text-align:center;"><strong>Ashish Yadav, Divyanshu Singh, Naveen Mishra</strong> </p>

- [Abstract](#abstract)
- [Introduction](#introduction)
  - [Intrinsic Plagiarism Detection](#intrinsic-plagiarism-detection)
  - [Extrinsic Plagiarism Detection](#extrinsic-plagiarism-detection)
  - [Traditional Approaches](#traditional-approaches)
  - [Problems with traditional Approaches](#problems-with-traditional-approaches)
- [Literature Review](#literature-review)
  - [BERT and Transformer Architecture](#bert-and-transformer-architecture)
  - [Attention:](#attention)
  - [Overview Of Transformer Architecture](#overview-of-transformer-architecture)
    - [Transformer Architecture](#transformer-architecture)
    - [Residual Connections](#residual-connections)
    - [Attention Mechanism](#attention-mechanism)
    - [Attention Heads](#attention-heads)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
    - [Encoder-Decoder self-attention](#encoder-decoder-self-attention)
    - [Masked attention](#masked-attention)
    - [Tokenization](#tokenization)
    - [Word Embeddings](#word-embeddings)
    - [Positional Embeddings](#positional-embeddings)
    - [Self Attention](#self-attention)
  - [BERT Architecture Overview](#bert-architecture-overview)
  - [Transfer Learning](#transfer-learning)
  - [Dataset and Libraries](#dataset-and-libraries)
  - [Proposed Methodology](#proposed-methodology)
    - [Baseline](#baseline)
    - [BERT model fine-tuning](#bert-model-fine-tuning)
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
<p style="color:gray; font-size:13px;">Figure-4: BERT Fine Tuning</p>
</div>
BERT builds on top of a number of clever ideas that have been bubbling up in the NLP community recently – including but not limited to Semi-supervised Sequence Learning (by Andrew Dai and Quoc Le), ELMo (by Matthew Peters and researchers from AI2 and UW CSE), ULMFiT (by fast.ai founder Jeremy Howard and Sebastian Ruder), the OpenAI transformer (by OpenAI researchers Radford, Narasimhan, Salimans, and Sutskever), and the Transformer (Vaswani et al).

There are a number of fairly complex concepts one needs to be aware of to properly wrap one’s head around what BERT is. So we will explain the BERT model, and give a brief overview of Transformer Architecture and Attention Mechanism only to describe our work and methodolgy.

## BERT and Transformer Architecture
BERT is based on the Transformer Architecture introduced in Attention is all you need paper, transformer is - in a nutshell - an Encoder-Decoder model that uses the Attention Mechanism for language modelling an boosting the speed with which these massive attention-based models can be trained. Transformrs were originally designed to work on Machine Translation tasks.

The input to the transformer are word-embeddings followed by a positinal encoding which accounts for the order and position of words in a text, without it the model would not be able to distinguish the context in which a word is being used.
Let's see why the order matters through an example:
Even though she did <i style="color:red;">not</i> win the award, she was satisfied.
Even though she did win the award, she was <i style="color:red;">not</i> satisfied.
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

<div style="width: 80%; margin: 0 auto; text-align:center;">
<image src="./plots/tf-3.png">
<p style="color:gray; font-size:13px;">Figure-7: Encoder-Decoder Submodules</p>
</div>

Attention mechanism is the most important part of the transformer. We'll first explain the attention and then the self-attention.

## Attention:
In a conventional RNN model, it takes two inputs at each timestep - current state (hidden units) and previous state(next word in the sequence). To turn words into numbers we use word embeddings which captures the semantic and other information about the word. These embeddings can be either learned with the model or we can used pretrained embeddings like Word2Vec or GloVe which were trained in an unsupervised manner on a huge datasets with millions of words.
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

**Let's pay attention now!**
A solution was proposed in Bahdanau et al., 2014 and Luong et al., 2015. These papers introduced and refined a technique called “Attention”, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

An attention model differs from a classic sequence-to-sequence model in two main ways:
First, the encoder passes a lot more data to the decoder. Instead of passing the last hidden state of the encoding stage, the encoder passes all the hidden states to the decoder:
<div style="width: 100%; margin: 0 auto; text-align:center;">
<image src="./plots/sqs-1.png">
<p style="color:gray; font-size:13px;">Figure-7: Sequence to Sequence Model</p>
</div>

Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:

1. Look at the set of encoder hidden states it received – each encoder hidden states is most associated with a certain word in the input sentence
2. Give each hidden states a score 
3. Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores

## Overview Of Transformer Architecture

### Transformer Architecture

### Residual Connections

### Attention Mechanism

### Attention Heads

### Encoder

### Decoder

### Encoder-Decoder self-attention

### Masked attention

### Tokenization

### Word Embeddings

### Positional Embeddings

### Self Attention

## BERT Architecture Overview

## Transfer Learning

## Dataset and Libraries

## Proposed Methodology

### Baseline

### BERT model fine-tuning

# Results