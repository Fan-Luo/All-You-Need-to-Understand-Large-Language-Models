# Train a Large Language Model
Large Language Models (LLMs) are deep learning models trained on vast amounts of text data to understand and generate human language. They are built on architectures like Transformers, which have proven effective in capturing the complexities of language, including syntax, semantics, and context. These models are characterized by their large number of parameters, often running into billions, which allow them to learn rich representations of language.

## Model Pre-training

###  Typical Pre-training Architectures

#### Decoder-only models (also called  left-to-right auto-regressive Transformer models)
Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. 
- The pretraining of decoder models usually revolves around predicting the next word in the sentence. That is, training on the causal language modeling (CLM) task.
- Optimized for generating outputs.
- Good for generative tasks such as text generation, and highly larger scale LM.
- For example: GPT, Claude, Gemini

#### Encoder-only models (also called auto-encoding Transformer models) 
These models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention. 
- The pre-training of these models usually involves corrupting a given sentence, and then challenging the model to recover or reconstruct the original sentence. For example, training on the masked language modeling (MLM) task.
- Optimized to acquire understanding from the input.
- Good for tasks that require understanding of the full input sentence, especially NLU tasks, such as sentence classification and named entity recognition (and more generally word classification), extractive question answering 
- For example: BERT, ALBERT, DistilBERT, ELECTRA, RoBERTa

#### Encoder-decoder models (also called sequence-to-sequence Transformer models)
Encoder-decoder models use both parts of the Transformer architecture. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.
- The pre-training of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, T5 is pre-trained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.
- Good for generative tasks that require an input, such as translation, summarization, or generative question answering.
- For example: BART, T5, mBART

Most recent LLMs are decoder-only models such as GPT, Claude, Gemini. Research shows that causal decoder-only models trained on an autoregressive language modeling objective exhibit the strongest zero-shot generalization after purely unsupervised pre-training. 


### Pre-training Tasks

LLMs are often trained on large amounts of raw text in a self-supervised fashion. Self-supervised learning is a type of training in which the objectives automatically computed from the inputs of the model. That means that humans are not needed to label the data! Here are two most popular pre-training tasks:

<strong>Causal language modeling (CLM)</strong>: a task to predict the next word in a sentence when having read the n previous words.It takes a sequence to be completed and outputs the complete sequence. The output depends on the past and present inputs, but not the future ones. 

<img src='https://media.licdn.com/dms/image/D5612AQEGMapa37P04A/article-inline_image-shrink_1500_2232/0/1678963893664?e=1727913600&v=beta&t=QJIefSssb5Qj-MqIKTTMALLer5nev6VSgWpWvbU02Jg' width="500" height="190">

<strong>Masked language modeling (MLM)</strong>: a percentage of words in a sentence are masked, and the model is then tasked with predicting those masked words using the other words in the same sentence. One way to visualize this is to think of it asa fill-in-the-blanks type of problem.

<img src='https://media.licdn.com/dms/image/D5612AQFKxvO7cb7FHw/article-inline_image-shrink_1500_2232/0/1678963947493?e=1727913600&v=beta&t=fqXHiH3-FQgwdipaW5cQ11HUVqzX5CLUHHP-8iOeLJY' width="510" height="130">

In CLM, the model only considers words to the left, while MLM considers words to the left and right. Therefore, CLM is unidirectional and MLM is bidirectional. For both of the two tasks, there is no single correct answer for the [MASK] prediction / completion. Instead of using classification metrics, the evaluation is based on the distribution of the text prediction. The common metrics for evaluation are cross-entropy loss and perplexity (PPL, also equivalent to the exponential of cross-entropy loss). 

Mmore auxiliary objectives for pre-training language models include Next Sentence Prediction (NSP), Sentence Order Prediction (SOP), Capital Word Prediction (CWP), Sentence Deshuffling (SDS), Sentence distance prediction (SDP), Masked Column Prediction (MCP), Discourse Relation Prediction (DRP), Translation Language Modeling (TLM), Information Retrieval Relevance (IRR), etcs.


## Supervised Fine-Tuning (SFT)

## Reinforcement Learning (RL)
