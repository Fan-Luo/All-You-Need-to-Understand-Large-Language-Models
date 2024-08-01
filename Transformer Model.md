# Transformer Model


Transformers are a type of deep learning architecture that was introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. (2017). 
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*F3ze0JiQNPsLTN8tDFKfaQ.png" style="width: 70%;">
</p>

### Encoder & Decoder Blocks
The main components of a transformer architecture are the encoder and the decoder, each composed of multiple stacked layers. 

<table style="border-collapse: collapse; border-spacing: 0; border:0px solid white; border-color:white; width:100%;">
  <tr style="border:0px solid white; border-spacing: 0; border-color:white;">
    <td style="width: 70%; border:0px solid white; border-spacing: 0; border-color:white; padding: 3px;">
     <img src="https://i.postimg.cc/hj4VdGkv/transformer.png" width="1100" height="470">
    </td>
    <td style="width: 20%; border:0px solid white; border-spacing: 0; border-color:white; padding: 3px;">
     <p style="font-size: 3px;"><strong>Encoder</strong> (left): The encoder receives an input and builds a (embedding) representation of it. </p>
     <img src="https://cdn.prod.website-files.com/6295808d44499cde2ba36c71/65f26289ed424b75f82fc84e_e7583-1hagtuyfnhwg45gzbtbnvsa.png" width="330" height="160">
     <p style="font-size: 3px;"><strong>Decoder</strong> (right): The decoder uses the encoder’s representations to generate an output sequence of words one token at a time. At each step, it is auto-regressive, consuming the previously generated token as additional input when generating the next. </p>
     <img src="https://media.licdn.com/dms/image/D5612AQGQ-71oqLUuOw/article-inline_image-shrink_1500_2232/0/1678960492392?e=1727913600&v=beta&t=CC3D87kQkP4rQ3YR8iWLnePuu86Z4SyEc711qAX9uNM" width="470" height="110">
      <p>In addition to the two sub-layers in each encoder layer, the decoder has an additional encoder-decoder attention sub-layer, which performs multi-head attention over the output of the encoder stack.</p>
    </td>
  </tr>
</table>

### Add & Norm Layer

In the Transformer model, each attention layer is followed by an **Add & Norm** layer. This layer is responsible for implementing two crucial operations: the residual connection and layer normalization. The **Add & Norm** blocks facilitate efficient training by addressing gradient flow and stabilizing the training process.

1. **Residual Connection**: The residual connection provides a direct path for the gradient during backpropagation. This allows gradients to flow more easily through the network, ensuring that the model's vectors are updated incrementally rather than being entirely replaced by the attention layers. This mechanism helps prevent issues related to vanishing or exploding gradients and supports more effective training.

2. **Layer Normalization**: Layer normalization is applied to the outputs of the residual connection. It normalizes the activations of the network to maintain a reasonable scale for the outputs. This normalization step enhances the stability and performance of the model, reducing the internal covariate shift and leading to faster convergence during training.


### The Final Linear and Softmax Layer
The Final Linear and Softmax Layer convert the vector output from the decoder stack into a word. 
- The Linear layer is a simple fully connected neural network that projects the hidden states vector produced by the stack of decoders, into a much, larger vector called a logits vector.
- The Softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.
<p align="center">
    <img src="https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png" width="500" height="310">
</p> 



### Attention Mechanism
The Scaled Dot-Product Attention mechanism is a fundamental component of the Transformer architecture, enabling the model to focus on different parts of the input sequence when generating an output. This attention function maps a query and a set of key-value pairs to an output representation, which is a weighted sum of the values. Here’s a high-level overview of how it works:
1. **Compute dot products**: Calculate the dot products between the query vector (Q) and each key vector (K). 
2. **Scaling**: Divide each result by a scaling factor, usually the square root of the dimension of the key vectors ($\sqrt{d_k}$). This scaling helps to stabilize the gradients during training, preventing large dot products in high dimensions.
3. **Softmax**: Apply the softmax function to obtain a probability distribution. This distribution represents the attention weights assigned to each value vector based on the query.
4. **Weighted sum**: Compute a weighted sum of the value vectors (V) by multiplying the attention weights (scores) with the values. Each value vector is weighted according to its corresponding attention weight. Sum up the weighted value vectors to get the attention-weighted representation.
   
<p>
  <img src="https://production-media.paperswithcode.com/methods/35184258-10f5-4cd0-8de3-bd9bc8f88dc3.png" width="150" height="250" style="display: inline-block; width: 26%; vertical-align: middle;"/>
  <img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" style="display: inline-block; width: 35%; vertical-align: middle;"/>
  <img src="https://media.licdn.com/dms/image/D5612AQH5bprcqhdJkw/article-inline_image-shrink_1500_2232/0/1678962340085?e=1727913600&v=beta&t=f1BlcRnf97bOr8WxZrC_zY7XtWaNg4fRm9BJSuNW2f8" style="margin-left: 140px; display: inline-block; width: 35%; vertical-align: middle;"/>
</p>


The output is a vector that combines information from the value vectors in a way that is most relevant to the given query. This vector is then passed to the next layer, typically a feed-forward neural network, to further process and produce the final output.

#### Self Attention
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. That is, as the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.
For self-attention, Q, K , V are created from the input vectors (i.e, the embeddings of input words). These vectors are created by multiplying the embeddings X by three weight matrices that we trained during the training process. 

#### Cross Attention
The encoder-decoder attention, also called cross-attention, is an attention layer only used in the decoder. The output of the top encoder in the stack is transformed into a set of attention vectors K and V. The query vectors Q come from the previous decoder layer. These are to be used by the decoder in its “encoder-decoder attention” sub-layer with the attention function computation process list above. As a result, it allows every position in the decoder to attend over all positions in the input sequence X. 
<img src="https://jalammar.github.io/images/t/transformer_decoding_2.gif"  width="500" height="310">

#### Multi-head attention 
Multi-head attention enhances the model’s ability to focus on different parts of the input sequence simultaneously by utilizing multiple attention heads. It works by linearly projecting the queries, keys and values multiple times with different linear projection weight matrices. This gives the attention layer multiple “representation subspaces”. On each of these projected versions, the attention function is performed in parallel, and the results are concatenated and once again projected, resulting in the final values. 
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*G5jNmnTIY_MShNnhuAL41Q.jpeg" width="580" height="330" />

##### How Multi-Head Attention Works

1. **Linear Projection**: The queries, keys, and values are linearly projected multiple times using different learned linear projections. Each projection maps the queries, keys, and values to different dimensions: `d_k`, `d_k`, and `d_v` respectively, where `d_k` and `d_v` are typically smaller than `d_model`.
2. **Parallel Attention**: 
For each projected version of the queries, keys, and values, we perform the attention function independently. This yields multiple Z matrices, one for each attention head.
3. **Condense these Z matrices down into a single matrix**:
The outputs from all h attention heads are concatenated. This concatenated output is then linearly projected back to the original d_model dimensionality to produce the final values. 
   
<div style="justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/b4f419df-9d98-489d-aaf7-94d1d7a145a0" width="190" style="display: inline-block; width: 34%; vertical-align: middle;"/>
  <img src="https://github.com/user-attachments/assets/3cb89132-e0f1-49b2-a5b2-b07f2573aa3f" width="280" height="300" align="right" style="display: inline-block; width: 53%;"/>
</div>
  
<br clear="both">

#### Attention Mask
The attention mask can also be used in the encoder/decoder to prevent the model from paying attention to: 
1) some special words — for instance, the special padding word used to make all the inputs the same length when batching together sentences,
2) 'the future' tokens in the decoder during training, as the goal is to predict the most likely token that follows another sequence (on a translating task for instance).
   
<p align="center">
  <img src="https://media.licdn.com/dms/image/D5612AQGw4-eABY4Bnw/article-inline_image-shrink_1500_2232/0/1679016009449?e=1727913600&v=beta&t=OnM860rqtr14Tx1sJaS0TdheBvuo40qwgdFsGlD1MIY"  width="360" height="130">
</p>

Masking works by setting those tokens in the attention map to -inf, meaning that they are converted into zero when fed to the Softmax operation in the attention function calculation. For example, to respect the rule of “no peeking ahead” in the decoder, instead of scoring each word of the input sentence against the current word, only scoring the words before in the input sentence against it. We obtain that result by adding a look-ahead mask as: 
<div style="display: flex; justify-content: space-between;">
  <img src="https://media.licdn.com/dms/image/D5612AQHO70IoVDdb4A/article-inline_image-shrink_1500_2232/0/1678963355973?e=1727913600&v=beta&t=PN6NH4o6uefk_Tjpf2fLSetP9XV4JDkivhVaBZq2bH4"  style="display: inline-block; width: 43%; height: auto; vertical-align: middle;"/>
  <img src="https://media.licdn.com/dms/image/D5612AQHhLqYFqxm4uQ/article-inline_image-shrink_1500_2232/0/1678994646683?e=1727913600&v=beta&t=LsLQKAFQaU294Vv7h2cYYMEjWXtcy3ev7i-qN88i8w4" align="right" style="display: inline-block; width: 50%; height: auto;"/>
</div>

### Positional Encoding

To process input sequences in parallel, transformers use positional encoding to provide information about the position of each element in the input sequence. The "positional encodings" are added to the input embeddings. This allows the model to maintain a sense of order and relative position, which is critical for understanding language.

<img src="https://media.licdn.com/dms/image/D5612AQEEpR5pB-P5jQ/article-inline_image-shrink_1500_2232/0/1678994722491?e=1727913600&v=beta&t=KTDSoY4YGoVebJkuMc0N92TvNB-06_QcynXpnFT1fh8"  style="width: 60%;">



## Summary
The Transformer architecture, introduced in the paper "Attention is All You Need", represents a groundbreaking shift in how we model sequences for natural language processing and other tasks. Unlike previous models reliant on recurrent or convolutional mechanisms, Transformers leverage self-attention mechanisms to process entire sequences in parallel, significantly improving both efficiency and performance. With its innovative use of self-attention and multi-head attention mechanisms, combined with positional encodings and robust training techniques, the Transformer has set new standards in performance and capability across a wide range of applications.


## References
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.´S., Kaiser, Ł., Polosukhin, I. (2017). *Attention Is All You Need*. In *Advances in Neural Information Processing Systems* (NeurIPS 2017). [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[2] Jay Alammar (2022). *[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

[3] Jakob Uszkoreit (2017) *[Transformer: A Novel Neural Network Architecture for Language Understanding](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)*

[4] Samuel Kierszbaum (2020) *[Masking in Transformers’ self-attention mechanism](https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c)*
