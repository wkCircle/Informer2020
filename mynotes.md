multiple time series [#105](https://github.com/zhouhaoyi/Informer2020/issues/105): 
Is there a way to deal with categorical feature? [#191](https://github.com/zhouhaoyi/Informer2020/issues/191)
Handling multi entity data [#271](https://github.com/zhouhaoyi/Informer2020/issues/271)

If each time series has only one feature, maybe you can treat all time series as one large multi-dimensional time series, and each time series is one feature of this large sequence. (in your example: 30000+features with 2000 length)
If your time series have many features, and there are spatial connections between different time series, maybe you can try to use graph convolution or graph attention with our model.
The input's shape of Informer model without input layer must be [batch_size, seq_len, dimension], so if your data is multi time series with multi variate, the input's shape of input layer may be [batch_size, seq_len, num_series, num_features].
If you want to use Informer to deal with multi time series whose features is more than 1, you need to modify input layer. A feasible solution is using emebdding layer for each categorical feature and aggregating the embeddings together, and then feed the embeddings to Informer.
Hi, Informer can deal with Categorical Feature, you can use an Embedding Layer to transform the categorical feature into contiguous vector before feed the features into the model.

shoule use Informerstack. (论文的实验使用了1/4L+L组合的informerstack模型。 #issue:186)

seq_len:	Input sequence length of Informer encoder (defaults to 96)
factor:   probsparse attn factor (defaults=5)
label_len:Start token length of Informer decoder (defaults to 48)
pred_len:	Prediction sequence length (defaults to 24)
(seq_len >= label_len)
(label_len takes the last part of seq_len. Hence, the start token in the decoder comes from the last part of encoder's input.)
Dataset 有 scale=True/False 可關閉 train/val/test loader standardization.

作法一: 每一個bottom-level series columnwise合併, 然後 --features M 預測全部cases 
作法二: 每一個bottom-level df columnwise合併, 然後透過embedding 得到唯一的series, 然後 --features M 預測全部case. (#191)


% informer model structure: 
Informer(
  (enc_embedding): DataEmbedding(
    (value_embedding): TokenEmbedding(
      (tokenConv): Conv1d(7, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular)
    )
    (position_embedding): PositionalEmbedding()
    (temporal_embedding): TimeFeatureEmbedding(
      (embed): Linear(in_features=4, out_features=512, bias=True)
    )
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (dec_embedding): DataEmbedding(
    (value_embedding): TokenEmbedding(
      (tokenConv): Conv1d(7, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular)
    )
    (position_embedding): PositionalEmbedding()
    (temporal_embedding): TimeFeatureEmbedding(
      (embed): Linear(in_features=4, out_features=512, bias=True)
    )
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (encoder): Encoder(
    (attn_layers): ModuleList(
      (0): EncoderLayer(
        (attention): AttentionLayer(
          (inner_attention): ProbAttention(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.05, inplace=False)
      )
      (1): EncoderLayer(
        (attention): AttentionLayer(
          (inner_attention): ProbAttention(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.05, inplace=False)
      )
    )
    (conv_layers): ModuleList(
      (0): ConvLayer(
        (downConv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ELU(alpha=1.0)
        (maxPool): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0): DecoderLayer(
        (self_attention): AttentionLayer(
          (inner_attention): ProbAttention(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (cross_attention): AttentionLayer(
          (inner_attention): FullAttention(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.05, inplace=False)
      )
    )
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (projection): Linear(in_features=512, out_features=7, bias=True)
)


% informer stack structure: 
