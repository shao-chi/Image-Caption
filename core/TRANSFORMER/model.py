import torch
import torch.nn as nn
import numpy as np

from core.TRANSFORMER.modules import EncoderBlock, DecoderBlock
from core.settings import PAD_IDX

class Transformer(nn.Module):

    def __init__(self, num_vocab, max_length,
                       encode_dim_positions,
                       encode_dim_features,
                       device,
                       dropout=0.2,

                       encode_input_size=512,
                       encode_q_k_dim=512,
                       encode_v_dim=512,
                       encode_hidden_size=2048,
                       encode_num_blocks=6,
                       encode_num_heads=8,

                       dim_word_embedding=512,
                       decode_input_size=512,
                       decode_q_k_dim=512,
                       decode_v_dim=512,
                       decode_hidden_size=2048,
                       decode_num_blocks=6,
                       decode_num_heads=8):
        super(Transformer, self).__init__()

        self.max_length = max_length
        self.device = device
        self.num_vocab = num_vocab

        self.encoder = Encoder(dim_positions=encode_dim_positions,
                               dim_features=encode_dim_features,
                               num_blocks=encode_num_blocks,
                               num_heads=encode_num_heads,
                               q_k_dim=encode_q_k_dim,
                               v_dim=encode_v_dim,
                               input_size=encode_input_size,
                               hidden_size=encode_hidden_size,
                               dropout=dropout)
        self.decoder = Decoder(num_vocab=num_vocab,
                               max_length=max_length,
                               dim_word_embedding=dim_word_embedding,
                               num_blocks=decode_num_blocks,
                               num_heads=decode_num_heads,
                               q_k_dim=decode_q_k_dim,
                               v_dim=decode_v_dim,
                               input_size=decode_input_size,
                               hidden_size=decode_hidden_size,
                               dropout=dropout)
        self.classifer = nn.Linear(decode_input_size, num_vocab)
        nn.init.xavier_normal_(self.classifer.weight)

        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='mean')


    def forward(self, object_features,
                      position_features,
                      target_caption):
        encode_output, _ = self.encoder(object_features=object_features,
                                        position_features=position_features)

        input_caption = target_caption[:, :-1].clone().long()
        target_caption = target_caption[:, 1:].clone().long().contiguous().view(-1)
        decode_output, _, _ = self.decoder(caption_vector=input_caption,
                                           encode_output=encode_output)
        output = self.classifer(decode_output)

        loss = self.loss(output.view(-1, output.size(2)), target_caption)

        return loss


    def generate_caption_vector(self, object_features,
                                      position_features):
        with torch.no_grad():
            encode_output, _ = self.encoder(object_features=object_features,
                                            position_features=position_features)

            batch_size = encode_output.size(0)
            input_caption = torch.zeros(batch_size, self.max_length+1) \
                                 .long() \
                                 .to(self.device)

            input_caption[:, 0] = 1
            attention_list = []
            for t in range(self.max_length-1):
                decode_input = input_caption[:, :t+1].clone()
                decode_output, _, attention = self.decoder(
                                                   caption_vector=decode_input,
                                                   encode_output=encode_output)
                attention_list.append(np.mean(attention.numpy()[:, :, t], axis=1))

                output = decode_output[:, t]
                output = self.classifer(output)
                output = self.softmax(output)
                output = torch.argmax(output, dim=1)
                
                input_caption[:, t+1] = output.long()

        return input_caption, attention_list


    def beam_search(self, object_features,
                          position_features,
                          beam_size=1):
        with torch.no_grad():
            encode_output, _ = self.encoder(object_features=object_features,
                                            position_features=position_features)

            batch_size = encode_output.size(0)
            input_caption = torch.zeros(beam_size, batch_size, self.max_length) \
                                 .long() \
                                 .to(self.device)

            input_caption[:, :, 0] = 1
            decode_input = input_caption[0, :, :1].clone()
            decode_output, _, _ = self.decoder(
                                                    caption_vector=decode_input,
                                                    encode_output=encode_output)
            output = decode_output[:, 0]
            output = self.classifer(output)
            output = self.softmax(output)

            topk_prob, topk_index = torch.topk(output,
                                               k=beam_size,
                                               dim=1,
                                               sorted=False)
            topk_prob = torch.transpose(topk_prob, 0, 1)
            topk_index = torch.transpose(topk_index, 0, 1)
            input_caption[:, :, 1] = topk_index

            logits_tmp_list = []
            for t in range(1, self.max_length-1):
                logits_tmp_list.clear()

                for b in range(beam_size):
                    decode_input = input_caption[b, :, :t+1].clone()
                    decode_output, _, _ = self.decoder(
                                                    caption_vector=decode_input,
                                                    encode_output=encode_output)

                    output = decode_output[:, t]
                    output = self.classifer(output)
                    output = self.softmax(output) + topk_prob[b].unsqueeze(1)
                    logits_tmp_list.append(output)

                logits = torch.cat(logits_tmp_list, 1)
                topk_prob, topk_index = torch.topk(logits,
                                                   k=beam_size,
                                                   dim=1,
                                                   sorted=False)
                topk_prob = torch.transpose(topk_prob, 0, 1)
                topk_index = torch.transpose(topk_index, 0, 1)

                tmp = torch.stack([torch.arange(batch_size)] * beam_size)
                tmp_index = topk_index // self.num_vocab
                input_caption = input_caption[tmp_index, tmp].clone()

                input_caption[:, :, t+1] = topk_index % self.num_vocab

        return input_caption[0]


class Encoder(nn.Module):

    def __init__(self, dim_positions,
                       dim_features,
                       num_blocks,
                       num_heads,
                       q_k_dim,
                       v_dim,
                       input_size,
                       hidden_size,
                       dropout):
        super(Encoder, self).__init__()

        self.position_embedding = nn.Linear(dim_positions, input_size)
        self.feature_embedding = nn.Linear(dim_features, input_size)
        self.norm = nn.LayerNorm(input_size, eps=1e-6)
        self.encoder = nn.ModuleList([
                EncoderBlock(input_size=input_size,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             q_k_dim=q_k_dim,
                             v_dim=v_dim,
                             dropout=dropout)
                    for _ in range(num_blocks)])

    def forward(self, object_features, position_features):
        embedded_position = self.position_embedding(position_features)
        # output = object_features + embedded_position

        embedded_feature = self.feature_embedding(object_features)
        output = embedded_feature + embedded_position
        output = self.norm(output)
        # output = torch.cat((embedded_feature, embedded_position), 2)

        # attention_list = None
        attention_list = []
        for block in self.encoder:
            output, attention = block(encode_input=output)
            attention_list += [attention]

        return output, attention_list


class Decoder(nn.Module):

    def __init__(self, num_vocab,
                       max_length,
                       dim_word_embedding,
                       num_blocks,
                       num_heads,
                       q_k_dim,
                       v_dim,
                       input_size,
                       hidden_size,
                       dropout):
        super(Decoder, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        max_length = max_length - 1
        self.max_length = max_length
        self.input_size = input_size

        self.word_embedding = nn.Embedding(num_embeddings=num_vocab,
                                           embedding_dim=dim_word_embedding,
                                           padding_idx=PAD_IDX)
        self.word_embedding_linear = nn.Linear(in_features=dim_word_embedding,
                                               out_features=input_size,
                                               bias=False)
        self.position_embedding = PositionalEncoding(dim_word_embedding=input_size,
                                                     num_positions=max_length)
        self.norm = nn.LayerNorm(normalized_shape=input_size,
                                 eps=1e-6)

        self.decoder = nn.ModuleList([
                DecoderBlock(input_size=input_size,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             q_k_dim=q_k_dim,
                             v_dim=v_dim,
                             dropout=dropout)
                    for _ in range(num_blocks)])

    def forward(self, caption_vector, encode_output):
        non_pad_mask = get_non_pad_mask(sequence=caption_vector)

        self_attention_mask_subsequent = \
                get_subsequent_mask(sequence=caption_vector)
        self_attention_mask_key_pad = \
                get_attention_key_pad_mask(k=caption_vector,
                                           q=caption_vector)
        self_attention_mask = \
                (self_attention_mask_key_pad + self_attention_mask_subsequent) \
                    .gt(0)

        word_embedding = self.word_embedding(caption_vector)
        word_embedding = self.word_embedding_linear(word_embedding)

        decode_output = self.position_embedding(word_embedding)
        decode_output = self.norm(decode_output)

        decode_attention_list = []
        decode_encode_attention_list = []
        for block in self.decoder:
            decode_output, decode_attention, decode_encode_attention = \
                block(decode_input=decode_output,
                      encode_output=encode_output,
                      non_pad_mask=non_pad_mask,
                      self_attention_mask=self_attention_mask)

            decode_attention_list += [decode_attention]
            decode_encode_attention_list += [decode_encode_attention]

        return decode_output, decode_attention, decode_encode_attention


class PositionalEncoding(nn.Module):

    def __init__(self, dim_word_embedding, num_positions):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            name='pos_table',
            tensor=self._get_sinusoid_encoding_table(
                            num_positions=num_positions,
                            dim_word_embedding=dim_word_embedding)
        )

    def _get_sinusoid_encoding_table(self, num_positions, dim_word_embedding):

        def get_position_angle_vec(position):
            return [position \
                    / np.power(10000, 2 * (hidden_j // 2) / dim_word_embedding) \
                        for hidden_j in range(dim_word_embedding)]

        sinusoid_table = np.array([get_position_angle_vec(position_i) \
                                    for position_i in range(num_positions)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def get_attention_key_pad_mask(k, q):
    mask = k.eq(PAD_IDX)
    mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)  # b x lq x lk

    return mask

def get_non_pad_mask(sequence):
    assert sequence.dim() == 2

    return sequence.ne(PAD_IDX).type(torch.float).unsqueeze(-1)

def get_subsequent_mask(sequence):
    batch_size, sequence_length = sequence.size()

    subsequent_mask = torch.triu(torch.ones((sequence_length, sequence_length),
                                            device=sequence.device,
                                            dtype=torch.uint8),
                                diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_pos_onehot(length):
    onehot = torch.zeros(length, length)
    idxs = torch.arange(length).long().view(-1, 1)
    onehot.scatter_(1, idxs, 1)

    return onehot