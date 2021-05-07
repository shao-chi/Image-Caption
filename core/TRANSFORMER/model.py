import torch
import torch.nn as nn
import numpy as np

from core.TRANSFORMER.modules import EncoderBlock, DecoderBlock, FeedForward
from core.TRANSFORMER.loss import FocalLoss

class Transformer(nn.Module):

    def __init__(self, num_vocab, max_length,
                       encode_dim_positions,
                       encode_dim_features,
                       device,
                       output_name,
                       encode_mask=False,
                       pad_idx=0,
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
                       decode_num_heads=8,
                       
                       move_first_image_feature=False,
                       split_position=False):
        super(Transformer, self).__init__()

        self.max_length = max_length
        self.device = device
        self.num_vocab = num_vocab
        self.pad_idx = pad_idx

        self.encoder = Encoder(dim_positions=encode_dim_positions,
                               dim_features=encode_dim_features,
                               num_blocks=encode_num_blocks,
                               num_heads=encode_num_heads,
                               q_k_dim=encode_q_k_dim,
                               v_dim=encode_v_dim,
                               input_size=encode_input_size,
                               hidden_size=encode_hidden_size,
                               dropout=dropout,
                               split_position=split_position,
                               encode_mask=encode_mask)
        self.decoder = Decoder(num_vocab=num_vocab,
                               max_length=max_length,
                               dim_word_embedding=dim_word_embedding,
                               num_blocks=decode_num_blocks,
                               num_heads=decode_num_heads,
                               q_k_dim=decode_q_k_dim,
                               v_dim=decode_v_dim,
                               input_size=decode_input_size,
                               hidden_size=decode_hidden_size,
                               dropout=dropout,
                               move_first_image_feature=move_first_image_feature)
        self.classifer = nn.Linear(decode_input_size, num_vocab)
        nn.init.xavier_normal_(self.classifer.weight)

        self.softmax = nn.Softmax(dim=1)
        
        if output_name.find('FocalLoss') != -1:
            self.loss = FocalLoss(ignore_index=self.pad_idx)
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='mean')


    def forward(self, object_features,
                      position_features,
                      target_caption):
        context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
                                                                 q=target_caption[:, :-1])

        encode_output, _ = self.encoder(object_features=object_features,
                                        position_features=position_features)

        input_caption = target_caption[:, :-1].clone().long()
        target_caption = target_caption[:, 1:].clone().long().contiguous().view(-1)
        decode_output, _, _ = self.decoder(caption_vector=input_caption,
                                           encode_output=encode_output,
                                           context_attention_mask=context_attention_mask)
        output = self.classifer(decode_output)

        loss = {}
        loss['loss'] = self.loss(output.view(-1, output.size(2)), target_caption)

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
                context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
                                                                         q=decode_input)

                decode_output, _, attention = self.decoder(
                                                   caption_vector=decode_input,
                                                   encode_output=encode_output,
                                                   context_attention_mask=context_attention_mask)
                attention_list.append(np.mean(attention.cpu().numpy()[:, :, t], axis=1))

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

            context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
                                                                     q=decode_input)

            decode_output, _, _ = self.decoder(caption_vector=decode_input,
                                               encode_output=encode_output,
                                               context_attention_mask=context_attention_mask)
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
                    context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
                                                                             q=decode_input)
                    decode_output, _, _ = self.decoder(
                                                    caption_vector=decode_input,
                                                    encode_output=encode_output,
                                                    context_attention_mask=context_attention_mask)

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

    def get_attention_key_pad_mask(self, k, q):
        assert k.size(0) == q.size(0)

        batch_size = k.size(0)
        mask = torch.count_nonzero(k, dim=2).eq(0)
        mask = mask.unsqueeze(1).expand(batch_size, q.size(1), k.size(1))  # b x lq x lk

        return mask


class Encoder(nn.Module):

    def __init__(self, dim_positions,
                       dim_features,
                       num_blocks,
                       num_heads,
                       q_k_dim,
                       v_dim,
                       input_size,
                       hidden_size,
                       dropout,
                       split_position,
                       encode_mask):
        super(Encoder, self).__init__()

        self.encode_mask = encode_mask
        self.split_position = split_position
        if split_position:
            self.object_embedding = nn.Linear(dim_positions-4, input_size, bias=False)
            self.position_embedding = nn.Linear(4, input_size, bias=False)
        else:
            self.position_embedding = nn.Linear(dim_positions, input_size, bias=False)

        self.feature_embedding = nn.Linear(dim_features, input_size, bias=False)
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
        non_pad_mask = self.get_non_pad_mask(position_features)
        self_attention_mask_subsequent = \
                            self.get_subsequent_mask(sequence=position_features)
        self_attention_mask_key_pad = \
                            self.get_attention_key_pad_mask(k=position_features,
                                                            q=position_features)
        self_attention_mask = \
                (self_attention_mask_key_pad + self_attention_mask_subsequent) \
                .gt(0)

        embedded_feature = self.feature_embedding(object_features)

        if self.split_position:
            positions = position_features[:, :, :4].clone()
            objects = position_features[:, :, 4:].clone()

            embedded_position = self.position_embedding(positions)
            embedded_objects = self.object_embedding(objects)
            output = embedded_feature + embedded_position + embedded_objects

        else:
            embedded_position = self.position_embedding(position_features)
            output = embedded_feature + embedded_position

        output = self.norm(output)
        
        attention_list = []
        for block in self.encoder:
            if self.encode_mask:
                output, attention = block(encode_input=output,
                                          non_pad_mask=non_pad_mask,
                                          attention_mask=self_attention_mask)
            else:
                output, attention = block(encode_input=output)
                
            attention_list += [attention]

        return output, attention_list

    def get_attention_key_pad_mask(self, k, q):
        assert k.size(0) == q.size(0)

        batch_size = k.size(0)
        mask = torch.count_nonzero(k, dim=2).eq(0)
        mask = mask.unsqueeze(1).expand(batch_size, q.size(1), k.size(1))  # b x lq x lk

        return mask

    def get_subsequent_mask(self, sequence):
        batch_size, sequence_length, _ = sequence.size()

        subsequent_mask = torch.triu(torch.ones((sequence_length, sequence_length),
                                                device=sequence.device,
                                                dtype=torch.uint8),
                                    diagonal=1)
        subsequent_mask = subsequent_mask \
                            .unsqueeze(0) \
                            .expand(batch_size, sequence_length, sequence_length)  # b x ls x ls

        return subsequent_mask

    def get_non_pad_mask(self, features):
        mask = torch.count_nonzero(features, dim=2).ne(0)

        return mask.type(torch.float).unsqueeze(-1)


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
                       dropout,
                       move_first_image_feature):
        super(Decoder, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        max_length = max_length - 1
        self.max_length = max_length
        self.input_size = input_size
        self.move_first_image_feature = move_first_image_feature

        self.word_embedding = nn.Embedding(num_embeddings=num_vocab,
                                           embedding_dim=dim_word_embedding,
                                           padding_idx=self.pad_idx)
        self.word_embedding_linear = nn.Linear(in_features=dim_word_embedding,
                                               out_features=input_size,
                                               bias=False)
        self.position_embedding = PositionalEncoding(dim_word_embedding=input_size,
                                                     num_positions=max_length)
        self.norm = nn.LayerNorm(normalized_shape=input_size,
                                 eps=1e-6)

        if self.move_first_image_feature:
            self.position_wise_1 = nn.Linear(input_size, hidden_size)
            self.position_wise_2 = nn.Linear(hidden_size, input_size)
            nn.init.xavier_normal_(self.position_wise_1.weight)
            nn.init.xavier_normal_(self.position_wise_2.weight)

            self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        self.decoder = nn.ModuleList([
                DecoderBlock(input_size=input_size,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             q_k_dim=q_k_dim,
                             v_dim=v_dim,
                             dropout=dropout)
                    for _ in range(num_blocks)])

    def forward(self, caption_vector, encode_output,
                      context_attention_mask=None):
        non_pad_mask = self.get_non_pad_mask(sequence=caption_vector)

        self_attention_mask_subsequent = \
                self.get_subsequent_mask(sequence=caption_vector)
        self_attention_mask_key_pad = \
                self.get_attention_key_pad_mask(k=caption_vector,
                                                q=caption_vector)
        self_attention_mask = \
                (self_attention_mask_key_pad + self_attention_mask_subsequent) \
                    .gt(0)

        word_embedding = self.word_embedding(caption_vector)
        word_embedding = self.word_embedding_linear(word_embedding)

        decode_output = self.position_embedding(word_embedding)
        decode_output = self.norm(decode_output)

        decode_attention_list = []
        context_attention_list = []
        for block in self.decoder:
            decode_output, decode_attention, context_attention = \
                block(decode_input=decode_output,
                      encode_output=encode_output,
                      non_pad_mask=non_pad_mask,
                      self_attention_mask=self_attention_mask,
                      context_attention_mask=context_attention_mask)

        decode_attention_list += [decode_attention]
        context_attention_list += [context_attention]

        if self.move_first_image_feature:
            first_encode = encode_output[:, 0].unsqueeze(1)
            decode = self.position_wise_1(decode_output + first_encode)
            decode = self.relu(decode)
            decode = self.position_wise_2(decode)
            decode = self.dropout(decode)
            decode_output = self.layer_norm(decode + decode_output)

        return decode_output, decode_attention, context_attention

    def get_attention_key_pad_mask(self, k, q):
        assert k.size(0) == q.size(0)

        batch_size = k.size(0)
        mask = k.eq(self.pad_idx)
        mask = mask.unsqueeze(1).expand(batch_size, q.size(1), k.size(1))  # b x lq x lk

        return mask

    def get_subsequent_mask(self, sequence):
        batch_size, sequence_length = sequence.size()

        subsequent_mask = torch.triu(torch.ones((sequence_length, sequence_length),
                                                device=sequence.device,
                                                dtype=torch.uint8),
                                    diagonal=1)
        subsequent_mask = subsequent_mask \
                            .unsqueeze(0) \
                            .expand(batch_size, sequence_length, sequence_length)  # b x ls x ls

        return subsequent_mask

    def get_non_pad_mask(self, sequence):
        assert sequence.dim() == 2

        return sequence.ne(self.pad_idx).type(torch.float).unsqueeze(-1)


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


# def get_pos_onehot(length):
#     onehot = torch.zeros(length, length)
#     idxs = torch.arange(length).long().view(-1, 1)
#     onehot.scatter_(1, idxs, 1)

#     return onehot