import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from core.TRANSFORMER.model import Encoder, Decoder
from core.config import PAD_IDX, WORD_TO_IDX_PATH


class PolicyNetwork(nn.Module):

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
                       decode_num_heads=8,
                       
                       move_first_image_feature=False,
                       split_position=False):
        super(PolicyNetwork, self).__init__()

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
                               dropout=dropout,
                               split_position=split_position)
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

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, object_features,
                      position_features,
                      target_caption):
        context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
                                                                 q=target_caption[:, :-1])

        encode_output, _ = self.encoder(object_features=object_features,
                                        position_features=position_features)

        input_caption = target_caption[:, :-1].clone().long()
        decode_output, _, _ = self.decoder(caption_vector=input_caption,
                                           encode_output=encode_output,
                                           context_attention_mask=context_attention_mask)
        output = self.classifer(decode_output)

        return output

    
    def sample(self, output):
        # output = self.forward(object_features=object_features,
        #                       position_features=position_features,
        #                       target_caption=target_caption)
        log_probs = F.log_softmax(output, dim=2)
        sequence = torch.argmax(log_probs, dim=2)

        return sequence, log_probs

        # encode_output, _ = self.encoder(object_features=object_features,
        #                                     position_features=position_features)

        # batch_size = encode_output.size(0)
        # input_caption = torch.zeros(batch_size, self.max_length) \
        #                          .long() \
        #                          .to(self.device)
        # log_probs = torch.zeros(batch_size, self.max_length-1, self.num_vocab) \
        #                          .to(self.device)

        # input_caption[:, 0] = 1
        # for t in range(self.max_length-1):
        #     decode_input = input_caption[:, :t+1].clone().long()
        #     context_attention_mask = self.get_attention_key_pad_mask(k=position_features,
        #                                                              q=decode_input)

        #     decode_output, _, _= self.decoder(caption_vector=decode_input,
        #                                       encode_output=encode_output,
        #                                       context_attention_mask=context_attention_mask)

        #     output = decode_output[:, t]
        #     output = self.classifer(output)

        #     log_prob = self.softmax(output)
        #     log_probs[:, t] = log_prob

        #     seq = torch.argmax(log_prob, dim=1)
        #     input_caption[:, t+1] = seq

        # return input_caption[:, 1:], log_probs


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

    def get_non_pad_mask(self, sequence):
        assert sequence.dim() == 2

        return sequence.ne(PAD_IDX).type(torch.float).unsqueeze(-1)
