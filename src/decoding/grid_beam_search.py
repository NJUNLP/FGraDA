import torch

from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper
from src.utils.common_utils import Constants
from copy import deepcopy


def combine_grid(grid_item1, grid_item2):
    beam_size = len(grid_item1['is_open'])
    src_sent_len = grid_item1['dec_state']['enc_attn_caches'][0][0].shape[1]
    tgt_sent_len = grid_item1['dec_state']['slf_attn_caches'][0][0].shape[1]
    head_dim = grid_item1['dec_state']['slf_attn_caches'][0][0].shape[2]

    grid = dict()
    grid['is_open'] = grid_item1['is_open'] + grid_item2['is_open']
    grid['given_constraint'] = grid_item1['given_constraint']
    grid['unfinished_constraint'] = grid_item1['unfinished_constraint'] + \
        grid_item2['unfinished_constraint']
    grid['current_constraint'] = grid_item1['current_constraint'] + \
        grid_item2['current_constraint']
    grid['beam_mask'] = torch.cat(
        [grid_item1['beam_mask'], grid_item2['beam_mask']], dim=1)
    grid['final_lengths'] = torch.cat(
        [grid_item1['final_lengths'], grid_item2['final_lengths']], dim=1)
    grid['beam_scores'] = torch.cat(
        [grid_item1['beam_scores'], grid_item2['beam_scores']], dim=1)
    grid['final_word_indices'] = torch.cat(
        [grid_item1['final_word_indices'], grid_item2['final_word_indices']], dim=1)

    dec_states = dict()
    dec_states['ctx'] = torch.cat(
        [grid_item1['dec_state']['ctx'], grid_item2['dec_state']['ctx']], dim=0)
    dec_states['ctx_mask'] = torch.cat(
        [grid_item1['dec_state']['ctx_mask'], grid_item2['dec_state']['ctx_mask']], dim=0)
    dec_states['enc_attn_caches'] = []
    for i in range(6):
        cache = []
        for j in range(2):
            item = torch.cat(
                [grid_item1['dec_state']['enc_attn_caches'][i][j].view(beam_size, src_sent_len, -1),
                 grid_item2['dec_state']['enc_attn_caches'][i][j].view(beam_size, src_sent_len, -1)], dim=0)
            item = item.view(-1, src_sent_len, head_dim)

            cache.append(item)

        dec_states['enc_attn_caches'].append(cache)

    dec_states['slf_attn_caches'] = []
    for i in range(6):
        cache = []
        for j in range(2):
            item = torch.cat(
                [grid_item1['dec_state']['slf_attn_caches'][i][j].view(beam_size, tgt_sent_len, -1),
                 grid_item2['dec_state']['slf_attn_caches'][i][j].view(beam_size, tgt_sent_len, -1)], dim=0)
            item = item.view(-1, tgt_sent_len, head_dim)

            cache.append(item)

        dec_states['slf_attn_caches'].append(cache)

    grid['dec_state'] = dec_states

    return grid


def generate_from_grid_item(nmt_model, grid_item, t, c, beam_size, alpha, weight):
    batch_size = 1
    next_scores, dec_states = nmt_model.decode(
        tgt_seq=grid_item['final_word_indices'].view(batch_size*beam_size, -1),
        dec_states=grid_item['dec_state'])

    next_scores = - next_scores  # convert to negative log_probs
    next_scores = next_scores.view(batch_size, beam_size, -1)  # [1, bm, v]
    next_scores = mask_scores(scores=next_scores,
                              beam_mask=grid_item['beam_mask'],
                              eos_idx=Constants.EOS)

    if c >= 1:
        if t == 1:
            next_scores = next_scores*weight

        elif t > 1:
            next_scores = torch.cat(
                [next_scores[:, :beam_size//2, :],
                 next_scores[:, beam_size//2:, :]*weight],
                dim=1)

    # [1, bm, v] = [1, bm, v] + [1, bm, 1]
    beam_scores = next_scores + grid_item['beam_scores'].unsqueeze(2)

    if t == c + 1 and beam_size > 1:
        if t == 1:
            # Force to select first beam at step 1
            beam_scores[:, 1:, :] = float('inf')
        elif t > 1:
            # Force to select first beam at step 2, 3, 4...
            beam_scores[:, 1:beam_size//2, :] = float('inf')

    # Length penalty
    if alpha > 0.0:
        normed_scores = beam_scores * (5.0 + 1.0) ** alpha / \
            (5.0 + grid_item['beam_mask'] +
                grid_item['final_lengths'].unsqueeze(2)) ** alpha
    else:
        normed_scores = beam_scores.detach().clone()  # [bm, v]

    normed_scores = normed_scores.view(batch_size, -1)

    return normed_scores, dec_states


def grid_beam_search(nmt_model, constraint, vocab_tgt, beam_size, max_steps, src_seqs, weight=1.0, only_top_level_beam=True, alpha=-1.0):
    if constraint == []:
        # indices are 0-based
        constraint_num = 0
    else:
        constraint_num = sum([len(cons) for cons in constraint])
        constraint = [[vocab_tgt.token2id(token)
                       for token in cons] for cons in constraint]

    enc_outputs = nmt_model.encode(src_seqs)
    init_dec_states = nmt_model.init_decoder(
        enc_outputs, expand_size=beam_size)

    # Prepare for beam searching
    batch_size = 1
    grid = dict()
    grid[(0, 0)] = {
        'is_open': [1]*beam_size,
        'given_constraint': constraint,
        'unfinished_constraint': [constraint for _ in range(beam_size)],
        'current_constraint': [[] for _ in range(beam_size)],
        'beam_mask': src_seqs.new(batch_size, beam_size).fill_(1).float(),
        'final_lengths': src_seqs.new(batch_size, beam_size).zero_().float(),
        'beam_scores': src_seqs.new(batch_size, beam_size).zero_().float(),
        'final_word_indices': src_seqs.new(batch_size, beam_size, 1).fill_(Constants.BOS),
        'dec_state': init_dec_states
    }

    for t in range(1, max_steps):
        # delete useless grid
        if t >= 2:
            last_last_t = t-2
            for last_last_c in range(max(0, constraint_num+t-max_steps), min(last_last_t, constraint_num)+1):
                grid.pop((last_last_t, last_last_c))

        for c in range(max(0, constraint_num+t-max_steps), min(t, constraint_num)+1):
            if t > c:
                grid_item = deepcopy(grid[(t-1, c)])
                state = 'only_generate'
                beam_size_tmp = beam_size
                if c > 0:
                    grid_item = deepcopy(combine_grid(
                        grid[(t-1, c)], grid[(t-1, c-1)]))
                    state = 'generate_and_constraint'
                    beam_size_tmp = beam_size*2
            elif t == c:
                grid_item = deepcopy(grid[(t-1, c-1)])
                state = 'only_constraint'
                beam_size_tmp = beam_size

            beam_scores, dec_states = generate_from_grid_item(
                nmt_model, grid_item, t, c, beam_size_tmp, alpha, weight)  # [1, bm, v]

            if state == 'only_generate':
                pass

            elif state == 'only_constraint':
                beam_scores = beam_scores.view(beam_size_tmp, -1)
                beam_scores_backup = beam_scores.clone()
                for idx in range(beam_size_tmp):
                    if grid_item['is_open'][idx] == 1:
                        beam_scores[idx, :] = float('inf')
                        for cons in grid_item['unfinished_constraint'][idx]:
                            beam_scores[idx, cons[0]
                                        ] = beam_scores_backup[idx, cons[0]]
                    elif grid_item['is_open'][idx] == 0:
                        beam_scores[idx, :] = float('inf')
                        beam_scores[idx, grid_item['current_constraint'][idx][0]] \
                            = beam_scores_backup[idx, grid_item['current_constraint'][idx][0]]
                beam_scores = beam_scores.view(batch_size, -1)

            elif state == 'generate_and_constraint':
                beam_scores = beam_scores.view(beam_size_tmp, -1)
                beam_scores_backup = beam_scores.clone()
                # generate from grid[(t-1, c)]
                for idx in range(0, beam_size_tmp//2):
                    if grid_item['is_open'][idx] == 0:
                        # not allowed to generate in the middle of constraint
                        beam_scores[idx, :] = float('inf')
                    elif grid_item['is_open'][idx] == 1:
                        pass

                # constraint from grid[(t-1, c-1)]
                for idx in range(beam_size_tmp//2, beam_size_tmp):
                    if grid_item['is_open'][idx] == 1:
                        # start a new constraint
                        beam_scores[idx, :] = float('inf')
                        # check if any constraint is unfinished
                        if len(grid_item['unfinished_constraint'][idx]) > 0:
                            for cons in grid_item['unfinished_constraint'][idx]:
                                beam_scores[idx, cons[0]
                                            ] = beam_scores_backup[idx, cons[0]]
                    elif grid_item['is_open'][idx] == 0:
                        # continue a constraint
                        beam_scores[idx, :] = float('inf')
                        cons = grid_item['current_constraint'][idx]
                        beam_scores[idx, cons[0]
                                    ] = beam_scores_backup[idx, cons[0]]
                beam_scores = beam_scores.view(batch_size, -1)

            # Get topK with beams
            vocab_size = int(beam_scores.size(-1)/beam_size_tmp)
            batch_size = 1
            _, indices = torch.topk(
                beam_scores, k=beam_size_tmp, dim=-1, largest=False, sorted=False)  # [1, bm]
            next_beam_ids = torch.div(
                indices, vocab_size, rounding_mode='floor')  # [1, bm]
            next_word_ids = indices % vocab_size  # [1, bm]

            # Re-arrange by new beam indices
            beam_scores = beam_scores.view(batch_size, -1)
            beam_scores = torch.gather(beam_scores, 1, indices)

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=grid_item['beam_mask'],
                                             batch_size=batch_size,
                                             beam_size=beam_size_tmp,
                                             gather_shape=[-1])

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=grid_item['final_word_indices'],
                                                      batch_size=batch_size,
                                                      beam_size=beam_size_tmp,
                                                      gather_shape=[batch_size * beam_size_tmp, -1])

            final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=grid_item['final_lengths'],
                                                 batch_size=batch_size,
                                                 beam_size=beam_size_tmp,
                                                 gather_shape=[-1])

            dec_states = nmt_model.reorder_dec_states(dec_states,
                                                      new_beam_indices=next_beam_ids,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size_tmp)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(Constants.EOS).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                       Constants.PAD)  # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # # If an EOS or PAD is encountered, set the beam mask to 0.0
            final_lengths += beam_mask

            final_word_indices = torch.cat(
                (final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

            # building grid[(t, c)]
            is_open = []
            unfinished_constraint = []
            current_constraint = []
            for beam_id in next_beam_ids.squeeze(0).cpu().numpy().tolist():
                is_open.append(grid_item['is_open'][beam_id])
                unfinished_constraint.append(
                    grid_item['unfinished_constraint'][beam_id])
                current_constraint.append(
                    grid_item['current_constraint'][beam_id])

            if state == 'only_generate':
                pass

            elif state == 'only_constraint':
                for idx in range(beam_size_tmp):
                    select_cons_token = next_word_ids[0, idx].item()
                    if is_open[idx] == 1:
                        # start a new constraint
                        for cons in unfinished_constraint[idx]:
                            if select_cons_token == cons[0]:
                                unfinished_constraint_beam = deepcopy(
                                    unfinished_constraint[idx])
                                unfinished_constraint_beam.remove(cons)
                                unfinished_constraint[idx] = unfinished_constraint_beam
                                current_constraint[idx] = cons
                                break
                        is_open[idx] = 0

                    else:
                        # continue a constaint
                        assert select_cons_token == current_constraint[idx][0]

                    # remove generated token from current_constraint and update is_open state
                    current_constraint_beam = deepcopy(current_constraint[idx])
                    current_constraint_beam.pop(0)
                    current_constraint[idx] = current_constraint_beam
                    if current_constraint[idx] == []:
                        is_open[idx] = 1

            elif state == 'generate_and_constraint':
                for idx in range(beam_size_tmp):
                    select_cons_token = next_word_ids[0, idx].item()

                    # generate from grid[(t-1, c)]
                    if next_beam_ids[0, idx].item() < beam_size_tmp//2:
                        # keep_open
                        is_open[idx] = 1

                    # generate from grid[(t-1, c-1)]
                    else:
                        if is_open[idx] == 1:
                            # start a new constraint
                            for cons in unfinished_constraint[idx]:
                                if select_cons_token == cons[0]:
                                    unfinished_constraint_beam = deepcopy(
                                        unfinished_constraint[idx])
                                    unfinished_constraint_beam.remove(cons)
                                    unfinished_constraint[idx] = unfinished_constraint_beam
                                    current_constraint[idx] = cons
                                    break
                            # close hyp
                            is_open[idx] = 0

                        else:
                            # continue a constraint
                            assert select_cons_token == current_constraint[idx][0]

                        current_constraint_beam = deepcopy(
                            current_constraint[idx])
                        current_constraint_beam.pop(0)
                        current_constraint[idx] = current_constraint_beam
                        if current_constraint[idx] == []:
                            is_open[idx] = 1

                is_open = is_open[:beam_size_tmp//2]
                unfinished_constraint = unfinished_constraint[:beam_size_tmp//2]
                current_constraint = current_constraint[:beam_size_tmp//2]
                beam_mask = beam_mask[:, :beam_size_tmp//2]
                final_lengths = final_lengths[:, :beam_size_tmp//2]
                beam_scores = beam_scores[:, :beam_size_tmp//2]
                final_word_indices = final_word_indices[:,
                                                        :beam_size_tmp//2, :]

                src_sent_len = dec_states['enc_attn_caches'][0][0].shape[1]
                tgt_sent_len = dec_states['slf_attn_caches'][0][0].shape[1]
                head_dim = dec_states['slf_attn_caches'][0][0].shape[2]
                dec_states_backup = deepcopy(dec_states)
                dec_states['ctx'] = dec_states_backup['ctx'][:beam_size_tmp//2, :, :]
                dec_states['ctx_mask'] = dec_states_backup['ctx_mask'][:beam_size_tmp//2, :]
                dec_states['enc_attn_caches'] = []
                for i in range(6):
                    cache = []
                    for j in range(2):
                        item = dec_states_backup['enc_attn_caches'][i][j].view(
                            beam_size_tmp, src_sent_len, -1)[:beam_size_tmp//2, :, :]
                        item = item.view(-1, src_sent_len, head_dim)
                        cache.append(item)

                    dec_states['enc_attn_caches'].append(cache)

                dec_states['slf_attn_caches'] = []
                for i in range(6):
                    cache = []
                    for j in range(2):
                        item = dec_states_backup['slf_attn_caches'][i][j].view(
                            beam_size_tmp, tgt_sent_len, -1)[:beam_size_tmp//2, :, :]
                        item = item.view(-1, tgt_sent_len, head_dim)
                        cache.append(item)

                    dec_states['slf_attn_caches'].append(cache)

            del grid_item
            grid[(t, c)] = {
                'is_open': is_open,
                'given_constraint': constraint,
                'unfinished_constraint': unfinished_constraint,
                'current_constraint': current_constraint,
                'beam_mask': beam_mask,
                'final_lengths': final_lengths,
                'beam_scores': beam_scores,
                'final_word_indices': final_word_indices,
                'dec_state': dec_states
            }

        # check whether beam search can be stoppped
        grid_num = 0
        for c in range(max(0, constraint_num+t-max_steps), min(t, constraint_num)+1):
            if grid[(t, c)]['beam_mask'].eq(0.0).all():
                grid_num += 1

        if grid_num == min(t, constraint_num)+1:
            break

    if not only_top_level_beam:
        # collect candidate beams
        beam_scores = []
        final_lengths = []
        final_word_indices = []
        for c in range(max(0, constraint_num+t-max_steps), min(t, constraint_num)+1):
            beam_scores.append(grid[(t, c)]['beam_scores'])
            final_lengths.append(grid[(t, c)]['final_lengths'])
            final_word_indices.append(grid[(t, c)]['final_word_indices'])
        beam_scores = torch.cat(beam_scores, dim=1)
        final_lengths = torch.cat(final_lengths, dim=1)
        final_word_indices = torch.cat(final_word_indices, dim=1)
        _, indices = torch.topk(
            beam_scores, k=beam_size, dim=-1, largest=False, sorted=False)  # [1, bm]
        beam_scores = torch.gather(beam_scores, 1, indices)
        final_lengths = torch.gather(final_lengths, 1, indices)
        _final_word_indices = []
        for i in indices.squeeze(0).cpu().numpy().tolist():
            _final_word_indices.append(
                final_word_indices[:, i, :].unsqueeze(1))
        final_word_indices = torch.cat(_final_word_indices, dim=1)

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / \
            (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

    return tensor_gather_helper(gather_indices=reranked_ids,
                                gather_from=final_word_indices[:, :, 1:].contiguous(
                                ),
                                batch_size=batch_size,
                                beam_size=beam_size,
                                gather_shape=[batch_size * beam_size, -1])
