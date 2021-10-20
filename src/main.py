import itertools
import os
import random
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
import json

import src.distributed as dist
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding import beam_search, ensemble_beam_search, grid_beam_search
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model, load_predefined_configs
from src.modules.criterions import NMTCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import build_scheduler
from src.utils.common_utils import *
from src.utils.configs import pretty_configs, add_user_configs, default_base_configs
from src.utils.logging import *
from src.utils.moving_average import MovingAverage


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def combine_from_all_shards(all_gathered_output):
    """Combine all_gathered output split by ```split_shards_iterator```
    """
    output = []
    for items in itertools.zip_longest(*all_gathered_output, fillvalue=None):
        for item in items:
            if item is not None:
                output.append(item)

    return output


def prepare_configs(config_path: str, predefined_config: str = "") -> dict:
    """Prepare configuration file"""
    # 1. Default configurations
    default_configs = default_base_configs()

    # 2. [Optional] Load pre-defined configurations
    default_configs = load_predefined_configs(
        default_configs, predefined_config)

    # 3. Load user configs
    config_path = os.path.abspath(config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    configs = add_user_configs(default_configs, configs)

    return configs


def prepare_data(seqs_x, seqs_y, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False,
                    pretrained_name2para=None,
                    finetune_with_regularization=None,
                    weight_lambda=0.0
                    ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """
    y_inp = seqs_y[:, :-1].contiguous()
    y_label = seqs_y[:, 1:].contiguous()

    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label,
                          reduce=False, normalization=normalization)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            elif pretrained_name2para and finetune_with_regularization == "L2":
                L2_distance = 0.0
                current_name2para = model.state_dict()
                for name, para in pretrained_name2para.items():
                    if "weight" in name:
                        L2_distance += torch.norm(para -
                                                  current_name2para[name], p='fro')**2
                loss = loss.sum()
                #print("loss : {}".format(loss))
                #print("L2 loss : {}".format(L2_distance))
                loss += weight_lambda * L2_distance
                #print("modified loss: {}".format(loss))
            else:
                loss = loss.sum()
        torch.autograd.backward(loss)
        return loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label,
                          normalization=normalization, reduce=True)
        return loss.item()


def inference(valid_iterator,
              model,
              vocab_tgt: Vocabulary,
              batch_size,
              max_steps,
              beam_size=5,
              alpha=-1.0,
              rank=0,
              world_size=1,
              using_numbering_iterator=True,
              ):
    model.eval()
    trans_in_all_beams = [[] for _ in range(beam_size)]

    # assert keep_n_beams <= beam_size

    if using_numbering_iterator:
        numbers = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_numbers = batch[0]

        if using_numbering_iterator:
            numbers += seq_numbers

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(
                nmt_model=model, beam_size=beam_size, max_steps=max_steps, src_seqs=x, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            for ii, sent_ in enumerate(sent_t):
                sent_ = vocab_tgt.ids2sent(sent_)
                if sent_ == "":
                    sent_ = '%s' % vocab_tgt.id2token(vocab_tgt.eos)
                trans_in_all_beams[ii].append(sent_)

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:

        if using_numbering_iterator:
            numbers = combine_from_all_shards(
                dist.all_gather_py_with_shared_fs(numbers))

        trans_in_all_beams = [combine_from_all_shards(dist.all_gather_py_with_shared_fs(trans)) for trans in
                              trans_in_all_beams]

    if using_numbering_iterator:
        origin_order = np.argsort(numbers).tolist()

        trans_in_all_beams = [[trans[ii] for ii in origin_order]
                              for trans in trans_in_all_beams]

    return trans_in_all_beams


def ensemble_inference(valid_iterator,
                       models,
                       vocab_tgt: Vocabulary,
                       batch_size,
                       max_steps,
                       beam_size=5,
                       alpha=-1.0,
                       rank=0,
                       world_size=1,
                       using_numbering_iterator=True
                       ):
    for model in models:
        model.eval()

    trans_in_all_beams = [[] for _ in range(beam_size)]

    # assert keep_n_beams <= beam_size

    if using_numbering_iterator:
        numbers = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_numbers = batch[0]

        if using_numbering_iterator:
            numbers += seq_numbers

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU)

        with torch.no_grad():
            word_ids = ensemble_beam_search(
                nmt_models=models,
                beam_size=beam_size,
                max_steps=max_steps,
                src_seqs=x,
                alpha=alpha
            )

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            for ii, sent_ in enumerate(sent_t):
                sent_ = vocab_tgt.ids2sent(sent_)
                if sent_ == "":
                    sent_ = '%s' % vocab_tgt.id2token(vocab_tgt.eos)
                trans_in_all_beams[ii].append(sent_)

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:
        if using_numbering_iterator:
            numbers = dist.all_gather_py_with_shared_fs(numbers)

        trans_in_all_beams = [combine_from_all_shards(
            trans) for trans in trans_in_all_beams]

    if using_numbering_iterator:
        origin_order = np.argsort(numbers).tolist()
        trans_in_all_beams = [[trans[ii] for ii in origin_order]
                              for trans in trans_in_all_beams]

    return trans_in_all_beams


def loss_evaluation(model, critic, valid_iterator, rank=0, world_size=1):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)

        x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

        loss = compute_forward(model=model,
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        n_sents = dist.all_reduce_py(n_sents)

    return float(sum_loss / n_sents)


def bleu_evaluation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_src,
                    vocab_tgt,
                    batch_size,
                    valid_dir="./valid",
                    max_steps=10,
                    beam_size=5,
                    alpha=-1.0,
                    rank=0,
                    world_size=1,
                    using_numbering_iterator=True
                    ):
    translations_in_all_beams = inference(
        valid_iterator=valid_iterator,
        model=model,
        vocab_tgt=vocab_tgt,
        batch_size=batch_size,
        max_steps=max_steps,
        beam_size=beam_size,
        alpha=alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=using_numbering_iterator
    )

    if rank == 0:
        if not os.path.exists(valid_dir):
            os.mkdir(valid_dir)

        hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

        with open(hyp_path, 'w') as f:
            for line in translations_in_all_beams[0]:
                f.write('%s\n' % line)
        with open(hyp_path) as f:
            bleu_v = bleu_scorer.corpus_bleu(f)
    else:
        bleu_v = 0.0

    if world_size > 1:
        bleu_v = dist.broadcast_py(bleu_v, root_rank=0)

    return bleu_v


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path is not None:
        INFO("Loading pretrained model from {}".format(pretrain_path))

        all_parameter_names = set(
            [name for name in nmt_model.state_dict().keys()])

        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():

            if name not in all_parameter_names:
                continue

            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue

            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" %
                                       time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     shuffle=training_configs['shuffle'],
                                     rank=rank)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True,
                                  world_size=world_size, rank=rank)

    bleu_scorer = SacreBLEUScorer(reference_path=data_configs["bleu_valid_reference"],
                                  num_refs=data_configs["num_refs"],
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(
        flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    best_model_saver = Saver(save_prefix=best_model_prefix,
                             num_max_keeping=training_configs['num_kept_best_model'])

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    INFO(nmt_model)

    critic = NMTCriterion(
        label_smoothing=model_configs['label_smoothing'], padding_idx=vocab_tgt.pad)

    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, flags.pretrain_path,
                          exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    # fix partial parameters during finetune
    if flags.pretrain_path:
        for name, parameters in nmt_model.named_parameters():
            if "decoder.embeddings.embeddings.weight" in name:
                parameters.requires_grad = False

    if flags.pretrain_path and training_configs["finetune_with_regularization"]:
        pretrained_name2para = copy.deepcopy(nmt_model.state_dict())
    else:
        pretrained_name2para = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=nmt_model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=nmt_model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=nmt_model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=nmt_model,
                                     optim=optim,
                                     lr_scheduler=scheduler,
                                     collections=model_collections,
                                     ma=ma, device=Constants.CURRENT_DEVICE)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=nmt_model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection(
        "is_early_stop", [False, ])[-1]

    train_loss_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:

            seqs_x, seqs_y = batch

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                x, y = prepare_data(seqs_x, seqs_y, cuda=Constants.USE_GPU)

                loss = compute_forward(model=nmt_model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"],
                                       pretrained_name2para=pretrained_name2para,
                                       finetune_with_regularization=training_configs[
                                           "finetune_with_regularization"],
                                       weight_lambda=training_configs["weight_lambda"])

                update_cycle -= 1
                grad_denom += batch_size
                train_loss += loss

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss = dist.all_reduce_py(train_loss)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(
                        ' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0

            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar(
                        "Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar(
                        "Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar(
                        "train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar(
                        "lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar(
                        "oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(nmt_model):

                    if ma is not None:
                        nmt_model.load_state_dict(
                            ma.export_ma_params(), strict=False)

                    valid_loss = loss_evaluation(model=nmt_model,
                                                 critic=critic,
                                                 valid_iterator=valid_iterator,
                                                 rank=rank,
                                                 world_size=world_size)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection(
                    "history_losses", valid_loss)

                min_history_loss = np.array(
                    model_collections.get_collection("history_losses")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar(
                        "loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar(
                        "best_loss", min_history_loss, global_step=uidx)

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
                                       every_n_step=training_configs['bleu_valid_freq'],
                                       min_step=training_configs['bleu_valid_warmup'],
                                       debug=flags.debug):
                with cache_parameters(nmt_model):

                    if ma is not None:
                        nmt_model.load_state_dict(
                            ma.export_ma_params(), strict=False)

                    valid_bleu = bleu_evaluation(uidx=uidx,
                                                 valid_iterator=valid_iterator,
                                                 batch_size=training_configs["bleu_valid_batch_size"],
                                                 model=nmt_model,
                                                 bleu_scorer=bleu_scorer,
                                                 vocab_src=vocab_src,
                                                 vocab_tgt=vocab_tgt,
                                                 valid_dir=flags.valid_path,
                                                 max_steps=training_configs["bleu_valid_configs"]["max_steps"],
                                                 beam_size=training_configs["bleu_valid_configs"]["beam_size"],
                                                 alpha=training_configs["bleu_valid_configs"]["alpha"],
                                                 world_size=world_size,
                                                 rank=rank,
                                                 )

                model_collections.add_to_collection(
                    key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(
                    np.array(model_collections.get_collection("history_bleus")).max())

                if summary_writer is not None:
                    summary_writer.add_scalar("bleu", valid_bleu, uidx)
                    summary_writer.add_scalar(
                        "best_bleu", best_valid_bleu, uidx)

                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        if rank == 0:
                            # 1. save the best model
                            torch.save(nmt_model.state_dict(),
                                       best_model_prefix + ".final")

                            # 2. record all several best models
                            best_model_saver.save(global_step=uidx,
                                                  model=nmt_model,
                                                  optim=optim,
                                                  lr_scheduler=scheduler,
                                                  collections=model_collections,
                                                  ma=ma)
                else:
                    bad_count += 1

                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                if summary_writer is not None:
                    summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count
                ))

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

                if not is_early_stop:
                    if rank == 0:
                        checkpoint_saver.save(global_step=uidx,
                                              model=nmt_model,
                                              optim=optim,
                                              lr_scheduler=scheduler,
                                              collections=model_collections,
                                              ma=ma)

        if training_progress_bar is not None:
            training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def translate(flags):
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_dataset = TextLineDataset(data_path=flags.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(flags.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if Constants.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    timer.tic()

    translations_in_all_beams = inference(
        valid_iterator=valid_iterator,
        model=nmt_model,
        vocab_tgt=vocab_tgt,
        batch_size=flags.batch_size,
        max_steps=flags.max_steps,
        beam_size=flags.beam_size,
        alpha=flags.alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=True
    )

    acc_time = timer.toc(return_seconds=True)
    acc_num_tokens = sum([len(line.strip().split())
                          for line in translations_in_all_beams[0]])

    INFO('Done. Speed: {0:.2f} words/sec'.format(acc_num_tokens / acc_time))

    if rank == 0:
        keep_n = flags.beam_size if flags.keep_n <= 0 else min(
            flags.beam_size, flags.keep_n)
        outputs = ['%s.%d' % (flags.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for ii in range(keep_n):
                for trans in translations_in_all_beams[ii]:
                    handles[ii].write('%s\n' % trans)


def ensemble_translate(flags):
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_dataset = TextLineDataset(data_path=flags.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_models = []

    model_path = flags.model_path

    for ii in range(len(model_path)):

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
        nmt_model.eval()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        INFO('Reloading model parameters...')
        timer.tic()

        params = load_model_parameters(model_path[ii], map_location="cpu")

        nmt_model.load_state_dict(params)

        if Constants.USE_GPU:
            nmt_model.cuda()

        nmt_models.append(nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    timer.tic()

    translations_in_all_beams = ensemble_inference(
        valid_iterator=valid_iterator,
        models=nmt_models,
        vocab_tgt=vocab_tgt,
        batch_size=flags.batch_size,
        max_steps=flags.max_steps,
        beam_size=flags.beam_size,
        alpha=flags.alpha,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=True
    )

    acc_time = timer.toc(return_seconds=True)
    acc_num_tokens = sum([len(line.strip().split())
                          for line in translations_in_all_beams[0]])

    INFO('Done. Speed: {0:.2f} words/sec'.format(acc_num_tokens / acc_time))

    if rank == 0:
        keep_n = flags.beam_size if flags.keep_n <= 0 else min(
            flags.beam_size, flags.keep_n)
        outputs = ['%s.%d' % (flags.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for ii in range(keep_n):
                for trans in translations_in_all_beams[ii]:
                    handles[ii].write('%s\n' % trans)


def GBS_inference(valid_iterator,
                  model,
                  constraints,
                  vocab_src,
                  vocab_tgt: Vocabulary,
                  batch_size,
                  max_steps,
                  beam_size=5,
                  alpha=-1.0,
                  weight=1.0,
                  only_top_level_beam=True,
                  rank=0,
                  world_size=1,
                  using_numbering_iterator=True
                  ):
    model.eval()
    trans_in_all_beams = [[] for _ in range(beam_size)]

    # assert keep_n_beams <= beam_size

    if using_numbering_iterator:
        numbers = []

    if rank == 0:
        infer_progress_bar = tqdm(total=len(valid_iterator),
                                  desc=' - (Infer)  ',
                                  unit="sents")
    else:
        infer_progress_bar = None

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_numbers = batch[0]

        if using_numbering_iterator:
            numbers += seq_numbers

        seqs_x = batch[1]

        if infer_progress_bar is not None:
            infer_progress_bar.update(len(seqs_x) * world_size)

        x = prepare_data(seqs_x, seqs_y=None, cuda=Constants.USE_GPU)

        with torch.no_grad():
            word_ids = grid_beam_search(
                nmt_model=model, constraint=constraints[seq_numbers[0]],
                vocab_tgt=vocab_tgt, beam_size=beam_size, max_steps=max_steps,
                src_seqs=x, weight=weight, only_top_level_beam=only_top_level_beam, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            for ii, sent_ in enumerate(sent_t):
                sent_ = vocab_tgt.ids2sent(sent_)
                if sent_ == "":
                    sent_ = '%s' % vocab_tgt.id2token(vocab_tgt.eos)
                trans_in_all_beams[ii].append(sent_)

    if infer_progress_bar is not None:
        infer_progress_bar.close()

    if world_size > 1:

        if using_numbering_iterator:
            numbers = combine_from_all_shards(
                dist.all_gather_py_with_shared_fs(numbers))

        trans_in_all_beams = [combine_from_all_shards(dist.all_gather_py_with_shared_fs(trans)) for trans in
                              trans_in_all_beams]

    if using_numbering_iterator:
        origin_order = np.argsort(numbers).tolist()

        trans_in_all_beams = [[trans[ii] for ii in origin_order]
                              for trans in trans_in_all_beams]

    return trans_in_all_beams


def GBS_translate(flags):
    Constants.USE_GPU = flags.use_gpu
    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if rank != 0:
        close_logging()

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_dataset = TextLineDataset(data_path=flags.source_path,
                                    vocabulary=vocab_src)
    INFO('Loading constraint...')
    constraints = json.load(open(flags.constraint_path, 'r'))

    assert flags.batch_size == 1
    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=True,
                                  buffer_size=100000,
                                  numbering=True,
                                  world_size=world_size,
                                  rank=rank
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(flags.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if Constants.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    timer.tic()

    translations_in_all_beams = GBS_inference(
        valid_iterator=valid_iterator,
        model=nmt_model,
        constraints=constraints,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        batch_size=flags.batch_size,
        max_steps=flags.max_steps,
        beam_size=flags.beam_size,
        alpha=flags.alpha,
        weight=flags.weight,
        only_top_level_beam=flags.only_top_level_beam,
        rank=rank,
        world_size=world_size,
        using_numbering_iterator=True,
    )

    acc_time = timer.toc(return_seconds=True)
    acc_num_tokens = sum([len(line.strip().split())
                          for line in translations_in_all_beams[0]])

    INFO('Done. Speed: {0:.2f} words/sec'.format(acc_num_tokens / acc_time))

    if rank == 0:
        keep_n = flags.beam_size if flags.keep_n <= 0 else min(
            flags.beam_size, flags.keep_n)
        outputs = ['%s.%d' % (flags.saveto, i) for i in range(keep_n)]

        with batch_open(outputs, 'w') as handles:
            for ii in range(keep_n):
                for trans in translations_in_all_beams[ii]:
                    handles[ii].write('%s\n' % trans)
