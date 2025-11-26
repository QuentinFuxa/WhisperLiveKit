import torch

# code for the end-of-word detection based on the CIF model proposed in Simul-Whisper


def load_cif(cfg, n_audio_state, device):
    """cfg: AlignAttConfig, n_audio_state: int, device: torch.device"""
    cif_linear = torch.nn.Linear(n_audio_state, 1)
    if cfg.cif_ckpt_path is None or not cfg.cif_ckpt_path:
        if cfg.never_fire:
            never_fire = True
            always_fire = False
        else:
            always_fire = True
            never_fire = False
    else:
        always_fire = False
        never_fire = cfg.never_fire
        checkpoint = torch.load(cfg.cif_ckpt_path)
        cif_linear.load_state_dict(checkpoint)
    cif_linear.to(device)
    return cif_linear, always_fire, never_fire


# from https://github.com/dqqcasia/mosst/blob/master/fairseq/models/speech_to_text/convtransformer_wav2vec_cif.py
def resize(alphas, target_lengths, threshold=0.999):
    """
    alpha in thresh=1.0 | (0.0, +0.21)
    target_lengths: if None, apply round and resize, else apply scaling
    """
    # sum
    _num = alphas.sum(-1)
    num = target_lengths.float()
    # scaling
    _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))
    # rm attention value that exceeds threashold
    count = 0
    while len(torch.where(_alphas > threshold)[0]):
        count += 1
        if count > 10:
            break
        xs, ys = torch.where(_alphas > threshold)
        for x, y in zip(xs, ys):
            if _alphas[x][y] >= threshold:
                mask = _alphas[x].ne(0).float()
                mean = 0.5 * _alphas[x].sum() / mask.sum()
                _alphas[x] = _alphas[x] * 0.5 + mean * mask

    return _alphas, _num


def fire_at_boundary(chunked_encoder_feature: torch.Tensor, cif_linear):
    content_mel_len = chunked_encoder_feature.shape[1]  # B, T, D
    alphas = cif_linear(chunked_encoder_feature).squeeze(dim=2)  # B, T
    alphas = torch.sigmoid(alphas)
    decode_length = torch.round(alphas.sum(-1)).int()
    alphas, _ = resize(alphas, decode_length)
    alphas = alphas.squeeze(0)  # (T, )
    threshold = 0.999
    integrate = torch.cumsum(
        alphas[:-1], dim=0
    )  # ignore the peak value at the end of the content chunk
    exceed_count = integrate[-1] // threshold
    integrate = (
        integrate - exceed_count * 1.0
    )  # minus 1 every time intergrate exceed the threshold
    important_positions = (integrate >= 0).nonzero(as_tuple=True)[0]
    if important_positions.numel() == 0:
        return False
    else:
        return important_positions[0] >= content_mel_len - 2
