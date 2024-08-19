import torch

# use optimizer to update BART embeddings to maximize the similarity score from a retrieval model. the retrieval model takes in the current BART embeddings and audio, and outputs the similarity score
def langevin_sim(mag_hyper, retrieval_model, config, step_size, sample, attn_masks, audio_src, mean, sigma,
                 alpha, time, time_next, prev_latent, i_max=3):

    I_M = i_max
    if time_next < 0:
        I_M = 0
    # I_M = 3
    # whether to use waveform or (audio_features+audio_feature_len)
    if config['audio_src_type'] == 'waveform':
        audio_input = audio_src['waveform']
        audio_len=None
    elif config['audio_src_type'] == 'audio_features':
        audio_input = audio_src['audio_features']
        audio_len = audio_src['audio_features_attn_mask'].sum(dim=-1)
    else:
        raise ValueError('audio_src_type not supported')
    bart_embs_param = torch.nn.parameter.Parameter(sample)

    with torch.enable_grad():
        for i in range(I_M):
            # Start the optimizer's gradient calculation process
            optimizer = torch.optim.Adagrad([bart_embs_param], lr=step_size)
            optimizer.zero_grad()
            # Pass the bart_embs_param and audio_input to the retrieval model to get the similarity score
            model_out = retrieval_model.langevin_loss(audio_input, bart_embs_param, attn_masks, audio_feature_lengths=audio_len)
            if sigma.mean() == 0:
                loss = model_out['loss'] + mag_hyper * ((mean - bart_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                loss = model_out['loss'] + mag_hyper * ((mean - bart_embs_param) ** 2 / sigma).mean(dim=0).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retrieval_model.parameters(), 20.0)
            optimizer.step()
            epsilon = torch.randn_like(bart_embs_param.data)
            bart_embs_param = torch.nn.parameter.Parameter((bart_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())

    return bart_embs_param.data

LANGEVIN_FN_DICT = {'sim':langevin_sim,}