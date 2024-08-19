system = dict(
    name='Audio-Diffusion-Captioning',
    notes='b20_mlen33_selfcond_BEATs_noali',
    device=[7],
    note_eval='norm0924_langevin_step_0.1_coef_0.00004_step_count_3',
    seed=42,
    mode='test',
    resume_train=False,
    ckpt_path=
    'output/20240115-091658.data_Clotho_latentTF_Extracted_scratch_v0.2_lr_0.0001_sche_cosine_warm_2000_bs_20/best_val_ema_loss.pth',
    debug=False,
    split_batches=True,
    mixed_precision=False,
    amp=False)
architecture = 'DiffusionLatentTransformer'
max_seq_len = 33
dataset = 'Clotho'
tx_dim = 768
keywords = False
model = dict(
    type='DiffusionLatentTransformer',
    ver='v0.2',
    pretrained_enc_dec_model='./pretrained_models/bart-base',
    tx_dim=768,
    tx_depth=12,
    tx_heads=12,
    latent_dim=768,
    scale_shift=True,
    dropout=0.1,
    audio_encoder_alignment=False,
    audio_encoder_alignment_loss_weight=1,
    audio_encoder_alignment_joint_embed_shape=768,
    audio_encoder_alignment_text_emb_type='sbert',
    self_condition=True,
    keyword_condition=False,
    keyword_condition_null_prob=0.6,
    reference_condition=False,
    req_audio_token_type_emb=True,
    req_audio_pos_emb=False,
    num_classes=0,
    class_conditional=False,
    class_unconditional_prob=0,
    max_seq_len=33,
    audio_feat_extract_dim=768,
    src_encoder=dict(model='Extracted', pretrained=False, frozen=False),
    src_encode_transformer=True,
    transformer_src_encoder_layers=2,
    wav=dict(
        sr=32000,
        window_size=1024,
        hop_length=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        spec_augmentation=True))
diffusion = dict(
    type='ddim',
    architecture='DiffusionLatentTransformer',
    max_seq_len=33,
    timesteps=1000,
    sampling_timesteps=250,
    normalize_latent=True,
    loss_type='l1',
    beta_schedule='linear',
    p2_loss_weight_gamma=0,
    objective='pred_x0',
    ddim_sampling_eta=1,
    ema_decay=0.9999,
    ema_update_every=1,
    langevin_sampling=dict(langevin_fn_type='sim',
                            audio_src_type='audio_features',
                            langevin_need_restore_norm=True,
                            ase_model=dict(wav=dict(sr=32000,window_size=1024,hop_length=320,mel_bins=64),
                                            joint_embed=1024,
                                            input_embedding_dim=768,
                                            cnn_encoder=dict(model='Extracted+TF',pretrained=False,freeze=False,TF_dim=768,audio_feat_extract_dim=768,transformer_src_encoder_layers=3),
                                            text_encoder='bert_embinput',
                                            bert_encoder=dict(type='pretrained_models/distilbert-base-uncased',freeze=False),
                                            training=dict(margin=0.2,freeze=False,loss='ntxent',clip_grad=2,l2_norm=True,l2=False,dropout=0.2,spec_augmentation=True,seed=20)),# set pretrained to False to avoid loading pretrained model
                            langevin_step_size=0.1,
                            langevin_step_count=3,
                            langevin_coeff=0.00004,
                            langevin_control_model_path='pretrained_models/ase_embinput/bartemb_t4_beats_tf3ly_cls_norm0924_distilbert_1e-5_data_Clotho_freeze_False_bs_22_lr_1e-05_margin_0.2_seed_20/models/best_model.pth',
                            ase_mean_scale='pretrained_models/ase_embinput/bartemb_trial1_data_Clotho_freeze_False_bs_20_lr_0.0001_margin_0.2_seed_20/models/ase_mean_scale.pkl',
                            ),
    ce_loss=dict(use=False, weight=0.001, label_smoothing_factor=0)
    )
data = dict(
    dataset='Clotho',
    batch_size=20,
    batch_size_eval=30,
    num_workers=18,
    max_seq_len=33,
    is_keyword=False,
    num_keywords=5,
    h5_with_feat_path='./data/Clotho/features/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1/all_feats_{split}.h5',
    length_distri_file='./data/Clotho/pth/clotho_length_distri.pth')
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
lr_scheduler = dict(type='cosine', num_warmup_steps=2000)
train = dict(
    max_epochs=100,
    monitor_metrics=['loss', 'ema_loss'],
    monitor_modes=['min', 'min'],
    gradient_accumulate_every=2,
    sample_every=99999,
    save_every=5,
    early_stop_patience=0)
eval = dict(
    sample_kwargs=dict(
        nucleus=dict(
            max_length=33,
            min_length=5,
            do_sample=True,
            top_p=0.95,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2),
        beam_5=dict(
            max_length=33,
            min_length=5,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2),
        beam_4=dict(
            max_length=33,
            min_length=5,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2)),
    samples_per_audio=50,
    word_ids_blacklist_file=
    'pretrained_models/bart-base-Clotho/vocab_banned_list.json',
    confine_to_word_list=True)
