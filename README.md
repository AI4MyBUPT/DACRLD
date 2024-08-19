# Diffusion-based Diverse Audio Captioning with Retrieval-guided Langevin Dynamics
This is the official code for paper Diffusion-based Diverse Audio Captioning with Retrieval-guided Langevin Dynamics.
## Requirements
- ``Python>=3.8``
- ``Pytorch==1.12.1``
- ``transformers==4.26.1``

Details of the python environment we use can be found in ``requirements.txt``.

## Data preparation
- Download AudioCaps and Clotho audio files and put them in ``data/{dataset_name}/waveforms/{split}``. The datasets can be downloaded from [https://github.com/XinhaoMei/audio-text_retrieval]().
- Make the hdfs files according to ``data_prep.py`` in [https://github.com/XinhaoMei/DCASE2021_task6_v2]().
- Extract features with the [BEATs](https://github.com/microsoft/unilm/tree/master/beats) model. Download the codebase and use the script in ``tools/{dataset}_feat_pkl.py``.
- Pack the extracted features with ``tools/{dataset}_feat_h5.py`` and put them to ``data/{dataset}/features/``.

## Evaluation tool preparation
- Use ``coco_caption/get_stanford_models.sh`` to download the required packages for caption evaluation.

## Model preparation
- Put [BART-base](https://huggingface.co/facebook/bart-base/tree/main) from huggingface to ``pretrained_models/bart-base``.
- Put retrieval models for Langevin sampling (our trained version can be downloaded from [link](https://drive.google.com/file/d/1N6AlQS-uix_1llzMBkaIzRpZq6B75V0V/view?usp=drive_link)) in ``pretrained_models/ase_embinput``. You can also train on your own with the code in ``retrieval_training``.
- To test with our trained diffusion models, you can download them from the previous link and put them in ``output``. Replace ``lib/config.py`` with ``config_db/test_{dataset}/config.py`` and read Testing and Evaluaton section.

## Training
- Replace ``lib/config.py`` with ``config_db/train_{dataset}/config.py``
- Run ``python main.py``

## Testing and Evaluaton
- Replace ``lib/config.py`` with ``config_db/test_{dataset}/config.py``
- Run ``python main.py``
- Run ``lib/eval_full.py --in_file {path_to_generated_text_ref_fname_file}``

## Cite
Yonggang Zhu, Aidong Men, Li Xiao, Diffusion-based Diverse Audio Captioning with Retrieval-guided Langevin Dynamics, Information Fusion, 2024