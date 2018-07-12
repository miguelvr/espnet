#!/bin/bash

[ ! -e steps ] && ln -s ../../wsj/asr1/steps steps
[ ! -e utils ] && ln -s ../../wsj/asr1/utils utils

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0
gpu=           # will be deprecated, please use ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data processing jobs
nj=32
nj_large=80

# data cleanup options
max_frames=3000
max_chars=400


# data directories
datadir=/data/CGN    # where CGN data directory is located, change it according to its location
fbankdir=data/fbank  # directory where the fbank features will be dumped
dumpdir=data/dump    # directory to dump full features

lang="nl"
comp="a;b;c;d;f;g;h;i;j;k;l;m;n;o"

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z ${gpu} ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ ${gpu} -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev

if [ ${stage} -le 0 ]; then
    echo -e "\nstage 0: Data preparation\n"
    if [ ! -d data ]; then
        ### Task dependent. You have to make data the following preparation part by yourself.
        ### But you can utilize Kaldi recipes in most cases
        local/data_prep.sh ${datadir} ${lang} ${comp}
    else
        echo "** data folder exits, skipping data preparation **"
    fi
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ]; then
    echo -e "\nstage 1: Feature Generation\n"
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases

    if [ ! -d ${fbankdir} ]; then
        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        for x in train dev; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} data/${x} exp/make_fbank/${x} ${fbankdir}
        done
    else
        echo "** ${fbankdir} folder exits, skipping make_fbank_pitch.sh step **"
    fi

    mv data/${train_set} data/${train_set}.original
    mv data/${train_dev} data/${train_dev}.original

    # remove utt having more than ${max_frames} frames
    # remove utt having more than ${max_chars} characters
    remove_longshortdata.sh \
        --maxframes ${max_frames} --maxchars ${max_chars} \
        data/${train_set}.original data/${train_set}

    remove_longshortdata.sh \
        --maxframes ${max_frames} --maxchars ${max_chars} \
        data/${train_dev}.original data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
fi

if [ ${stage} -le 2 ]; then
    echo -e "\nstage 2: Features Dump\n"
    # dump features
    echo "Dumping train features"
    dump.sh --cmd "$train_cmd" --nj ${nj_large} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    echo "Dumping dev features"
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
if [ ${stage} -le 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo -e "\nstage 3: Dictionary Preparation\n"
    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py --skip-ncols 1 --nchar 1 --non-lang-syms local/non-language-symbols.txt data/${train_set}/text \
        | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | sed '/<unk>/d' \
        | awk '{print $0 " " NR+1}' >> ${dict}

    wc -l ${dict}
fi

if [ ${stage} -le 4 ]; then
    echo -e "\nstage 4: Json Data Preparation\n"
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms local/non-language-symbols.txt \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms local/non-language-symbols.txt \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
if [ ${stage} -le 5 ]; then
    echo -e "\nstage 5: LM Preparation\n"
    lmdatadir=data/local/lm_train
    if [ ! -d ${lmexpdir} ]; then
        mkdir -p ${lmdatadir}
        mkdir -p ${lmexpdir}
        text2token.py -s 1 -n 1 data/${train_set}/text -l local/non-language-symbols.txt \
            | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 data/${train_dev}/text -l local/non-language-symbols.txt \
            | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/valid.txt

        # use only 1 gpu
        if [ ${ngpu} -gt 1 ]; then
            echo "LM training does not support multi-gpu. single gpu will be used."
        fi
        ${cuda_cmd} ${lmexpdir}/train.log \
            lm_train.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --verbose 1 \
            --outdir ${lmexpdir} \
            --train-label ${lmdatadir}/train.txt \
            --valid-label ${lmdatadir}/valid.txt \
            --epoch 60 \
            --batchsize 256 \
            --dict ${dict}
    else
        echo "** Found existing language model, skipping RNN-LM training**"
    fi
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 6 ]; then
    echo -e "\nstage 6: Network Training\n"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
    exit 0
fi

recog_set=${train_dev}
if [ ${stage} -le 7 ]; then
    echo -e "\nstage 7: Decoding\n"
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

