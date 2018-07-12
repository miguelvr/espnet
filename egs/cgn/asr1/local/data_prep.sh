#!/bin/bash

# Preparation for CGN data

if [ $# -le 2 ]; then
   echo "Arguments should be <CGN root> <language> <comps>, see ../run.sh for example."
   exit 1;
fi

cgn=$1
lang=$2
comps=$3

base=`pwd`
dir=`pwd`/data/local/data
dictdir=`pwd`/data/local/dict_nosp
mkdir -p $dir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh 	# Needed for KALDI_ROOT

cd ${dir}

# create train & dev set
## Create .flist files (containing a list of all .wav files in the corpus)
rm -f temp.flist
IFS=';'
for l in ${lang}; do
	for i in ${comps}; do
		find ${cgn}/data/audio/wav/comp-${i}/${l} -name '*.wav' >>temp.flist
	done
done

IFS=' '
# now split into train and dev
# telephony quality samples are excluded because they have a lower sample rate
grep -vF -f ${local}/nbest-dev-2008.txt temp.flist | grep -v 'comp-c\|comp-d' | sort >train.flist
grep -F -f ${local}/nbest-dev-2008.txt temp.flist | grep -v 'comp-c\|comp-d' | sort >dev.flist
rm -f temp.flist

# create utt2spk, spk2utt, txt, segments, scp, spk2gender
for x in train dev; do
    echo "Processing ${x}.flist"
	${local}/process_flist.pl ${cgn} ${x}
	recode -d h..u8 ${x}.txt					# CGN is not in utf-8 by default
	echo "--> Generating ${x}/spk2utt"
	cat ${x}.utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${x}.spk2utt || exit 1;
done

cd ${base}
# move everything to the right place
for x in train dev; do
    echo "Fixing ${x} directory"
	mkdir -p data/${x}
	cp ${dir}/${x}_wav.scp data/${x}/wav.scp || exit 1;
	cp ${dir}/${x}.txt data/${x}/text || exit 1;
	cp ${dir}/${x}.spk2utt data/${x}/spk2utt || exit 1;
	cp ${dir}/${x}.utt2spk data/${x}/utt2spk || exit 1;
	cp ${dir}/${x}.segments data/${x}/segments || exit 1;
	${utils}/filter_scp.pl data/${x}/spk2utt ${dir}/${x}.spk2gender > data/${x}/spk2gender || exit 1;
	${utils}/fix_data_dir.sh data/${x} || exit 1;
done

echo "Data preparation succeeded"
