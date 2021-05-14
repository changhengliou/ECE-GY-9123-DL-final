# Paper to PPT

This repository contains code for sentence selection from papers for slide generation.

## About this code

* PyTorch version: This code requires PyTorch v1.3.0. (`tensor.bool()`)
* Python version: This code requires Python 3.  

## How to run

Use as a submodule of [BERTIterativeLabeling - Repos](https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling)

> To run in CPU mode just delete the line `-gpus 0 \` in `run.sh` or `test.sh`

## Setup the environment

### Package Requirements

```txt
nltk numpy pytorch
```

## Training

The file `run.sh` is an example. Modify it according to your configuration.

```sh
bash run.sh QTYPE VERSION
```

> * QTYPE: . (all), baseline, contribution, future, dataset
> * VERSION: customized version name, can be ignored
> * SCRIPT_NAME: any `run_*.sh` script (except `run_all*.sh` and `run_settings_on.sh`)

```sh
# Run all question types
bash run_all.sh VERSION

# Run all question types using the script
run_all_with.sh SCRIPT_NAME VERSION

# Run all settings on a single question type
# This will create a folder with PREFIX-TENSORBOARD_TEMP which is used for tensorboard
run_settings_on.sh QTYPE PREFIX
```

You can run the tensorboard at the model's folder.

```sh
tensorboard --bind_all --logdir . --port 6006
```

> Make sure you've enable the port for tensorboard.
>
> For example for port 6006:
>
> ```sh
> # on Ubuntu
> sudo ufw allow 6006/tcp
> ```

## Evaluation

The file `test.sh` is an example. Modify it according to your configuration. (e.g. the max decode step)

```sh
bash test.sh QTYPE VERSION EPOCH
```

> * QTYPE: . (all), baseline, contribution, future, dataset
> * VERSION: customized version name, can be ignored
> * EPOCH: specific epoch to test. if none it will just use the latest one

For debuging decoding output => Set `-beam_size 5 -n_best_size 5`, and make sure `beam_size` >= `n_best_size` (Otherwise just set them to 1)

> The usage of other scripts which prefix is start with `test_` are with the same usabe (without `EPOCH` argument)

Recommend to test all trained model with all different evaluation settings.

```sh
bash test_all.sh VERSION
```

### Analysis

Print the top-k probability distribution by using `test_decode_step_1.sh` or `test_decode_step_3.sh`.
The usage (arguments) is the same as `test.sh`

To test on multiple epoch just use `test_multiple_epoch.sh`

## Log-linear Model (Teacher model)

The file `train_loglinear.sh` is an example. Modify it according to your configuration.

> Loss Example
>
> * [knowledge-distillation-pytorch/net.py at master · peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py)
> * [knowledge-distillation-pytorch/train.py at master · peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/train.py)

## Baselines

* Preprocessed data will be in `baselines/data` but Mead will be in `baselines/mead/mead/data/MEAD_TEST`.
* Ouput result will be in `baselines/output`.

### Mead

* [Document](baselines/mead/mead/docs/meaddoc.pdf)
  * 4.2 Using .meadrc files to specify defaults
  * 18 Converting HTML and text files to docsent and cluster format

Setup

`baselines/mead`

```sh
cd baselines/mead

# Install mead
./setup_mead.sh
```

**Setup Addons First**: Follow this [README](baselines/mead/mead/bin/addons/formatting)

You can change the `PERCENT` in `eval_mead.py` first.

```sh
# Run mead
python3 run_mead.py

# Evaluate mead
python3 eval_mead.py
```

Ouput will be `mead/data/MEAD_TEST`

* `MEAD_TEST.extract`
* `MEAD_TEST.summary`

### LexRank

You can modify parameters in `run_lexrank.py`. Either set `PERCENT` or set it to `None` and set `SUMMARY_SIZE`.

```sh
cd baselines/LexRank
python3 run_lexrank.py
```

> * [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)
> * [lexrank · PyPI](https://pypi.org/project/lexrank/)
>
> LexRank is an unsupervised approach to text summarization based on graph-based centrality scoring of sentences. The main idea is that sentences “recommend” other similar sentences to the reader. Thus, if one sentence is very similar to many others, it will likely be a sentence of great importance. The importance of this sentence also stems from the importance of the sentences “recommending” it. Thus, to get ranked highly and placed in a summary, a sentence must be similar to many sentences that are in turn also similar to many other sentences. This makes intuitive sense and allows the algorithms to be applied to any arbitrary new text.
>
> ```sh
> cd baselines/LexRank
> bash setup_and_run_example.sh
> ```

### SumBasic

You can modify parameters in `run_lexrank.py`. Either set `PERCENT` or set it to `None` and set `NUM_SENTENCES`.

```sh
cd baselines/SumBasic
bash run_ours.sh
```

> This repository was forked in Github.

## Analysis of Model Complexity

```sh
bash statistics.sh
```

---

## Trouble Shooting

* `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` (sometimes will happened while testing, not sure what cause this recently)
* [libstdc++.so.6: version `GLIBCXX_3.4.22' not found · Issue #13 · lhelontra/tensorflow-on-arm](https://github.com/lhelontra/tensorflow-on-arm/issues/13) (due to gensim => scipy, somehow only happen on GCR2)

Mead

* `text2cluster.pl`
  * 68: `open (OUTFILE, ">$dir/$dir.cluster") || die "Can't open cluster file\n";` => Can't open cluster file
    * `open (OUTFILE, ">$dir.cluster") || die "Can't open cluster file\n";` will work...

```txt
sh: 1:  /mnt/d/Program/BERTIterativeLabeling/Paper2PPT/baselines/mead/mead/bin/../data/MEAD_TEST/docsent: not found
FATAL: Feature Calculation returned 32512
```

* `./mead/bin/driver.pl`
* `./mead/bin/mead.pl`
  * 269: `Debug("Feature Calculation returned $ret",3,"Driver");`

```txt
not well-formed (invalid token) at line 200, column 44, byte 32039 at /mnt/d/Program/BERTIterativeLabeling/Paper2PPT/baselines/mead/mead/bin/feature-scripts/../../lib/XML/Parser.pm line 185.
FATAL: Feature Calculation returned 65280
```

* `lib/XML/Parser.pm`
  * 185: `$result = $expat->parse($arg);`

```txt
Cannot open DBM /mnt/d/Program/BERTIterativeLabeling/Paper2PPT/baselines/mead/mead/bin/../data/MEAD_TEST/docsent at /mnt/d/Program/BERTIterativeLabeling/Paper2PPT/baselines/mead/mead/bin/feature-scripts/../../lib/Essence/IDF.pm line 44.
FATAL: Feature Calculation returned 5376
```

* `lib/Essence/IDF.pm`
  * 44: `die "Cannot open DBM $dbmname"`

## Perl notes

* cpan
  * install YAML
  * install CPAN
  * reload cpan
  * install App::cpanminus
* cpanm
* instmodsh
  * l
