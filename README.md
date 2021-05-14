## ECE-9123 DL final project
### For abstractive text summarization: 
- BART [link](https://arxiv.org/abs/1910.13461)
- PEGASUS [link](https://arxiv.org/abs/1912.08777)

See `final_project.ipynb`. For the pytorch checkpoint of BART fine-tuned on Arxiv scientific dataset, see `bart_split_15.pt` in [here](https://drive.google.com/drive/folders/1bBj6XNVjm5GI8es275EFcRscYyebfdEF?usp=sharing).

### For extractive text summarization:
Towards Topic-Aware Slide Generation For Academic Papers With Unsupervised Mutual Learning [link](https://www.microsoft.com/en-us/research/uploads/prod/2021/01/AAAI_2021_Preprint__Copy_.pdf)

Source code is in `Paper2PPT/`, the evaulation metric is in `Paper2PPT/log.txt`. For the model checkpoint, see [here](https://drive.google.com/drive/folders/1bBj6XNVjm5GI8es275EFcRscYyebfdEF?usp=sharing).

The code in `Paper2PPT` is mostly taken from the author's repo [(see here)](https://github.com/daviddwlee84/TopicAwarePaperSlideGeneration) with some modification.

To run the model, execute `Paper2PPT/neusum_pt/run_all.sh`, which which starts to train model for each of these 4 topic: `baseline`, `future`, `contribution` andd `dataset`.

To evaluate the mode, execute `Paper2PPT/neusum_pt/test_all.sh`.