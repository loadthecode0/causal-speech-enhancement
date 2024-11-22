# causal-speech-enhancement
Project repo for Deep Learning 02456 at DTU

4 nov

- [x] todos on slack
- [x] hpc init
- [ ] and training
- [ ] downsizing data
- [ ] 1-2 weeks -> setup data and training loop, reduce loss, gap bw two models
- [ ] then tinker

11 Nov
- [ ] Resampling, downsampling -> why and how?
- [ ] Loss functions
- [ ] Conv TAS Net
- try batch size = 1
- torch compile if too slow
- wav2vec is open source - on outputs- good starting poit -similar to what they did before
- so we need to basically figure out a good way to do the transfer learning
 
## References
EARS_DATASET
```
@inproceedings{richter2024ears,
  title={{EARS}: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation},
  author={Richter, Julius and Wu, Yi-Chiao and Krenn, Steven and Welker, Simon and Lay, Bunlong and Watanabe, Shinjii and Richard, Alexander and Gerkmann, Timo},
  booktitle={ISCA Interspeech},
  year={2024}
}
```
## Reference Links:
1. HPC Related:
https://www.hpc.dtu.dk/?page_id=59#scratch
https://www.hpc.dtu.dk/?page_id=927#Scratch_usage
https://www.hpc.dtu.dk/?page_id=2534
https://www.gbar.dtu.dk/index.php/faq
https://www.hpc.dtu.dk/?page_id=2501
https://www.hpc.dtu.dk/?page_id=4317
https://www.hpc.dtu.dk/?page_id=59#scratch
https://www.hpc.dtu.dk/?page_id=927#Scratch_usage
https://arxiv.org/abs/1809.07454

3. Dataset:
https://github.com/sp-uhh/ears_benchmark
https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

5. Model Architecture Related:
https://github.com/pytorch/audio/blob/main/src/torchaudio/models/conv_tasnet.py
https://docs.python.org/3/library/glob.html

4. Literature:
https://arxiv.org/pdf/2408.11842v2
https://arxiv.org/abs/1809.07454 


### Aastha says: Kill me already >.<