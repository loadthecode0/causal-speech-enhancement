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
