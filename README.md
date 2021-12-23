# Official repository for "A little bit more: Bitplane-wise bit-depth recovery" [[IEEE]](https://ieeexplore.ieee.org/document/9606525) [[arXiv]](https://arxiv.org/abs/2005.01091)

### BibTeX
```BibTeX
@ARTICLE{9606525,
  author={Punnappurath, Abhijith and Brown, Michael S},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Little Bit More: Bitplane-Wise Bit-Depth Recovery}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3125692}}
```

***

### Requirements
This code was tested on
 - Python 3.6
 - Tensorflow 1.12.0
 - Keras 2.2.4
 - scikit-image 0.15.0
 - opencv 3.3.1
 
Or create a new [conda](https://conda.io) environment with

    conda env create -f environment.yml
    
and activate it with

    conda activate bitmore

***

### Testing code
#### Sample usage
- To test 4 to 8-bit recovery on the [Kodak dataset](http://r0k.us/graphics/kodak/) (which has already been downloaded to `./data/Test/Kodak`) using our D16 model, run
  
  ```
  python test.py --set_names Kodak --type_8_or_16 0 --quant 4 --quant_end 8 --dep 16 
  ```
  
  (Note: Set `--type_8_or_16 0` for 8-bit images and  `--type_8_or_16 1` for 16-bit images according to the corresponding `--set_names` folder)

- To test 6 to 16-bit recovery on [this sample image](https://media.xiph.org/sintel/sintel-1k-png16/00017023.png) from the Sintel dataset (which has already been downloaded to `./data/Test/Sintel_sample`) using our D4 model and save the result, run

  ```
  python test.py --set_names Sintel_sample --type_8_or_16 1 --quant 6 --quant_end 16 --dep 4 --save_result 1  
  ```
- To test 4 to 8-bit recovery on the Kodak dataset and on the sample image from the Sintel dataset using our D4 model, run
  
  ```
  python test.py --set_names Kodak,Sintel_sample --type_8_or_16 0,1 --quant 4 --quant_end 8 --dep 4 
  ```
  
***


### Main paper results
1. To reproduce the numbers in Table I, run [this code](./download_data_and_test/download_Sintel_test_set.m) to download the data, and [this code](./download_data_and_test/test_table_I_Sintel.txt) to produce the outputs.
2. To reproduce the numbers in Table II, run [this code](./download_data_and_test/download_Adobe_MIT_test_set.m) to download the data, and [this code](./download_data_and_test/test_table_II_Adobe_MIT.txt) to produce the outputs.     
     - Note: Downloading and preparing Adobe MIT test data can take a while!
3. To reproduce the numbers in Table III, follow [these instructions](./download_data_and_test/download_TESTIMAGES_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_III_TESTIMAGES_1200.txt) to produce the outputs.
4. To reproduce the numbers in Table IV on the [Kodak dataset](http://r0k.us/graphics/kodak/) which has already been downloaded to ./data/Test/Kodak, run [this code](./download_data_and_test/test_table_IV_Kodak.txt).
5. To reproduce the numbers in Table V, follow [these instructions](./download_data_and_test/download_ESPL_v2_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_V_ESPL_v2.txt) to produce the outputs.


***

### Supplementary material results
1. To reproduce the numbers in Table S4, follow [these instructions](./download_data_and_test/download_MS_COCO_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S4_MS_COCO.txt) to produce the outputs.
2. To reproduce the numbers in Table S5, follow [these instructions](./download_data_and_test/download_TESTIMAGES_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S5_TESTIMAGES_800.txt) to produce the outputs.
3. To reproduce the numbers in Table S6, and run [this code](./download_data_and_test/test_table_S6.txt).
4. To reproduce the numbers in Table S7, and run [this code](./download_data_and_test/test_table_S7_Adobe_MIT.txt).
5. To reproduce the numbers in Table S8, follow [these instructions](./download_data_and_test/download_BSD_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S8.txt) to produce the outputs.

***

### Training code 
- Run [this code](./download_data_and_test/download_train_val_data.m) to download training data.
  - Note: Downloading and preparing training data can take a while! 
- To train a model that predicts the 5<sup>th</sup> bit, run
  ```
  python train.py --quant 4 
  ```
  - Note: The images are quantized to 4 bits, and the model is trained to predict the (4+1)<sup>th</sup> bit.
- To perform 4 to 8-bit recovery, train four separate models as
  ```
  python train.py --quant 4 
  python train.py --quant 5 
  python train.py --quant 6 
  python train.py --quant 7 
  ```
- The number of residual units is set to 4 by default, i.e., D4 model. To train the D16 model, pass `--dep 16` as argument.
- The models are saved to `./models`.
