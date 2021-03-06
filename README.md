# Official repository for "[A little bit more: Bitplane-wise bit-depth recovery](https://arxiv.org/abs/2005.01091)"

### BibTeX
```BibTeX
@misc{punnappurath2020little,
    title={A Little Bit More: Bitplane-Wise Bit-Depth Recovery},
    author={Abhijith Punnappurath and Michael S. Brown},
    year={2020},
    eprint={2005.01091},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
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
- To test 4 to 8-bit recovery on the [Kodak dataset](http://r0k.us/graphics/kodak/) (which has already been downloaded to ./data/Test/Kodak) using our D16 model, run
  
  ```
  python test.py --set_names Kodak --type_8_or_16 0 --quant 4 --quant_end 8 --dep 16 
  ```
  
  (Note: --type_8_or_16 flag -> 0 = 8-bit images, 1 = 16-bit images in the corresponding --set_names folder)

- To test 6 to 16-bit recovery on [this sample image](https://media.xiph.org/sintel/sintel-1k-png16/00017023.png) from the Sintel dataset (which has already been downloaded to ./data/Test/Sintel_sample) using our D4 model and save the result, run

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
1. To reproduce the numbers in Table S1, follow [these instructions](./download_data_and_test/download_MS_COCO_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S1_MS_COCO.txt) to produce the outputs.
2. To reproduce the numbers in Table S2, follow [these instructions](./download_data_and_test/download_TESTIMAGES_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S2_TESTIMAGES_800.txt) to produce the outputs.
3. To reproduce the numbers in Table S3, and run [this code](./download_data_and_test/test_table_S3.txt).
4. To reproduce the numbers in Table S4, and run [this code](./download_data_and_test/test_table_S4_Adobe_MIT.txt).
5. To reproduce the numbers in Table S5, follow [these instructions](./download_data_and_test/download_BSD_dataset.txt) to download the data, and run [this code](./download_data_and_test/test_table_S5.txt) to produce the outputs.
