# fastText4j

Java port of C++ version of Facebook Research [fastText][fasttext].

This implementation supports prediction for supervised and unsupervised models, whether they are quantized or not.
Please use C++ version of fastText for train, test and quantization.

## Implementations

This library offers two implementations of fastText library:
* A regular in-memory model, which is a simple port of the C++ version
* A memory-mapped version of the model, allowing a lower RAM usage

This second implementation relies on memory-mapped IO for reading the dictionary and the input matrix.


**In order to be able to use this second implementation, you will have to convert your fastText model to the appropriate mmapped model format.**


## Converting fastText model to memory-mapped model

Use the following command to obtain a zip archive containing an executable jar with dependencies and a bash script to launch the jar:

```
$ mvn install -Papp
```

The zip archive will be built in the `app` folder. You can then use this distribution to run the mmap model conversion:

```
$ cd app
$ unzip fasttext4j-app.zip
$ ./fasttext-mmap.sh -input <fastText-model-path> -output <fasttext-mmap-model-path>
```

## References

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

### FastText.zip: Compressing text classification models

[3] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [*FastText.zip: Compressing text classification models*](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```

(\* These authors contributed equally.)

[fasttext]: <https://github.com/facebookresearch/fastText/>