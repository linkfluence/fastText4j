# fastText4j

Java port of C++ version of Facebook Research [fastText][fasttext].

This implementation supports prediction for supervised and unsupervised models, whether they are quantized or not.
Please use C++ version of fastText for train, test and quantization.

## Supported fastText version

fastText4j currently supports models from fastText 1b version (support of subwords for supervised models).

## Implementation

This library offers two implementations of fastText library:
* A regular in-memory model, which is a simple port of the C++ version
* A memory-mapped version of the model, allowing a lower RAM usage

This second implementation relies on memory-mapped IO for reading the dictionary and the input matrix.

**Note: In order to be able to use this second implementation, you will have to convert your fastText model 
to the appropriate memory-mapped model format.**

## Requirements

To build and use fastText4j, you will need: 
* Java 8 or above
* Maven

## Building fastText4j

This project uses maven as build tool. To build fastText4j, use the following:

``` shell
$ mvn package
```

## Memory-mapped model

### Converting fastText model to memory-mapped model

You can convert both non-quantized and quantized fastText models to memory-mapped models. 
You will have to use the binary model `.bin` or `.ftz` for the conversion step.  

Use the following command to obtain a zip archive containing an executable jar 
with dependencies and a bash script to launch the jar:

``` shell
$ mvn install -Papp
```

The zip archive will be built in the `app` folder. You can then use this distribution to run the mmap model conversion:

``` shell
$ cd app
$ unzip fasttext4j-app.zip
$ ./fasttext-mmap.sh -input <fastText-model-path> -output <fasttext-mmap-model-path>
```

### Using the memory-mapped model

#### Model loading

Loading a memory-mapped model with fastText4j is completely transparent. 
You just have to provide the path `<fasttext-mmap-model-path>` that you passed to the `output` parameter above.

#### Closing the model

When loading a memory-mapped model, fastText4j internally opens FileChannels that will need to be closed.
To properly close your memory-mapped model, you will need to call the `.close()` method on your FastText object.

#### Multithreaded use

The memory-mapped FastText may only be used from one thread, because it is not thread safe 
(it keeps internal state like the mapped file positions). 

To allow multithreaded use, every FastText instance must be cloned before being used in another thread. 

## FastText references

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
