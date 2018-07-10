## Build

This project uses maven as build tool. To build fastText4j, use the following:

``` shell
$ mvn package
```

## Usage

**FastText4j implementation only supports prediction for supervised and unsupervised models. Please use C++ version of fastText for train, test and quantization.**

You can use your fastText models with fastText4j for both supervised and unsupervised usages.

Here is a sample code on how to use fastText4j in your java projects.

``` java
import fasttext.FastText;
import fasttext.FastTextPrediction;
import fasttext.FastTextSynonym;
import fasttext.Vector;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Example {

  static public void main(String[] args) {

    String path = args[0];
    FastText model = null;

    try {

      /* First you will have to load your model */
      model = FastText.loadModel(path);


      /* Using fastText4j for supervised classification */

      FastTextPrediction singlePrediction = model.predict(Arrays.asList("hello", "world", "!"));

      List<FastTextPrediction> predictions = model.predictAll(Arrays.asList("hello", "world", "!"));


      /* Using fastText4j in an unsupervised fashion */

      // Nearest neighbors queries
      List<FastTextSynonym> nearestNeighbors = model.nn("hello", 10);

      System.out.println("Nearest neighbors of \"hello\" are: ");
      for (FastTextSynonym s : nearestNeighbors) {
        System.out.println(s.word() + " -> " + s.cosineSimilarity());
      }

      // Analogy queries
      List<FastTextSynonym> analogies = model.analogies("berlin", "germany", "france", 10);

      System.out.println("Analogies: \"berlin\" + \"germany\" - \"france\" : ");
      for (FastTextSynonym s : analogies) {
        System.out.println(s.word() + " -> " + s.cosineSimilarity());
      }

      // Obtaining a word vector
      Vector wv = model.getWordVector("world");

      // Obtaining word vectors
      List<Vector> wvs = model.getWordVectors(Arrays.asList("hello", "world", "!"));

      // Obtaining sentence (document) embeddings
      Vector sv = model.getSentenceVector(Arrays.asList("hello", "world", "!"));

    } catch (Exception e) {

      System.out.println("Oops something went wrong. Exception: " + e.getMessage());

    } finally {

      // Closing is only mandatory for memory-mapped models
      if (model != null) {
        try {
          model.close();
        } catch (IOException e) {
          System.out.println("Error while closing fastText model");
        }
      }

    }

  }

}
```

## Convert a model to memory mapped version

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
