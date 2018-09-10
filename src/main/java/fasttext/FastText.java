package fasttext;

import com.google.common.collect.MinMaxPriorityQueue;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import fasttext.mmap.MMapDictionary;
import fasttext.mmap.MMapMatrix;
import fasttext.mmap.MMapQMatrix;
import fasttext.store.InputStreamFastTextInput;
import fasttext.store.MMapFile;
import fasttext.store.OutputStreamFastTextOutput;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.io.*;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.*;

/**
 * Java FastText implementation.
 *
 * <p>Only prediction (supervised, unsupervised) is implemented.
 * Please use Cpp FastText for train, test and quantization.
 *
 * <p>Two implementations of the FastText model are available. One of them
 * is an in-memory implementation, which is a simple port of the C++ version.
 * The second is a memory-mapped version, using memory-mapped IO for reading
 * the dictionary and the input matrix.
 *
 * <p>The memory-mapped {@code FastText} may only be used from one thread,
 * because it is not thread safe (it keeps internal state like mmap file position).
 * To allow multithreaded use, every {@code FastText} instance must be cloned before
 * used in another thread. Subclasses must therefore implement {@link #clone()},
 * returning a new {@code FastText} which operates on the same underlying
 * resources, but positioned independently.
 *
 * <p>The memory-mapped {@code FastText} should be closed to properly close
 * the underlying resources.
 *
 * <p>The memory-mapped {@code FastText} requires the regular fastText binary model
 * to be converted.
 * {@see saveAsMemoryMappedModel(String)}
 */
public class FastText {

  public static int FASTTEXT_VERSION = 12; /* Version 1b */
  public static int FASTTEXT_FILEFORMAT_MAGIC_INT = 793712314;

  private final static Logger logger = Logger.getLogger(FastText.class.getName());

  private final Args args;
  private final int version;

  private BaseDictionary dict;
  private Model model;

  private ReadableMatrix input;
  private Matrix output;

  private ReadableQMatrix qinput;
  private QMatrix qoutput;

  private final boolean quant;
  private final boolean mmap;

  private Matrix wordVectors = null;

  private FastText(Args args,
                   int version,
                   BaseDictionary dict,
                   ReadableMatrix input,
                   Matrix output,
                   boolean quant,
                   ReadableQMatrix qinput,
                   QMatrix qoutput,
                   boolean mmap) {
    this.args = args;
    this.version = version;
    this.dict = dict;
    this.input = input;
    this.output = output;
    this.quant = quant;
    this.qinput = qinput;
    this.qoutput = qoutput;
    this.mmap = mmap;
    this.model = buildModel(args, input, output, quant, args.getQOut(), qinput, qoutput);
  }

  private Model buildModel(Args args,
                           ReadableMatrix input,
                           Matrix output,
                           boolean quant,
                           boolean qout,
                           ReadableQMatrix qinput,
                           QMatrix qoutput) {
    Model m = new Model(args, 0, input, output, quant, qout, qinput, qoutput);
    if (args.getModel() == Args.ModelName.SUP) {
      m.setTargetCounts(Longs.toArray(dict.getCounts(Dictionary.EntryType.LABEL)));
    } else {
      m.setTargetCounts(Longs.toArray(dict.getCounts(Dictionary.EntryType.WORD)));
    }
    return m;
  }

  public BaseDictionary getDictionary() {
    return this.dict;
  }

  public Args getArgs() { return this.args; }

  private void signModel(int magic, int version, OutputStreamFastTextOutput os) throws IOException {
    os.writeInt(magic);
    os.writeInt(version);
  }

  private static boolean checkModel(int magic, int version) throws IOException {
    if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT) {
      logger.error("Unhandled file format");
      return false;
    }
    if (version > FASTTEXT_VERSION) {
      logger.error("Input model version (" + version + ") doesn't match current version (" + FASTTEXT_VERSION + ")");
      return false;
    }
    return true;
  }

  private void predict(List<Integer> words, List<FastTextPrediction> predictions, int k, float threshold) {
    if (!words.isEmpty()) {
      Vector hidden = new Vector(args.getDimension());
      Vector output = new Vector(dict.nLabels());
      MinMaxPriorityQueue<Pair<Float, Integer>> modelPredictions = MinMaxPriorityQueue
        .orderedBy(new Model.HeapComparator<Integer>())
        .expectedSize(dict.nLabels())
        .create();
      int[] input = Ints.toArray(words);
      if (words.isEmpty()) {
        return;
      }
      model.predict(input, k, threshold, modelPredictions, hidden, output);
      while (!modelPredictions.isEmpty()) {
        Pair<Float, Integer> pred = modelPredictions.pollFirst();
        predictions.add(new FastTextPrediction(dict.getLabel(pred.last()), pred.first()));
      }
    }
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns the prediction with highest probability.
   * k and threshold will be applied together to determine the returned labels.
   * @param s input document
   * @param k controls the number of returned labels. A choice of 5, will return the 5 most probable labels
   * @param threshold filters the returned labels by a threshold on probability. A choice of 0.5 will return labels with at least 0.5 probability
   * @return top predictions (max k) with probability above threshold
   */
  public List<FastTextPrediction> predict(String s, int k, float threshold) {
    List<Integer> words = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    List<FastTextPrediction> predictions = new ArrayList<>(dict.nLabels());
    dict.getLine(s, words, labels);
    predict(words, predictions, k, threshold);
    return predictions;
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns the prediction with highest probability if above the given threshold, otherwise returns null.
   * @param s input document
   * @param k controls the number of returned labels. A choice of 5, will return the 5 most probable labels
   * @return k top prediction
   */
  public List<FastTextPrediction> predict(String s, int k) {
    return predict(s, k, 0f);
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns the prediction with highest probability if above the given threshold, otherwise returns null.
   * @param s input document
   * @param threshold filters the returned label by a threshold on probability
   * @return top prediction
   */
  public FastTextPrediction predict(String s, float threshold) {
    List<FastTextPrediction> predictions = predict(s, 1, threshold);
    if (!predictions.isEmpty()) {
      return predictions.get(0);
    } else {
      return null;
    }
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns the prediction with highest probability.
   * @param s input document
   */
  public FastTextPrediction predict(String s) {
    return predict(s, 0f);
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns prediction on all labels which have probability above threshold.
   * @param s input document
   * @param threshold filters the returned label by a threshold on probability
   * @return all predictions with probability above threshold
   */
  public List<FastTextPrediction> predictAll(String s, float threshold) {
    return predict(s, dict.nLabels(), threshold);
  }

  /**
   * Classifies a document represented as a String with whitespace separated tokens.
   * Returns prediction on all labels.
   * @param s input document
   * @return predictions on all labels
   */
  public List<FastTextPrediction> predictAll(String s) { return predictAll(s, 0f); }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns the predictions with highest probability.
   * k and threshold will be applied together to determine the returned labels.
   * @param tokens input document
   * @param k number of predictions
   * @param threshold filters the returned label by a threshold on probability
   * @return top predictions (max k) with probability above threshold
   */
  public List<FastTextPrediction> predict(List<String> tokens, int k, float threshold) {
    List<Integer> words = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    List<FastTextPrediction> predictions = new ArrayList<>(dict.nLabels());
    dict.getLine(tokens, words, labels);
    predict(words, predictions, k, threshold);
    return predictions;
  }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns the predictions with highest probability.
   * k and threshold will be applied together to determine the returned labels.
   * @param tokens input document
   * @param k number of predictions
   * @return top predictions (max k) with probability above threshold
   */
  public List<FastTextPrediction> predict(List<String> tokens, int k) {
    return predict(tokens, k, 0f);
  }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns the prediction with highest probability if above threshold, otherwise returns null.
   * @param tokens input document
   * @param threshold filters the returned label by a threshold on probability
   * @return top prediction if probability above threshold
   */
  public FastTextPrediction predict(List<String> tokens, float threshold) {
    List<FastTextPrediction> predictions = predict(tokens, 1, threshold);
    if (!predictions.isEmpty()) {
      return predictions.get(0);
    } else {
      return null;
    }
  }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns the prediction with highest probability.
   * @param tokens input document
   * @return top prediction
   */
  public FastTextPrediction predict(List<String> tokens) {
    return predict(tokens, 0f);
  }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns prediction on all labels filtered by threshold on probability.
   * @param tokens input document
   * @param threshold filters the returned label by a threshold on probability
   * @return predictions on all labels with probability above threshold
   */
  public List<FastTextPrediction> predictAll(List<String> tokens, float threshold) {
    return predict(tokens, dict.nLabels(), threshold);
  }

  /**
   * Classifies a document represented as a list of tokens.
   * Returns prediction on all labels.
   * @param tokens input document
   * @return predictions on all labels
   */
  public List<FastTextPrediction> predictAll(List<String> tokens) {
    return predictAll(tokens, 0f);
  }

  /**
   * Gives the vector of a word.
   * @param word
   * @return word vector
   */
  public Vector getWordVector(String word) {
    Vector vec = new Vector(args.getDimension());
    List<Integer> ngrams = dict.getSubwords(word);
    vec.zero();
    for (int it : ngrams) {
      if (quant) {
        vec.addRow(qinput, it);
      } else {
        vec.addRow(input, it);
      }
    }
    if (ngrams.size() > 0) {
      vec.mul(1.0f / (float) ngrams.size());
    }
    return vec;
  }

  /**
   * Gives the vectors of a list of words.
   * @param words
   * @return word vectors
   */
  public List<Vector> getWordVectors(List<String> words) {
    List<Vector> vecs = new ArrayList<>(words.size());
    for (String word : words) {
      vecs.add(getWordVector(word));
    }
    return vecs;
  }

  /**
   * Gives the vector of a sentence.
   * @param sentence Tokenized sentence
   * @return sentence vector
   */
  public Vector getSentenceVector(List<String> sentence) {
    Vector svec = new Vector(args.getDimension());
    svec.zero();
    if (args.getModel() == Args.ModelName.SUP) {
      List<Integer> tokens = new ArrayList<>();
      List<Integer> labels = new ArrayList<>();
      dict.getLine(sentence, tokens, labels);
      for (int i = 0; i < tokens.size(); i++) {
        if (quant) {
          svec.addRow(qinput, tokens.get(i));
        } else {
          svec.addRow(input, tokens.get(i));
        }
      }
      if (!tokens.isEmpty()) {
        svec.mul(1.0f / (float) tokens.size());
      }
    } else {
      int count = 0;
      for (String word : sentence) {
        Vector vec = getWordVector(word);
        float norm = vec.norm();
        if (norm > 0) {
          vec.mul(1.0f / norm);
          svec.addVector(vec);
          count++;
        }
      }
      if (count > 0) {
        svec.mul(1.0f / (float) count);
      }
    }
    return svec;
  }

  /**
   * Gives the vectors corresponding to a list of sentences.
   * @param sentences List of tokenized sentence
   * @return List of sentence vectors
   */
  public List<Vector> getSentenceVectors(List<List<String>> sentences) {
    List<Vector> svecs = new ArrayList<>(sentences.size());
    for (List<String> s : sentences) {
      svecs.add(getSentenceVector(s));
    }
    return svecs;
  }

  /**
   * Gives the ngram vectors for a word.
   * @param word query word
   * @return ngram vectors
   */
  public List<Vector> ngramVectors(String word) {
    List<Vector> vecs = new ArrayList<>();
    Vector vec = new Vector(args.getDimension());
    List<Integer> ngrams = dict.getSubwords(word);
    for (int i = 0; i < ngrams.size(); i++) {
      vec.zero();
      if (ngrams.get(i) >= 0) {
        if (quant) {
          vec.addRow(qinput, ngrams.get(i));
        } else {
          vec.addRow(input, ngrams.get(i));
        }
      }
      vecs.add(vec);
    }
    return vecs;
  }

  /**
   * Gives the vector for a text (used in supervised settings).
   * @param text input text
   * @return text vector
   */
  public Vector textVector(String text) {
    List<Integer> tokens = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    Vector vec = new Vector(args.getDimension());
    vec.zero();
    dict.getLine(text, tokens, labels);
    for (Integer token : tokens) {
      if (quant) {
        vec.addRow(input, token);
      } else {
        vec.addRow(qinput, token);
      }
    }
    if (!tokens.isEmpty()) {
      vec.mul(1.0f / (float) tokens.size());
    }
    return vec;
  }

  /**
   * Gives the vectors for a list of text (used in supervised settings).
   * @param texts list of text
   * @return text vectors
   */
  public List<Vector> textVectors(List<String> texts) {
    List<Vector> vecs = new ArrayList<>(texts.size());
    for (String text : texts) {
      vecs.add(textVector(text));
    }
    return vecs;
  }

  private void precomputeWordVectors() {
    if (wordVectors == null) {
      logger.info("Precomputing word vectors...");
      wordVectors = new Matrix(dict.nWords(), args.getDimension());
      wordVectors.zero();
      for (int i = 0; i < dict.nWords(); i++) {
        String word = dict.getWord(i);
        Vector vec = getWordVector(word);
        float norm = vec.norm();
        if (norm > 0) {
          wordVectors.addRow(vec, i, 1.0f / norm);
        }
      }
      logger.info("Done. Word vectors precomputed.");
    } else {
      logger.debug("Word vectors are already precomputed.");
    }
  }

  private List<FastTextSynonym> findNN(Vector queryVec, int k, Set<String> banSet) {
    precomputeWordVectors();
    float queryNorm = queryVec.norm();
    if (Math.abs(queryNorm) < 1e-8) {
      queryNorm = 1.0f;
    }
    MinMaxPriorityQueue<Pair<Float, String>> heap = MinMaxPriorityQueue
      .orderedBy(new Model.HeapComparator<String>())
      .expectedSize(dict.nLabels())
      .create();
    for (int i = 0; i < dict.nWords(); i++) {
      String word = dict.getWord(i);
      float dp = wordVectors.dotRow(queryVec, i);
      heap.add(new Pair<>(dp / queryNorm, word));
    }
    List<FastTextSynonym> syns = new ArrayList<>();
    int i = 0;
    while (i < k && heap.size() > 0) {
      Pair<Float, String> synonym = heap.pollFirst();
      boolean banned = banSet.contains(synonym.last());
      if (!banned) {
        syns.add(new FastTextSynonym(synonym.last(), synonym.first()));
        i++;
      }
    }
    return syns;
  }

  /**
   * Nearest neighbor queries. Returns the closest words to a query word.
   * @param queryWord query word
   * @param k nearest neighbors number
   * @return k nearest neighbors
   */
  public List<FastTextSynonym> nn(String queryWord, int k) {
    Set<String> banSet = new HashSet<>();
    banSet.add(queryWord);
    Vector queryVec = getWordVector(queryWord);
    return findNN(queryVec, k, banSet);
  }

  /**
   * Word analogies. It takes a word triplet and returns the analogies.
   * @param queryA first word of the triplet
   * @param queryB second word of the triplet
   * @param queryC last word of the triplet
   * @param k number of analogies
   * @return k analogies
   */
  public List<FastTextSynonym> analogies(String queryA, String queryB, String queryC, int k) {
    Set<String> banSet = new HashSet<>();
    precomputeWordVectors();
    Vector buffer;
    Vector query = new Vector(args.getDimension());
    query.zero();
    // + A
    banSet.add(queryA);
    buffer = getWordVector(queryA);
    query.addVector(buffer, 1.0f);
    // - B
    banSet.add(queryB);
    buffer = getWordVector(queryB);
    query.addVector(buffer, -1.0f);
    // + C
    banSet.add(queryC);
    buffer = getWordVector(queryC);
    query.addVector(buffer, 1.0f);

    return findNN(query, k, banSet);
  }

  @Override
  public FastText clone() throws CloneNotSupportedException {
    FastText ft = this;
    if (mmap) {
      ft.dict = dict.clone();
      if (quant) {
        ft.qinput = qinput.clone();
      } else {
        ft.input = input.clone();
      }
      if (quant && args.getQOut()) {
        ft.qoutput = new QMatrix(qoutput);
      } else {
        ft.output = new Matrix(output);
      }
      ft.model = buildModel(ft.args, ft.input, ft.output,
        ft.quant, ft.args.getQOut(), ft.qinput, ft.qoutput);
    }
    return ft;
  }

  public void close() throws IOException {
    dict.close();
    if (quant) {
      qinput.close();
    } else {
      input.close();
    }
  }

  /**
   * Load fastText model from file path.
   * If the file is a directory, it tries to load a memory-mapped model.
   * If it is a single file, it tries to open an in-memory fastText model from binary model.
   */
  public static FastText loadModel(String filename) throws IOException {
    File f = new File(filename);
    if (f.isDirectory()) { // Memory-mapped format
      logger.info("Loading memory-mapped FastText model from: " + filename);
      File fb = new File(f.getAbsolutePath() + "/model.bin");
      File fq = new File(f.getAbsolutePath() + "/model.ftz");
      File modelFile;
      if (fb.exists()) {
        modelFile = fb;
      } else if (fq.exists()) {
        modelFile = fq;
      } else {
        throw new IOException("Model core file cannot be opened for loading");
      }
      Path dictFilePath = FileSystems.getDefault().getPath(f.getAbsolutePath() + "/dict.mmap");
      Path inFilePath = FileSystems.getDefault().getPath(f.getAbsolutePath() + "/in.mmap");
      MMapFile dictFile = new MMapFile(dictFilePath);
      MMapFile inFile = new MMapFile(inFilePath);
      try (InputStream is = new FileInputStream(modelFile)) {
        return loadModel(is, dictFile, inFile);
      }
    } else {
      logger.info("Loading in-memory FastText model from:" + filename);
      if (!f.canRead()) {
        throw new IllegalArgumentException("Model file cannot be opened for loading");
      }
      try (InputStream is = new FileInputStream(f)) {
        return loadModel(is);
      }
    }
  }

  private static FastText loadModel(InputStream in,
                                    MMapFile dictFile,
                                    MMapFile inputFile) throws IOException {
    try (InputStreamFastTextInput is = new InputStreamFastTextInput(in)) {
      int magic = is.readInt();
      int version = is.readInt();
      if (!checkModel(magic, version)) {
        throw new IllegalArgumentException("Model file has wrong file format");
      }
      long start = System.nanoTime();
      logger.info("Loading model arguments");
      Args args = Args.load(is);
      if (version == 11) {
        // backward compatibility: old supervised models do not use char ngrams.
        if (args.getModel() == Args.ModelName.SUP) {
          args.setMaxn(0);
        }
        // backward compatibility: use max vocabulary size as word2intSize.
        args.setUseMaxVocabularySize(true);
      }
      logger.info("Loading memory-mapped dictionary");
      MMapDictionary dict = MMapDictionary.load(args, dictFile);
      boolean quant = is.readBoolean();
      MMapMatrix wi = null;
      MMapQMatrix qwi = null;
      if (quant) {
        logger.info("Model is quantized. Loading quantized input matrix");
        qwi = MMapQMatrix.load(inputFile);
        logger.info("... done");
      } else {
        logger.info("Loading input matrix");
        wi = MMapMatrix.load(inputFile);
        logger.info("... done");
      }
      if (!quant && dict.isPruned()) {
        throw new IllegalArgumentException("Invalid model file.\n" +
            "Please download the updated model from www.fasttext.cc.\n");
      }
      boolean qout = is.readBoolean();
      args.setQOut(qout);
      Matrix wo = null;
      QMatrix qwo = null;
      if (quant && args.getQOut()) {
        logger.info("Classifier is quantized. Loading quantized output matrix");
        qwo = QMatrix.load(is);
        logger.info("... done");
      } else {
        logger.info("Loading output matrix");
        wo = Matrix.load(is);
        logger.info("... done");
      }
      logger.info("Initiating model");
      FastText fastText = new FastText(args, version, dict, wi, wo, quant, qwi, qwo, true);
      long end = System.nanoTime();
      double took = (end - start) / 1000000000d;
      logger.info(String.format(Locale.ENGLISH, "FastText model loaded (%.3fs)", took));
      return fastText;
    }

  }

  /**
   * Load a fastText model from a fastText binary format, reading from InputStream in.
   */
  public static FastText loadModel(InputStream in) throws IOException {
    try (InputStreamFastTextInput is = new InputStreamFastTextInput(in)) {
      int magic = is.readInt();
      int version = is.readInt();
      if (!checkModel(magic, version)) {
        throw new IllegalArgumentException("Model file has wrong file format");
      }
      long start = System.nanoTime();
      logger.info("Loading model arguments");
      Args args = Args.load(is);
      if (version == 11) {
        // backward compatibility: old supervised models do not use char ngrams.
        if (args.getModel() == Args.ModelName.SUP) {
          args.setMaxn(0);
        }
        // backward compatibility: use max vocabulary size as word2intSize.
        args.setUseMaxVocabularySize(true);
      }
      logger.info("Loading dictionary");
      Dictionary dict = Dictionary.load(args, is);
      boolean quant = is.readBoolean();
      Matrix wi = null;
      QMatrix qwi = null;
      if (quant) {
        logger.info("Model is quantized. Loading quantized input matrix");
        qwi = QMatrix.load(is);
        logger.info("... done");
      } else {
        logger.info("Loading input matrix");
        wi = Matrix.load(is);
        logger.info("... done");
      }
      if (!quant && dict.isPruned()) {
        throw new IllegalArgumentException("Invalid model file.\n" +
            "Please download the updated model from www.fasttext.cc.\n");
      }
      boolean qout = is.readBoolean();
      args.setQOut(qout);
      Matrix wo = null;
      QMatrix qwo = null;
      if (quant && args.getQOut()) {
        logger.info("Classifier is quantized. Loading quantized output matrix");
        qwo = QMatrix.load(is);
        logger.info("... done");
      } else {
        logger.info("Loading output matrix");
        wo = Matrix.load(is);
        logger.info("... done");
      }
      logger.info("Initiating model");
      FastText fastText = new FastText(args, version, dict, wi, wo, quant, qwi, qwo, false);
      long end = System.nanoTime();
      double took = (end - start) / 1000000000d;
      logger.info(String.format(Locale.ENGLISH, "FastText model loaded (%.3fs)", took));
      return fastText;
    }
  }

  private void ensureFilePath(File f) {
    if (f.exists()) {
      f.delete();
    }
    if (f.getParentFile() != null) {
      f.getParentFile().mkdirs();
    }
  }

  /**
   * Save the current fastText model to a fastText binary format to the specified file path.
   */
  public void saveModel(String filename) throws IOException {
    if (mmap) {
      throw new IllegalArgumentException("Cannot save memory-mapped model");
    }
    if (quant) {
      filename += ".ftz";
    } else {
      filename += ".bin";
    }
    File f = new File(filename);
    ensureFilePath(f);
    if (args.getVerboseLevel() > 1) {
      logger.info("Saving model to " + f.getCanonicalPath());
    }
    try (OutputStream os = new FileOutputStream(f)) {
      saveModel(os);
    }
    logger.info("... done");
  }

  /**
   * Save the current fastText model to a fastText binary format, writing to OutputStream out.
   */
  public void saveModel(OutputStream out) throws IOException {
    if (mmap) {
      throw new IllegalArgumentException("Cannot save memory-mapped model");
    }
    try (OutputStreamFastTextOutput os = new OutputStreamFastTextOutput(out)) {
      signModel(FASTTEXT_FILEFORMAT_MAGIC_INT, version, os);
      args.save(os);
      ((Dictionary) dict).save(os);
      os.writeBoolean(quant);
      if (quant) {
        ((QMatrix) qinput).save(os);
      } else {
        ((Matrix) input).save(os);
      }
      os.writeBoolean(args.getQOut());
      if (quant && args.getQOut()) {
        qoutput.save(os);
      } else {
        output.save(os);
      }
    }
  }

  /**
   * Save the current fastText model to a memory-mapped model.
   * @param dirName mmap model output path
   */
  public void saveAsMemoryMappedModel(String dirName) throws IOException {
    if (mmap) {
      throw new IllegalArgumentException("Cannot save from memory-mapped model");
    }

    long start = System.nanoTime();
    File dir = new File(dirName);

    String modelFilename = "model";
    String dictionaryFilename = "dict";
    String inputFilename = "in";

    if (quant) {
      modelFilename += ".ftz";
    } else {
      modelFilename += ".bin";
    }
    File modelFile = new File(dir.getAbsolutePath() + "/" + modelFilename);
    ensureFilePath(modelFile);
    if (args.getVerboseLevel() > 1) {
      logger.info("Saving core model to " + modelFile.getCanonicalPath());
    }
    try (OutputStreamFastTextOutput os = new OutputStreamFastTextOutput(new FileOutputStream(modelFile))) {
      signModel(FASTTEXT_FILEFORMAT_MAGIC_INT, version, os);
      args.save(os);
      os.writeBoolean(quant);
      os.writeBoolean(args.getQOut());
      if (quant && args.getQOut()) {
        qoutput.save(os);
      } else {
        output.save(os);
      }
    }

    dictionaryFilename += ".mmap";
    File dictFile = new File(dir.getAbsolutePath() + "/" + dictionaryFilename);
    ensureFilePath(dictFile);
    if (args.getVerboseLevel() > 1) {
      logger.info("Saving memory-mapped dictionary model to " + dictFile.getCanonicalPath());
    }
    try (FileOutputStream os = new FileOutputStream(dictFile)) {
      dict.saveToMMap(os);
    }

    inputFilename += ".mmap";
    File inputFile = new File(dir.getAbsolutePath() + "/" + inputFilename);
    ensureFilePath(inputFile);
    if (args.getVerboseLevel() > 1) {
      logger.info("Saving memory-mapped input matrix model to " + inputFile.getCanonicalPath());
    }
    try (FileOutputStream os = new FileOutputStream(inputFile)) {
      if (quant) {
        qinput.saveToMMap(os);
      } else {
        input.saveToMMap(os);
      }
    }
    long end = System.nanoTime();
    double took = (end - start) / 1000000000d;
    logger.info(String.format(Locale.ENGLISH,
        "FastText model successfully converted to mmapped (took %.3fs).", took));
  }


  public static void main(String[] args) throws Exception {

    Options options = new Options();

    Option input = new Option("i", "input", true, "input model path");
    input.setRequired(true);
    options.addOption(input);

    Option output = new Option("o", "output", true, "output model path");
    output.setRequired(true);
    options.addOption(output);

    CommandLineParser parser = new DefaultParser();
    HelpFormatter formatter = new HelpFormatter();
    CommandLine cmd;

    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      System.out.println(e.getMessage());
      formatter.printHelp("fasttext.FastText", options);

      System.exit(1);
      return;
    }

    String inputModelPath = cmd.getOptionValue("input");
    String baseOutputPath = cmd.getOptionValue("output");

    logger.info("Loading fastText model to convert...");
    FastText model = FastText.loadModel(inputModelPath);

    logger.info("Saving fastText model to memory-mapped model...");
    model.saveAsMemoryMappedModel(baseOutputPath);

  }

}