package fasttext;

import com.google.common.collect.MinMaxPriorityQueue;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static fasttext.util.io.IOUtils.readBoolean;
import static fasttext.util.io.IOUtils.readInt;

/**
 * Java FastText implementation
 * Only prediction (supervised, unsupervised) is implemented
 * Please use Cpp FastText for train, test and quantization
 */
public class FastText {

  public static int FASTTEXT_VERSION = 11;
  public static int FASTTEXT_FILEFORMAT_MAGIC_INT = 793712314;

  private final static Logger logger = Logger.getLogger(FastText.class.getName());

  private Args args;
  private Dictionary dict;

  private Model model;

  private Matrix input;
  private Matrix output;

  private QMatrix qinput;
  private QMatrix qoutput;

  private boolean quant;

  private Matrix wordVectors = null;

  private int tokenCount;

  public FastText() {}

  private boolean checkModel(InputStream is) throws IOException {
    int magic;
    int version;
    magic = readInt(is);
    if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT) {
      logger.error("Unhandled file format");
      return false;
    }
    version = readInt(is);
    if (version != FASTTEXT_VERSION) {
      logger.error("Input model version (" + version + ") doesn't match current version (" + FASTTEXT_VERSION + ")");
      return false;
    }
    return true;
  }

  public List<FastTextPrediction> predict(String s, int k) {
    List<Integer> words = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    List<FastTextPrediction> predictions = new ArrayList<>(dict.nLabels());
    dict.getLine(s, words, labels, model.rng());
    if (!words.isEmpty()) {
      Vector hidden = new Vector(args.getDimension());
      Vector output = new Vector(dict.nLabels());
      MinMaxPriorityQueue<Pair<Float, Integer>> modelPredictions = MinMaxPriorityQueue
        .orderedBy(new Model.HeapComparator<Integer>())
        .expectedSize(dict.nLabels())
        .create();
      int[] input = words.stream().mapToInt(i->i).toArray();
      model.predict(input, k, modelPredictions, hidden, output);
      for (Pair<Float, Integer> pred : modelPredictions) {
        logger.warn(pred.first() + " " + pred.last() + " ");
        predictions.add(new FastTextPrediction(dict.getLabel(pred.last()), pred.first()));
      }
      System.out.println("\n");
    }
    return predictions;
  }

  public List<FastTextPrediction> predict(String s) {
    return predict(s, 1);
  }

  public List<FastTextPrediction> predict(List<String> tokens, int k) {
    List<Integer> words = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    List<FastTextPrediction> predictions = new ArrayList<>(dict.nLabels());
    dict.getLine(tokens, words, labels, model.rng());
    if (!words.isEmpty()) {
      Vector hidden = new Vector(args.getDimension());
      Vector output = new Vector(dict.nLabels());
      MinMaxPriorityQueue<Pair<Float, Integer>> modelPredictions = MinMaxPriorityQueue
        .orderedBy(new Model.HeapComparator<Integer>())
        .expectedSize(dict.nLabels())
        .create();
      int[] input = words.stream().mapToInt(i->i).toArray();
      model.predict(input, k, modelPredictions, hidden, output);
      for (Pair<Float, Integer> pred : modelPredictions) {
        predictions.add(new FastTextPrediction(dict.getLabel(pred.last()), pred.first()));
      }
    }
    return predictions;
  }

  public List<FastTextPrediction> predict(List<String> tokens) {
    return predict(tokens, 1);
  }

  public Vector getWordVector(String word) {
    Vector vec = new Vector(args.getDimension());
    List<Integer> ngrams = dict.getNGrams(word);
    vec.zero();
    for (int it : ngrams) {
      if (quant) {
        logger.warn("Computing word vector from quantized model using approximations");
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

  public List<Vector> getWordVectors(List<String> words) {
    List<Vector> vecs = new ArrayList<>(words.size());
    for (String word : words) {
      vecs.add(getWordVector(word));
    }
    return vecs;
  }

  public Vector getSentenceVector(List<String> sentence) {
    Vector vec = new Vector(args.getDimension());
    Vector svec = new Vector(args.getDimension());
    svec.zero();
    for (String word : sentence) {
      getWordVector(word);
      vec.mul(1.0f / vec.norm());
      svec.addVector(vec);
    }
    svec.mul(1.0f / (float) sentence.size());
    return svec;
  }

  public List<Vector> getSentenceVectors(List<List<String>> sentences) {
    List<Vector> svecs = new ArrayList<>(sentences.size());
    for (List<String> s : sentences) {
      svecs.add(getSentenceVector(s));
    }
    return svecs;
  }

  public List<Vector> ngramVectors(String word) {
    List<Vector> vecs = new ArrayList<>();
    Vector vec = new Vector(args.getDimension());
    List<Integer> ngrams = dict.getNGrams(word);
    for (int i = 0; i < ngrams.size(); i++) {
      vec.zero();
      if (ngrams.get(i) >= 0) {
        if (quant) {
          logger.warn("Computing ngram model from quantized model using approximations");
          vec.addRow(qinput, ngrams.get(i));
        } else {
          vec.addRow(input, ngrams.get(i));
        }
      }
      vecs.add(vec);
    }
    return vecs;
  }

  public Vector textVector(String text) {
    List<Integer> tokens = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    Vector vec = new Vector(args.getDimension());
    vec.zero();
    dict.getLine(text, tokens, labels, model.rng());
    for (Integer token : tokens) {
      if (quant) {
        logger.warn("Computing text vector from quantized model using approximations");
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
        wordVectors.addRow(vec, i, 1.0f / norm);
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

  public List<FastTextSynonym> nn(String queryWord, int k) {
    Set<String> banSet = new HashSet<>();
    banSet.add(queryWord);
    Vector queryVec = getWordVector(queryWord);
    return findNN(queryVec, k, banSet);
  }

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

  public void supervised(Model model, float lr, List<Integer> line, List<Integer> labels) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void cbow(Model model, float lr, List<Integer> line) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void skipgram(Model model, float lr, List<Integer> line) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void quantize(Args qargs) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void test(InputStream is, int k) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  private void trainThread(int threadId) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void train(Args args) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void loadVectors(String filename) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void saveOutput() throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void saveModel() throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void loadModel(String filename) throws IOException {
    logger.info("Loading FastText model from: " + filename);
    File f = new File(filename);
    if (!f.canRead()) {
      logger.error("Model file cannot be opened for loading");
      throw new IllegalArgumentException("Model file cannot be opened for loading");
    }
    BufferedInputStream is = null;
    try {
      is = new BufferedInputStream(new FileInputStream(f));
      loadModel(is);
    } finally {
      if (is != null) {
        is.close();
      }
    }
  }

  public void loadModel(InputStream is) throws IOException {
    long start = System.currentTimeMillis();

    if (!checkModel(is)) {
      throw new IllegalArgumentException("Model file has wrong file format");
    }

    logger.info("Loading model arguments");
    args = new Args();
    args.load(is);

    logger.info("Loading dictionary");
    dict = new Dictionary(args);
    dict.load(is);

    boolean quantInput = readBoolean(is);
    if (quantInput) {
      logger.info("Model is quantized. Loading quantized input matrix");
      quant = true;
      qinput = new QMatrix();
      qinput.load(is);
      logger.info("... done");
    } else {
      logger.info("Loading input matrix");
      quant = false;
      input = new Matrix();
      input.load(is);
      logger.info("... done");
    }

    boolean qout = readBoolean(is);
    args.setQOut(qout);
    if (quant && args.getQOut()) {
      logger.info("Classifier is quantized. Loading quantized output matrix");
      qoutput = new QMatrix();
      qoutput.load(is);
      logger.info("... done");
    } else {
      logger.info("Loading output matrix");
      output = new Matrix();
      output.load(is);
      logger.info("... done");
    }

    logger.info("Loading ML model");
    model = new Model(input, output, args, 0);
    model.setQuantization(qinput, qoutput, quant, args.getQOut());

    if (args.getModel() == Args.ModelName.SUP) {
      logger.info("Initiating supervised model");
      model.setTargetCounts(dict.getCounts(Dictionary.EntryType.LABEL).stream().mapToLong(i->i).toArray());
    } else {
      logger.info("Initiating unsupervised model");
      model.setTargetCounts(dict.getCounts(Dictionary.EntryType.WORD).stream().mapToLong(i->i).toArray());
    }

    long end = System.currentTimeMillis();
    long took = (end - start) / 1000;

    logger.info("FastText model loaded (took " + took + "s).");
  }

}
