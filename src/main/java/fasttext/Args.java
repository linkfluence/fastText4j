package fasttext;

import fasttext.store.InputStreamFastTextInput;
import fasttext.store.OutputStreamFastTextOutput;

import java.io.IOException;

public class Args {

  public enum LossName {

    HS(1), NS(2), SOFTMAX(3);
    private int value;

    LossName(int value) {
      this.value = value;
    }

    public int getValue() {
      return this.value;
    }

    public static LossName fromValue(int value) throws IllegalArgumentException {
      try {
        value -= 1;
        return LossName.values()[value];
      } catch (ArrayIndexOutOfBoundsException e) {
        throw new IllegalArgumentException("Unknown loss_name enum value :" + value);
      }
    }

  }

  public enum ModelName {

    CBOW(1), SG(2), SUP(3);
    private int value;

    ModelName(int value) {
      this.value = value;
    }

    public int getValue() {
      return this.value;
    }

    public static ModelName fromValue(int value) throws IllegalArgumentException {
      try {
        value -= 1;
        return ModelName.values()[value];
      } catch (ArrayIndexOutOfBoundsException e) {
        throw new IllegalArgumentException("Unknown model_name enum value :" + value);
      }
    }

  }

  private final int dim;
  private final int ws;
  private final int epoch;
  private final int minCount;
  private final int neg;
  private final int wordNgrams;
  private final LossName loss;
  private final ModelName model;
  private final int bucket;
  private final int minn;

  private int maxn;

  private final int lrUpdateRate;
  private final double t;

  private String label = "__label__";
  private int verbose = 2;
  private boolean qout = false;

  // custom fastText4j param to handle backward compatibility
  private boolean useMaxVocabularySize = false;

  private Args(int dim, int ws, int epoch, int minCount, int neg, int wordNgrams,
               LossName loss, ModelName model, int bucket, int minn, int maxn,
               int lrUpdateRate, double t) {
    this.dim = dim;
    this.ws = ws;
    this.epoch = epoch;
    this.minCount = minCount;
    this.neg = neg;
    this.wordNgrams = wordNgrams;
    this.loss = loss;
    this.model = model;
    this.bucket = bucket;
    this.minn = minn;
    this.maxn = maxn;
    this.lrUpdateRate = lrUpdateRate;
    this.t = t;
  }

  public int getDimension() {
    return this.dim;
  }

  public int getWindowSize() {
    return this.ws;
  }

  public int getEpoch() {
    return this.epoch;
  }

  public int getMinCount() {
    return this.minCount;
  }

  public int getNeg() {
    return this.neg;
  }

  public int getWordNGrams() {
    return this.wordNgrams;
  }

  public LossName getLoss() {
    return this.loss;
  }

  public ModelName getModel() {
    return this.model;
  }

  public int getBucketNumber() {
    return this.bucket;
  }

  public int getMinn() {
    return this.minn;
  }

  public int getMaxn() {
    return this.maxn;
  }

  public void setMaxn(int maxn) { this.maxn = maxn; }

  public int getLearningRateUpdateRate() {
    return this.lrUpdateRate;
  }

  public double getSamplingTreshold() {
    return this.t;
  }

  public String getLabelPrefix() {
    return this.label;
  }

  public void setLabelPrefix(String label) { this.label = label; }

  public int getVerboseLevel() {
    return this.verbose;
  }

  public void setVerboseLevel(int verbose) {
    this.verbose = verbose;
  }

  public boolean getQOut() {
    return this.qout;
  }

  public void setQOut(boolean qout) {
    this.qout = qout;
  }

  public boolean getUseMaxVocabularySize() { return this.useMaxVocabularySize; }

  public void setUseMaxVocabularySize(boolean useMaxVocabularySize) {
    this.useMaxVocabularySize = useMaxVocabularySize;
  }

  public void save(OutputStreamFastTextOutput os) throws IOException {
    os.writeInt(dim);
    os.writeInt(ws);
    os.writeInt(epoch);
    os.writeInt(minCount);
    os.writeInt(neg);
    os.writeInt(wordNgrams);
    os.writeInt(loss.getValue());
    os.writeInt(model.getValue());
    os.writeInt(bucket);
    os.writeInt(minn);
    os.writeInt(maxn);
    os.writeInt(lrUpdateRate);
    os.writeDouble(t);
  }

  public static Args load(InputStreamFastTextInput is) throws IOException {
    int dim = is.readInt();
    int ws = is.readInt();
    int epoch = is.readInt();
    int minCount = is.readInt();
    int neg = is.readInt();
    int wordNgrams = is.readInt();
    LossName loss = LossName.fromValue(is.readInt());
    ModelName model = ModelName.fromValue(is.readInt());
    int bucket = is.readInt();
    int minn = is.readInt();
    int maxn = is.readInt();
    int lrUpdateRate = is.readInt();
    double t = is.readDouble();
    return new Args(dim, ws, epoch, minCount, neg, wordNgrams,
      loss, model, bucket, minn, maxn, lrUpdateRate, t);
  }

}