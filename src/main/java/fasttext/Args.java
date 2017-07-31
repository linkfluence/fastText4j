package fasttext;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static fasttext.util.io.IOUtils.readInt;
import static fasttext.util.io.IOUtils.readDouble;

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

  private double lr = 0.05f;
  private int dim = 100;
  private int ws = 5;
  private int epoch = 5;
  private int minCount = 5;
  private int minCountLabel = 0;
  private int neg = 5;
  private int wordNgrams = 1;
  private LossName loss = LossName.NS;
  private ModelName model = ModelName.SG;
  private int bucket = 2000000;
  private int minn = 3;
  private int maxn = 6;
  private int thread = 12;
  private int lrUpdateRate = 100;
  private double t = 1e-4;
  private String label = "__label__";
  private int verbose = 2;
  private String pretrainedVectors = "";
  private int saveOutput = 0;

  private boolean qout = false;
  private boolean retrain = false;
  private boolean qnorm = false;
  private int cutoff = 0;
  private int dsub = 2;

  public double getLearningRate() {
    return this.lr;
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

  public int getMinCountLabel() {
    return this.minCountLabel;
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

  public int getThread() {
    return this.thread;
  }

  public int getLearningRateUpdateRate() {
    return this.lrUpdateRate;
  }

  public double getSamplingTreshold() {
    return this.t;
  }

  public String getLabelPrefix() {
    return this.label;
  }

  public int getVerboseLevel() {
    return this.verbose;
  }

  public String getPretrainedVectors() {
    return this.pretrainedVectors;
  }

  public boolean getQOut() {
    return this.qout;
  }

  public void setQOut(boolean qout) {
    this.qout = qout;
  }

  public boolean getQNorm() {
    return this.qnorm;
  }

  public boolean getRetrain() {
    return this.retrain;
  }

  public int getCutOff() {
    return this.cutoff;
  }

  public int getSubDimension() {
    return this.dsub;
  }

  public int getSaveOutput() {
    return this.saveOutput;
  }

  void save(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  void load(InputStream is) throws IOException {
    dim = readInt(is);
    ws = readInt(is);
    epoch = readInt(is);
    minCount = readInt(is);
    neg = readInt(is);
    wordNgrams = readInt(is);
    loss = LossName.fromValue(readInt(is));
    model = ModelName.fromValue(readInt(is));
    bucket = readInt(is);
    minn = readInt(is);
    maxn = readInt(is);
    lrUpdateRate = readInt(is);
    t = readDouble(is);
  }

}
