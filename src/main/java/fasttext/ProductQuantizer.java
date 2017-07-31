package fasttext;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;

import static fasttext.util.io.IOUtils.readFloat;
import static fasttext.util.io.IOUtils.readInt;

public class ProductQuantizer {

  private static final int NUM_BITS = 8;
  private static final int KSUB = 1 << NUM_BITS;
  private static final int MAX_POINTS_PER_CLUSTER = 256;
  private static final int MAX_POINTS = MAX_POINTS_PER_CLUSTER * KSUB;
  private static final int SEED = 1234;
  private static final int NUM_ITER = 25;
  private static final double EPS = 1e-7;

  private int dim;
  private int nsubq;
  private int dsub;
  private int lastdsub;

  private float[] centroids;
  private Random rng;

  public float distL2(float[] x, float[] y, int d) {
    return distL2(x, y, d, 0, 0);
  }

  public float distL2(float[] x, float[] y, int d, int xpos, int ypos) {
    float dist = 0.0f;
    for (int i = 0; i < d; i++) {
      float tmp = x[i + xpos] - y[i + ypos];
      dist += tmp * tmp;
    }
    return dist;
  }

  public ProductQuantizer() {}

  public ProductQuantizer(int dim, int dsub) {
    this.dim = dim;
    this.nsubq = dim / dsub;
    this.dsub = dsub;
    this.centroids = new float[dim * KSUB];
    this.rng = new Random(SEED);
    this.lastdsub = dim % dsub;
    if (this.lastdsub == 0)
      this.lastdsub = dsub;
    else
      this.nsubq++;
  }

  public float getCentroids(int position) {
    return centroids[position];
  }

  public int getCentroidsPosition(int m, int i) {
    if (m == nsubq - 1) {
      return m * KSUB * dsub + i * lastdsub;
    } else {
      return (m * KSUB + i) * dsub;
    }
  }

  private float assignCentroid(float[] x, int xStartPosition, int c0Position, int[] code, int codeStartPosition, int d) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  private void eStep(float[] x, int cPosition, int[] codes, int d, int n) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  private void mStep(float[] x0, int cPosition, int[] codes, int d, int n) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  private void kmeans(float[] x, int cPosition, int n, int d) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void train(int n, float[] x) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void computeCode(float[] x, int[] codes, int xBeginPosition, int codeBeginPosition) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void computeCodes(float[] x, int[] codes, int m) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public float mulCode(Vector x, int[] codes, int t, float alpha) {
    float res = 0.0f;
    int d = dsub;
    int codePos = nsubq + t;
    for (int m = 0; m < nsubq; m++) {
      int c = getCentroidsPosition(m, codes[m + codePos]);
      if (m == nsubq - 1) {
        d = lastdsub;
      }
      for(int n = 0; n < d; n++) {
        res += x.data[m * dsub + n] * centroids[c * n];
      }
    }
    return res * alpha;
  }

  public void addCode(Vector x, int[] codes, int t, float alpha) {
    int d = dsub;
    int codePos = nsubq * t;
    for (int m = 0; m < nsubq; m++) {
      int c = getCentroidsPosition(m, codes[m + codePos]);
      if (m == nsubq - 1) {
        d = lastdsub;
      }
      for(int n = 0; n < d; n++) {
        x.data[m * dsub + n] += alpha * centroids[c + n];
      }
    }
  }

  public void save(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void load(InputStream is) throws IOException {
    dim = readInt(is);
    nsubq = readInt(is);
    dsub = readInt(is);
    lastdsub = readInt(is);
    centroids = new float[dim * KSUB];
    for (int i = 0; i < centroids.length; i++) {
      float c = readFloat(is);
      centroids[i] = c;
    }
  }

}
