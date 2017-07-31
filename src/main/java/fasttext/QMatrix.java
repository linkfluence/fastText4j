package fasttext;

import com.google.common.base.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static fasttext.util.io.IOUtils.*;

public class QMatrix {

  private ProductQuantizer npq;
  private ProductQuantizer pq;
  private int codeSize = 0;
  private int[] codes;
  private int[] normCodes;
  private boolean qnorm = false;
  int m = 0;
  int n = 0;

  public QMatrix() {}

  public QMatrix(Matrix mat, int dsub, boolean qnorm) {
    this.qnorm = qnorm;
    this.m = mat.m;
    this.n = mat.n;
    codeSize = (this.m * (int) Math.ceil(this.n / dsub));
    this.codes = new int[codeSize];
    this.pq = new ProductQuantizer(n, dsub);
    if (this.qnorm) {
      this.normCodes = new int[this.m];
      this.npq = new ProductQuantizer(1, 1);
    }
    quantize(mat);
  }

  public void quantizeNorm(Vector norms) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void quantize(Matrix matrix) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void addToVector(Vector x, int t) {
    float norm = 1f;
    if (qnorm) {
      int cPosition = npq.getCentroidsPosition(0, normCodes[t]);
      norm = npq.getCentroids(cPosition);
    }
    pq.addCode(x, codes, t, norm);
  }

  public float dotRow(Vector vec, int i) {
    Preconditions.checkPositionIndex(i, m);
    Preconditions.checkArgument(vec.size() == n);
    float norm = 1f;
    if (qnorm) {
      int cPosition = npq.getCentroidsPosition(0, normCodes[i]);
      norm = npq.getCentroids(cPosition);
    }
    return pq.mulCode(vec, codes, i, norm);
  }

  public int m() {
    return m;
  }

  public int n() {
    return n;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Matrix(m=");
    builder.append(m);
    builder.append(", n=");
    builder.append(n);
    builder.append(", codeSize=");
    builder.append(codeSize);
    builder.append(", codes=");
    if (codes != null) {
      builder.append("[");
      for (float d : codes) {
        builder.append(d).append(' ');
      }
      if (builder.length() > 1) {
        builder.setLength(builder.length() - 1);
      }
      builder.append("]");
    } else {
      builder.append("null");
    }
    builder.append(", qnorm=");
    builder.append(qnorm);
    builder.append(", normCodes=");
    if (normCodes != null) {
      builder.append("[");
      for (float d : normCodes) {
        builder.append(d).append(' ');
      }
      if (builder.length() > 1) {
        builder.setLength(builder.length() - 1);
      }
      builder.append("]");
    } else {
      builder.append("null");
    }
    builder.append(")");
    return builder.toString();
  }


  public void save(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void load(InputStream is) throws IOException {
    qnorm = readBoolean(is);
    m = (int) readLong(is);
    n = (int) readLong(is);
    codeSize = readInt(is);
    codes = new int[codeSize];
    for (int i = 0; i < codeSize; i++) {
      int c = readByte(is);
      codes[i] = c;
    }
    pq = new ProductQuantizer();
    pq.load(is);
    if (qnorm) {
      normCodes = new int[m];
      for (int i = 0; i < m; i++) {
        int c = readByte(is);
        normCodes[i] = c;
      }
      npq = new ProductQuantizer();
      npq.load(is);
    }
  }

}

