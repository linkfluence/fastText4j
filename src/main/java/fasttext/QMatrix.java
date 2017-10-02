package fasttext;

import com.google.common.base.Preconditions;
import fasttext.store.InputStreamFastTextInput;
import fasttext.store.OutputStreamFastTextOutput;
import fasttext.store.OutputStreamResourceOutput;

import java.io.IOException;
import java.io.OutputStream;

public class QMatrix implements ReadableQMatrix {

  private final ProductQuantizer npq;
  private final ProductQuantizer pq;
  private final QCodeArray codes;
  private final QCodeArray normCodes;
  private final boolean qnorm;
  private final int m;
  private final int n;

  private QMatrix(boolean qnorm,
                  int m,
                  int n,
                  QCodeArray codes,
                  ProductQuantizer pq,
                  QCodeArray normCodes,
                  ProductQuantizer npq) {
    this.qnorm = qnorm;
    this.m = m;
    this.n = n;
    this.codes = codes;
    this.pq = pq;
    this.normCodes = normCodes;
    this.npq = npq;
  }

  public QMatrix(Matrix mat, int dsub, boolean qnorm) {
    this.qnorm = qnorm;
    this.m = mat.m();
    this.n = mat.n();
    int codeSize = (this.m * (int) Math.ceil(this.n / dsub));
    this.codes = new QCodeArray(codeSize);
    this.pq = new ProductQuantizer(n, dsub);
    if (this.qnorm) {
      this.normCodes = new QCodeArray(this.m);
      this.npq = new ProductQuantizer(1, 1);
    } else {
      this.normCodes = null;
      this.npq = null;
    }
    quantize(mat);
  }

  public QMatrix(QMatrix mat) {
    this.qnorm = mat.qnorm;
    this.m = mat.m;
    this.n = mat.n;
    this.codes = mat.codes;
    this.pq = mat.pq;
    if (mat.qnorm) {
      this.normCodes = mat.normCodes;
      this.npq = mat.npq;
    } else {
      this.normCodes = null;
      this.npq = null;
    }
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
      int cPosition = npq.getCentroidsPosition(0, normCodes.get(t));
      norm = npq.getCentroid(cPosition);
    }
    pq.addCode(x, codes, t, norm);
  }

  public float dotRow(Vector vec, int i) {
    Preconditions.checkPositionIndex(i, m);
    Preconditions.checkArgument(vec.size() == n);
    float norm = 1f;
    if (qnorm) {
      int cPosition = npq.getCentroidsPosition(0, normCodes.get(i));
      norm = npq.getCentroid(cPosition);
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
    builder.append(codes.size());
    builder.append(", codes=");
    builder.append(codes.toString());
    builder.append(", qnorm=");
    builder.append(qnorm);
    builder.append(", normCodes=");
    if (normCodes != null) {
      builder.append(normCodes.toString());
    } else {
      builder.append("null");
    }
    builder.append(")");
    return builder.toString();
  }

  public static QMatrix load(InputStreamFastTextInput is) throws IOException {
    boolean qnorm = is.readBoolean();
    int m = (int) is.readLong();
    int n = (int) is.readLong();
    int codeSize = is.readInt();
    int[] rawCodes = new int[codeSize];
    for (int i = 0; i < codeSize; i++) {
      int c = is.readByteAsInt();
      rawCodes[i] = c;
    }
    QCodeArray codes = new QCodeArray(rawCodes);
    ProductQuantizer pq = ProductQuantizer.load(is);
    QCodeArray normCodes = null;
    ProductQuantizer npq = null;
    if (qnorm) {
      int[] rawNormCodes = new int[m];
      for (int i = 0; i < m; i++) {
        int c = is.readByteAsInt();
        rawNormCodes[i] = c;
      }
      normCodes = new QCodeArray(rawNormCodes);
      npq = ProductQuantizer.load(is);
    }
    return new QMatrix(qnorm, m, n, codes, pq, normCodes, npq);
  }

  public void save(OutputStreamFastTextOutput os) throws IOException {
    os.writeBoolean(qnorm);
    os.writeLong(m);
    os.writeLong(n);
    os.writeInt(codes.size());
    for (int i = 0; i < codes.size(); i++) {
      os.writeIntAsByte(codes.get(i));
    }
    pq.save(os);
    if (qnorm) {
      for (int i = 0; i < m; i++) {
        os.writeIntAsByte(normCodes.get(i));
      }
      npq.save(os);
    }
  }

  public void saveToMMap(OutputStream os) throws IOException {
    int bufferSize = 37 + codes.size() + pq.centroids().length * 4;
    if (qnorm) {
      bufferSize += m + 16 + npq.centroids().length * 4;
    }
    try (OutputStreamResourceOutput fos = new OutputStreamResourceOutput("qmatrix", os, bufferSize)) {
      fos.writeBoolean(qnorm);
      fos.writeLong(m);
      fos.writeLong(n);
      fos.writeInt(codes.size());
      for (int i = 0; i < codes.size(); i++) {
        fos.writeIntAsByte(codes.get(i));
      }
      // pq
      fos.writeInt(pq.dim());
      fos.writeInt(pq.nsubq());
      fos.writeInt(pq.dsub());
      fos.writeInt(pq.lastdsub());
      for (int i = 0; i < pq.centroids().length; i++) {
        fos.writeFloat(pq.getCentroid(i));
      }
      if (qnorm) {
        for (int i = 0; i < m; i++) {
          fos.writeIntAsByte(normCodes.get(i));
        }
        // npq
        fos.writeInt(npq.dim());
        fos.writeInt(npq.nsubq());
        fos.writeInt(npq.dsub());
        fos.writeInt(npq.lastdsub());
        for (int i = 0; i < npq.centroids().length; i++) {
          fos.writeFloat(npq.getCentroid(i));
        }
      }
    }
  }

  public void close() {}

  @Override
  public QMatrix clone() throws CloneNotSupportedException {
    return (QMatrix) super.clone();
  }

}

