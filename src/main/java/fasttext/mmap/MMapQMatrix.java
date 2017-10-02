package fasttext.mmap;

import com.google.common.base.Preconditions;
import fasttext.ProductQuantizer;
import fasttext.QCodes;
import fasttext.ReadableQMatrix;
import fasttext.Vector;
import fasttext.store.MMapFile;
import fasttext.store.ResourceInput;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;

/** Memory-mapped {@link ReadableQMatrix}. Only supports read-only operations */
public class MMapQMatrix implements ReadableQMatrix {

  /** QCodes reading from ResourceInput */
  public static class MMapQCodes implements QCodes, Cloneable, Closeable {

    private ResourceInput in;

    private final long offset;
    private final int codeSize;

    private MMapQCodes(ResourceInput in, long offset, int codeSize) {
      this.in = in;
      this.offset = offset;
      this.codeSize = codeSize;
    }

    public int get(int i) {
      Preconditions.checkPositionIndex(i, codeSize);
      try {
        in.seek(offset + i);
        return in.readByteAsInt();
      } catch (IOException ex) {
        throw new IllegalArgumentException("Could not get code for i = " + i);
      }
    }

    public int size() {
      return codeSize;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("MMapQCodes(size=");
      builder.append(size());
      builder.append(", offset=");
      builder.append(offset);
      builder.append(")");
      return builder.toString();
    }

    @Override
    public MMapQCodes clone() throws CloneNotSupportedException {
      MMapQCodes q = (MMapQCodes) super.clone();
      q.in = in.clone();
      return q;
    }

    public void close() throws IOException {
      in.close();
    }

  }

  private final ProductQuantizer npq;
  private final ProductQuantizer pq;
  private boolean qnorm;
  private final int m;
  private final int n;

  private final MMapFile mmapFile;

  private MMapQCodes codes;
  private MMapQCodes normCodes;

  private MMapQMatrix(MMapFile mmapFile,
                      boolean qnorm,
                      int m,
                      int n,
                      MMapQCodes codes,
                      MMapQCodes normCodes,
                      ProductQuantizer pq,
                      ProductQuantizer npq) {
    this.mmapFile = mmapFile;
    this.qnorm = qnorm;
    this.m = m;
    this.n = n;
    this.codes = codes;
    this.normCodes = normCodes;
    this.pq = pq;
    this.npq = npq;
  }

  private int codesByteArrayLength() {
    return codes.size() * Byte.BYTES;
  }

  private int normCodesByteArrayLength() {
    return m * Byte.BYTES;
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
    builder.append(", MMapFile=(");
    builder.append(mmapFile.getPath().toString());
    builder.append("))");
    return builder.toString();
  }

  public static MMapQMatrix load(MMapFile mmap) throws IOException {
    ResourceInput in = mmap.openInput();
    boolean qnorm = in.readBoolean();
    int m = (int) in.readLong();
    int n = (int) in.readLong();
    int codeSize = in.readInt();
    MMapQCodes codes = new MMapQCodes(in, 21, codeSize);
    in.skipBytes(codeSize);
    // pq
    int dim = in.readInt();
    int nsubq = in.readInt();
    int dsub = in.readInt();
    int lastdsub = in.readInt();
    float[] centroids = new float[ProductQuantizer.findCentroidsSize(dim)];
    for (int i = 0; i < centroids.length; i++) {
      centroids[i] = in.readFloat();
    }
    ProductQuantizer pq = new ProductQuantizer(dim, nsubq, dsub, lastdsub, centroids);
    MMapQCodes normCodes = null;
    ProductQuantizer npq = null;
    if (qnorm) {
      normCodes = new MMapQCodes(in,37 + codeSize + centroids.length * 4, m);
      in.skipBytes(m);
      // npq
      int normDim = in.readInt();
      int normNsubq = in.readInt();
      int normDsub = in.readInt();
      int normLastdsub = in.readInt();
      float[] normCentroids = new float[ProductQuantizer.findCentroidsSize(normDim)];
      for (int i = 0; i < normCentroids.length; i++) {
        normCentroids[i] = in.readFloat();
      }
      npq = new ProductQuantizer(normDim, normNsubq, normDsub, normLastdsub, normCentroids);
    }
    return new MMapQMatrix(mmap, qnorm, m, n, codes, normCodes, pq, npq);
  }

  @Override
  public MMapQMatrix clone() throws CloneNotSupportedException {
    MMapQMatrix m = (MMapQMatrix) super.clone();
    m.codes = codes.clone();
    if (qnorm) {
      m.normCodes = normCodes.clone();
    }
    return m;
  }

  public void close() throws IOException {
    codes.close();
    if (qnorm) {
      normCodes.close();
    }
  }

  public void saveToMMap(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

}
