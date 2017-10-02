package fasttext.mmap;

import com.google.common.base.Preconditions;
import fasttext.ReadableMatrix;
import fasttext.Vector;
import fasttext.store.MMapFile;
import fasttext.store.ResourceInput;

import java.io.IOException;
import java.io.OutputStream;

/** Memory-mapped {@link ReadableMatrix}. Only supports read-only operations. */
public class MMapMatrix implements ReadableMatrix {

  private final int m;
  private final int n;
  private final MMapFile mmapFile;
  private ResourceInput in;

  private MMapMatrix(MMapFile mmapFile, ResourceInput in, int m, int n) {
    this.mmapFile = mmapFile;
    this.in = in;
    this.m = m;
    this.n = n;
  }

  private float readAt(int i, int j) {
    try {
      in.seek(16 + i * n + j);
      return in.readFloat();
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read float from matrix at i=" + i + " j=" + j);
    }
  }

  private float[] readRow(int i) {
    float[] r = new float[n];
    try {
      in.seek(16 + i * n);
      for (int j = 0; j < n; j++) {
        r[j] = in.readFloat();
      }
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read row " + i + " from matrix");
    }
    return r;
  }

  public float[] atRow(int i) {
    return readRow(i);
  }

  public float at(int i, int j) {
    return readAt(i, j);
  }

  public float dotRow(final Vector vec, int i) {
    Preconditions.checkPositionIndex(i, m);
    Preconditions.checkArgument(vec.size() == n);
    float d = 0.0f;
    float[] r = atRow(i);
    for (int j = 0; j < n; j++) {
      d += r[j] * vec.at(j);
    }
    return d;
  }

  public float l2NormRow(int i) {
    float norm = 0.0f;
    float[] r = atRow(i);
    for (int j = 0; j < n; j++) {
      float v = r[j];
      norm += v * v;
    }
    return (float) Math.sqrt(norm);
  }

  public Vector l2NormRow(Vector norms) {
    Preconditions.checkArgument(norms.size() == m);
    for (int i = 0; i < m; i++) {
      norms.set(i, l2NormRow(i));
    }
    return norms;
  }

  public int m() { return m; }

  public int n() { return n; }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Matrix(m=");
    builder.append(m);
    builder.append(", n=");
    builder.append(n);
    builder.append(", data=");
    builder.append(", mmap=MMapFile(");
    builder.append(mmapFile.getPath().toString());
    builder.append("))");
    return builder.toString();
  }

  public static MMapMatrix load(MMapFile mmap) throws IOException {
    ResourceInput in = mmap.openInput();
    int m = (int) in.readLong();
    int n = (int) in.readLong();
    return new MMapMatrix(mmap, in, m, n);
  }

  @Override
  public MMapMatrix clone() throws CloneNotSupportedException {
    MMapMatrix m = (MMapMatrix) super.clone();
    m.in = in.clone();
    return m;
  }

  public void close() throws IOException {
    in.close();
  }

  public void saveToMMap(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

}
