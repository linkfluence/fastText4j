package fasttext;

import com.google.common.base.Preconditions;
import fasttext.store.InputStreamFastTextInput;
import fasttext.store.OutputStreamFastTextOutput;
import fasttext.store.OutputStreamResourceOutput;
import fasttext.util.Randoms;

import java.io.*;
import java.util.Random;

public class Matrix implements ReadableMatrix {

  private final float[] data;
  private final int m;
  private final int n;

  private Matrix(int m, int n, float[] data) {
    this.m = m;
    this.n = n;
    this.data = data;
  }

  public Matrix(int m, int n) {
    this.m = m;
    this.n = n;
    data = new float[this.m * this.n];
  }

  public Matrix(Matrix other) {
    this.m = other.m;
    this.n = other.n;
    data = new float[this.m * this.n];
    for (int i = 0; i < (this.m * this.n); i++) {
      data[i] = other.data[i];
    }
  }

  public void zero() {
    for (int i = 0; i < (m * n); i++) {
      data[i] = 0.0f;
    }
  }

  public void uniform(float a) {
    Random rng = new Random(1L);
    for (int i = 0; i < (m * n); i++) {
      data[i] = Randoms.randomFloat(rng, -a, a);
    }
  }

  public float[] atRow(int i) {
    float[] r = new float[n];
    for (int j = 0; j < n; j++) {
      r[j] = data[i * n];
    }
    return r;
  }

  public float at(int i, int j) {
    return data[i * n + j];
  }

  public float dotRow(final Vector vec, int i) {
    Preconditions.checkPositionIndex(i, m);
    Preconditions.checkArgument(vec.size() == n);
    float d = 0.0f;
    for (int j = 0; j < n; j++) {
      d += data[i * n + j] * vec.at(j);
    }
    return d;
  }

  public void addRow(final Vector vec, int i, float a) {
    Preconditions.checkPositionIndex(i, m);
    Preconditions.checkArgument(vec.size() == n);
    for (int j = 0; j < n; j++) {
      data[i * n + j] += a * vec.at(j);
    }
  }

  public void multiplyRow(final Vector nums) {
    multiplyRow(nums, 0, -1);
  }

  public void multiplyRow(final Vector nums, int ib, int ie) {
    if (ie == -1) {
      ie = m;
    }
    Preconditions.checkPositionIndex(ie, nums.size());
    for (int i = ib; i < ie; i++) {
      float num = nums.at(i - ib);
      if (n != 0) {
        for (int j = 0; j < this.n; j++) {
          data[i * n + j] *= num;
        }
      }
    }
  }

  public void divideRow(final Vector denoms) {
    divideRow(denoms, 0, -1);
  }

  public void divideRow(final Vector denoms, int ib, int ie) {
    if (ie == -1) {
      ie = m;
    }
    Preconditions.checkPositionIndex(ie, denoms.size());
    for (int i = ib; i < ie; i++) {
      float denom = denoms.at(i - ib);
      if (denom != 0) {
        for (int j = 0; j < this.n; j++) {
          data[i * n + j] /= denom;
        }
      }
    }
  }

  public float l2NormRow(int i) {
    float norm = 0.0f;
    for (int j = 0; j < n; j++) {
      float v = data[i * n + j];
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

  public int m() {
    return this.m;
  }

  public int n() {
    return this.n;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Matrix(m=");
    builder.append(m);
    builder.append(", n=");
    builder.append(n);
    builder.append(", data=");
    if (data != null) {
      builder.append("[");
      for (int i = 0; i < m && i < 10; i++) {
        for (int j = 0; j < n && j < 10; j++) {
          builder.append(data[i * n + j]).append(",");
        }
      }
      builder.setLength(builder.length() - 1);
      builder.append("]");
    } else {
      builder.append("null");
    }
    builder.append(")");
    return builder.toString();
  }

  public float[] toArray() {
    return this.data;
  }

  public static Matrix load(InputStreamFastTextInput is) throws IOException {
    int m = (int) is.readLong();
    int n = (int) is.readLong();
    float[] data = new float[m * n];
    for (int i = 0; i < m * n; i++) {
      float d = is.readFloat();
      data[i] = d;
    }
    return new Matrix(m, n, data);
  }

  public void save(OutputStreamFastTextOutput os) throws IOException {
    os.writeLong(m);
    os.writeLong(n);
    for (int i = 0; i < m * n; i++) {
      os.writeFloat(data[i]);
    }
  }

  public void saveToMMap(OutputStream os) throws IOException {
    int bufferSize = 16 + 4 * m * n;
    try (OutputStreamResourceOutput fos = new OutputStreamResourceOutput("matrix", os, bufferSize)) {
      fos.writeLong(m);
      fos.writeLong(n);
      for (int i = 0; i < m * n; i++) {
        fos.writeFloat(data[i]);
      }
    }
  }

  public void close() {}

  public Matrix clone() throws CloneNotSupportedException {
    return (Matrix) super.clone();
  }

}
