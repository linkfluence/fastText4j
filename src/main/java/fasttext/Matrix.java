package fasttext;

import com.google.common.base.Preconditions;
import fasttext.util.Utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;

import static fasttext.util.io.IOUtils.*;

public class Matrix {

  float[] data;
  int m = 0;
  int n = 0;

  public Matrix() {}

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
      data[i] = Utils.randomFloat(rng, -a, a);
    }
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

  public float[] data() {
    return this.data;
  }

  void save(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  void load(InputStream is) throws IOException {
    m = (int) readLong(is);
    n = (int) readLong(is);
    data = new float[m * n];
    for (int i = 0; i < m * n; i++) {
      float d = readFloat(is);
      data[i] = d;
    }
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

}
