package fasttext;

import com.google.common.base.Preconditions;

public class Vector {

  final int m;
  final float[] data;

  public Vector(int size) {
    this.m = size;
    this.data = new float[size];
  }

  public int size() {
    return this.m;
  }

  public void zero() {
    for (int i = 0; i < m; i++) {
      data[i] = 0.0f;
    }
  }

  public float norm() {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
      sum += data[i] * data[i];
    }
    return (float) Math.sqrt(sum);
  }

  public void mul(float a) {
    for (int i = 0; i < m; i++) {
      data[i] *= a;
    }
  }

  public void addVector(Vector source) {
    Preconditions.checkArgument(source.size() == m);
    for(int i = 0; i < m; i++) {
      data[i] += source.at(i);
    }
  }

  public void addVector(Vector source, float s) {
    Preconditions.checkArgument(source.size() == m);
    for(int i = 0; i < m; i++) {
      data[i] += s * source.at(i);
    }
  }

  public void addRow(ReadableMatrix A, int i, float a) {
    Preconditions.checkPositionIndex(i, A.m());
    Preconditions.checkArgument(m == A.n());
    for (int j = 0; j < A.n(); j++) {
      data[j] += a * A.at(i, j);
    }
  }

  public void addRow(ReadableMatrix A, int i) {
    Preconditions.checkPositionIndex(i, A.m());
    Preconditions.checkArgument(m == A.n());
    for (int j = 0; j < A.n(); j++) {
      data[j] += A.at(i, j);
    }
  }

  public void addRow(ReadableQMatrix A, int i) {
    Preconditions.checkArgument(i >= 0);
    A.addToVector(this, i);
  }

  public void mul(ReadableMatrix A, Vector vec) {
    Preconditions.checkArgument(m == A.m());
    Preconditions.checkArgument(A.n() == vec.size());
    for (int i = 0; i < m; i++) {
      data[i] = A.dotRow(vec, i);
    }
  }

  public void mul(ReadableQMatrix A, Vector vec) {
    Preconditions.checkArgument(m == A.m());
    Preconditions.checkArgument(A.n() == vec.size());
    for (int i = 0; i < m; i++) {
      data[i] = A.dotRow(vec, i);
    }
  }

  public int argmax() {
    float max = data[0];
    int argmax = 0;
    for (int i = 1; i < m; i++) {
      if (data[i] > max) {
        max = data[i];
        argmax = i;
      }
    }
    return argmax;
  }

  public void addAt(int i, float v) {
    Preconditions.checkPositionIndex(i, m);
    data[i] += v;
  }

  public void set(int i, float v) {
    Preconditions.checkPositionIndex(i, m);
    data[i] = v;
  }

  public float at(int i) {
    Preconditions.checkPositionIndex(i, m);
    return data[i];
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Vector(");
    builder.append("m=");
    builder.append(m);
    builder.append(", data=");
    if (data != null) {
      builder.append("[");
      for (float d : data) {
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

  public float[] toArray() {
    return this.data;
  }

}
