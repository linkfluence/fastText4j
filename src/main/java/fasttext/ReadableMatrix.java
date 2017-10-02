package fasttext;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;

/** Interface for read-only matrices */
public interface ReadableMatrix extends Closeable, Cloneable {

  float[] atRow(int i);

  float at(int i, int j);

  float dotRow(final Vector vec, int i);

  float l2NormRow(int i);

  Vector l2NormRow(Vector norms);

  int m();

  int n();

  ReadableMatrix clone() throws CloneNotSupportedException;

  void close() throws IOException;

  void saveToMMap(OutputStream os) throws IOException;

}
