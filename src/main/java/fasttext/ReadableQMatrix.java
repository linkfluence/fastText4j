package fasttext;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;

/** Interface for read-only quantized matrices */
public interface ReadableQMatrix extends Closeable, Cloneable {

  void addToVector(Vector x, int t);

  float dotRow(Vector vec, int i);

  int m();

  int n();

  ReadableQMatrix clone() throws CloneNotSupportedException;

  void close() throws IOException;

  void saveToMMap(OutputStream os) throws IOException;

}
