package fasttext.store;

import java.io.BufferedInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;

/** Implementation class for buffered {@link FastTextInput} that writes to an {@link InputStream} */
public class InputStreamFastTextInput extends FastTextInput implements Closeable {

  private final BufferedInputStream is;

  public InputStreamFastTextInput(InputStream in, int bufferSize) {
    this.is = new BufferedInputStream(in, bufferSize);
  }

  public InputStreamFastTextInput(InputStream in) {
    this.is = new BufferedInputStream(in);
  }

  @Override
  public byte readByte() throws IOException {
    return (byte) is.read();
  }

  @Override
  public void readBytes(byte[] b, int offset, int len) throws IOException {
    is.read(b, offset, len);
  }

  @Override
  public void close() throws IOException {
    is.close();
  }

}
