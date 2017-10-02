package fasttext.store;

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;

/** Implementation class for buffered {@link FastTextOutput} that writes to an {@link OutputStream} */
public class OutputStreamFastTextOutput extends FastTextOutput implements Closeable {

  private final BufferedOutputStream os;

  public OutputStreamFastTextOutput(OutputStream out, int bufferSize) {
    this.os = new BufferedOutputStream(out, bufferSize);
  }

  public OutputStreamFastTextOutput(OutputStream out) {
    this.os = new BufferedOutputStream(out);
  }

  @Override
  public final void writeByte(byte b) throws IOException {
    os.write(b);
  }

  @Override
  public final void writeBytes(byte[] b, int offset, int length) throws IOException {
    os.write(b, offset, length);
  }

  @Override
  public void close() throws IOException {
    os.close();
  }


}
