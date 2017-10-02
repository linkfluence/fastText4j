package fasttext.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

/** Extension of {@link DataOutput} following fastText models output writes. */
public abstract class FastTextOutput extends DataOutput {

  /**
   * Writes an int as four bytes.
   * <p>
   * 32-bit unsigned integer written as four bytes, low-order bytes first.
   *
   * @see FastTextInput#readInt()
   */
  public void writeInt(int i) throws IOException {
    writeByte((byte) i);
    writeByte((byte)(i >>  8));
    writeByte((byte)(i >> 16));
    writeByte((byte)(i >> 24));
  }

  /**
   * Writes a short as two bytes.
   * @see FastTextInput#readShort()
   */
  public void writeShort(short i) throws IOException {
    writeByte((byte) i);
    writeByte((byte)(i >>  8));
  }

  /**
   * Writes a long as eight bytes.
   * <p>
   * 64-bit unsigned integer written as eight bytes, low-order bytes first.
   *
   * @see FastTextInput#readLong()
   */
  public void writeLong(long i) throws IOException {
    writeInt((int) i);
    writeInt((int) (i >> 32));
  }

  /**
   * Writes a string followed by 0 to mark end of String.
   *
   * @see FastTextInput#readString()
   */
  public void writeString(String s) throws IOException {
    byte[] barr = s.getBytes(StandardCharsets.UTF_8);
    writeBytes(barr, barr.length);
    writeByte((byte) 0);
  }


}
