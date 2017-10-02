package fasttext.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

/** Extension of {@link DataInput} following fastText models input reads. */
public abstract class FastTextInput extends DataInput {

  /**
   * Reads two bytes and returns a short.
   * @see FastTextOutput#writeByte(byte)
   */
  @Override
  public short readShort() throws IOException {
    return (short) ((readByte() & 0xFF) | ((readByte() & 0xFF) <<  8));
  }

  /**
   * Reads four bytes written with low-order first and returns an int.
   * @see FastTextOutput#writeInt(int)
   */
  @Override
  public int readInt() throws IOException {
    return (readByte() & 0xFF) | ((readByte() & 0xFF) <<  8)
      | ((readByte() & 0xFF) << 16) | ((readByte() & 0xFF) << 24);
  }

  /**
   * Reads eight bytes written with low-order first and returns a long.
   * @see FastTextOutput#writeLong(long)
   */
  @Override
  public long readLong() throws IOException {
    return (readByte() & 0xFFL) | (readByte() & 0xFFL) << 8
      | (readByte() & 0xFFL) << 16 | (readByte() & 0xFFL) << 24
      | (readByte() & 0xFFL) << 32 | (readByte() & 0xFFL) << 40
      | (readByte() & 0xFFL) << 48 | (readByte() & 0xFFL) << 56;
  }

  /**
   * Reads a string.
   * @see FastTextOutput#writeString(String)
   */
  @Override
  public String readString() throws IOException {
    int b = readByteAsInt();
    if (b < 0) {
      return null;
    }
    int i = -1;
    StringBuilder sb = new StringBuilder();
    int barrSize = 128;
    byte[] barr = new byte[barrSize];
    // ascii space, \n, \0
    while (b > -1 && b != 32 && b != 10 && b != 0) {
      barr[++i] = (byte) b;
      b = readByteAsInt();
      if (i == barrSize - 1) {
        sb.append(new String(barr));
        i = -1;
      }
    }
    sb.append(new String(barr, 0, i + 1, StandardCharsets.UTF_8));
    return sb.toString();
  }

}
