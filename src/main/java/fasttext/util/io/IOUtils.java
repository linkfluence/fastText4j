package fasttext.util.io;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Utils function to read raw types from binary model
 */
public class IOUtils {

  public static int readInt(InputStream is) throws IOException {
    byte[] b = new byte[Integer.BYTES];
    is.read(b);
    return getInt(b);
  }

  public static int getInt(byte[] b) {
    return b[0] & 0xFF | (b[1] & 0xFF) << 8 | (b[2] & 0xFF) << 16 | (b[3] & 0xFF) << 24;
  }

  public static long readLong(InputStream is) throws IOException {
    byte[] b = new byte[Long.BYTES];
    is.read(b);
    return getLong(b);
  }

  public static long getLong(byte[] b) {
    return b[0] & 0xFFL | (b[1] & 0xFFL) << 8 | (b[2] & 0xFFL) << 16 | (b[3] & 0xFFL) << 24
      | (b[4] & 0xFFL) << 32 | (b[5] & 0xFFL) << 40 | (b[6] & 0xFFL) << 48 | (b[7] & 0xFFL) << 56;
  }

  public static float readFloat(InputStream is) throws IOException {
    byte[] b = new byte[Float.BYTES];
    is.read(b);
    return getFloat(b);
  }

  public static float getFloat(byte[] b) {
    return Float.intBitsToFloat(b[0] & 0xFF | (b[1] & 0xFF) << 8 | (b[2] & 0xFF) << 16 | (b[3] & 0xFF) << 24);
  }

  public static double readDouble(InputStream is) throws IOException {
    byte[] b = new byte[Double.BYTES];
    is.read(b);
    return getDouble(b);
  }

  public static double getDouble(byte[] b) {
    return Double.longBitsToDouble(getLong(b));
  }

  public static int readByte(InputStream is) throws IOException {
    return is.read() & 0xFF;
  }

  public static String readString(InputStream is, String charsetName) throws IOException {
    int b = is.read();
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
      b = is.read();
      if (i == barrSize - 1) {
        sb.append(new String(barr));
        i = -1;
      }
    }
    sb.append(new String(barr, 0, i + 1, charsetName));
    return sb.toString();
  }

  public static boolean readBoolean(InputStream is) throws IOException {
    int ch = is.read();
    if (ch < 0)
      throw new EOFException();
    return (ch != 0);
  }

}
