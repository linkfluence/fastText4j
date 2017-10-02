package fasttext.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;

/**
 * Abstract base class for performing write operations of low-level
 * data types.
 *
 * <p>{@code DataOutput} may only be used from one thread, because it is not
 * thread safe (it keeps internal state like file position).
 *
 * <p>From Lucene DataOutput.
 */
public abstract class DataOutput {

  /**
   * Writes a single byte.
   * <p>
   * The most primitive data type is an eight-bit byte. Files are
   * accessed as sequences of bytes. All other data types are defined
   * as sequences of bytes, so file formats are byte-order independent.
   *
   * @see DataInput#readByte()
   */
  public abstract void writeByte(byte b) throws IOException;

  /**
   * Writes an array of bytes.
   * @param b the bytes to write
   * @param length the number of bytes to write
   * @see DataInput#readBytes(byte[],int,int)
   */
  public void writeBytes(byte[] b, int length) throws IOException {
    writeBytes(b, 0, length);
  }

  /**
   * Writes an array of bytes.
   * @param b the bytes to write
   * @param offset the offset in the byte array
   * @param length the number of bytes to write
   * @see DataInput#readBytes(byte[],int,int)
   */
  public abstract void writeBytes(byte[] b, int offset, int length) throws IOException;

  /**
   * Writes an int as four bytes.
   * <p>
   * 32-bit unsigned integer written as four bytes, high-order bytes first.
   *
   * @see DataInput#readInt()
   */
  public void writeInt(int i) throws IOException {
    writeByte((byte)(i >> 24));
    writeByte((byte)(i >> 16));
    writeByte((byte)(i >>  8));
    writeByte((byte) i);
  }

  /**
   * Writes a short as two bytes.
   * @see DataInput#readShort()
   */
  public void writeShort(short i) throws IOException {
    writeByte((byte)(i >>  8));
    writeByte((byte) i);
  }

  /**
   * Writes a long as eight bytes.
   * <p>
   * 64-bit unsigned integer written as eight bytes, high-order bytes first.
   *
   * @see DataInput#readLong()
   */
  public void writeLong(long i) throws IOException {
    writeInt((int) (i >> 32));
    writeInt((int) i);
  }

  /**
   * Writes a float as four bytes
   * <p>
   * 32-bit float written as four bytes, high-order bytes first.
   *
   * @see DataInput#readFloat()
   */
  public void writeFloat(float f) throws IOException {
    writeInt(Float.floatToIntBits(f));
  }

  /**
   * Writes a double as four bytes
   * <p>
   * 64-bit double written as four bytes, high-order bytes first.
   *
   * @see DataInput#readDouble()
   */
  public void writeDouble(double d) throws IOException {
    writeLong(Double.doubleToRawLongBits(d));
  }

  /**
   * Writes a boolean as a single byte
   * <p>
   * boolean written as a single byte.
   *
   * @see DataInput#readBoolean()
   */
  public void writeBoolean(boolean b) throws IOException {
    if (b) {
      writeByte((byte) (1 & 0xFF));
    } else {
      writeByte((byte) 0);
    }
  }

  /**
   * Writes a string.
   * <p>
   * Writes strings as UTF-8 encoded bytes. First the length, in bytes, followed by the bytes.
   *
   * @see DataInput#readString()
   */
  public void writeString(String s) throws IOException {
    byte[] barr = s.getBytes(StandardCharsets.UTF_8);
    writeInt(barr.length);
    writeBytes(barr, barr.length);
  }

  /**
   * Writes an int as single byte
   *
   * @see DataInput#readByteAsInt()
   */
  public void writeIntAsByte(int i) throws IOException {
    writeByte((byte) (i & 0xFF));
  }

  private static int COPY_BUFFER_SIZE = 16384;
  private byte[] copyBuffer;

  /** Copy numBytes bytes from input to ourself. */
  public void copyBytes(DataInput input, long numBytes) throws IOException {
    assert numBytes >= 0: "numBytes=" + numBytes;
    long left = numBytes;
    if (copyBuffer == null)
      copyBuffer = new byte[COPY_BUFFER_SIZE];
    while(left > 0) {
      final int toCopy;
      if (left > COPY_BUFFER_SIZE)
        toCopy = COPY_BUFFER_SIZE;
      else
        toCopy = (int) left;
      input.readBytes(copyBuffer, 0, toCopy);
      writeBytes(copyBuffer, 0, toCopy);
      left -= toCopy;
    }
  }

  /**
   * Writes a String map.
   * <p>
   * First the size is written as an {@link #writeInt(int) Int},
   * followed by each key-value pair written as two consecutive
   * {@link #writeString(String) String}s.
   *
   * @param map Input map.
   * @throws NullPointerException if {@code map} is null.
   */
  public void writeMapOfStrings(Map<String,String> map) throws IOException {
    writeInt(map.size());
    for (Map.Entry<String, String> entry : map.entrySet()) {
      writeString(entry.getKey());
      writeString(entry.getValue());
    }
  }

  /**
   * Writes a String set.
   * <p>
   * First the size is written as an {@link #writeInt(int) Int},
   * followed by each value written as a
   * {@link #writeString(String) String}.
   *
   * @param set Input set.
   * @throws NullPointerException if {@code set} is null.
   */
  public void writeSetOfStrings(Set<String> set) throws IOException {
    writeInt(set.size());
    for (String value : set) {
      writeString(value);
    }
  }

}
