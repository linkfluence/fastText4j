package fasttext.store;

import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Abstract base class for performing read operations of low-level
 * data types.
 *
 * <p>{@code DataInput} may only be used from one thread, because it is not
 * thread safe (it keeps internal state like file position). To allow
 * multithreaded use, every {@code DataInput} instance must be cloned before
 * used in another thread. Subclasses must therefore implement {@link #clone()},
 * returning a new {@code DataInput} which operates on the same underlying
 * resource, but positioned independently.
 *
 * <p> From Lucene DataInput.
 */
public abstract class DataInput implements Cloneable {

  private static final int SKIP_BUFFER_SIZE = 1024;

  /* This buffer is used to skip over bytes with the default implementation of
   * skipBytes. The reason why we need to use an instance member instead of
   * sharing a single instance across threads is that some delegating
   * implementations of DataInput might want to reuse the provided buffer in
   * order to eg. update the checksum. If we shared the same buffer across
   * threads, then another thread might update the buffer while the checksum is
   * being computed, making it invalid.
   */
  private byte[] skipBuffer;

  /**
   * Reads and returns a single byte.
   * @see DataOutput#writeByte(byte)
   */
  public abstract byte readByte() throws IOException;

  /**
   * Reads a specified number of bytes into an array at the specified offset.
   * @param b the array to read bytes into
   * @param offset the offset in the array to start storing bytes
   * @param len the number of bytes to read
   * @see DataOutput#writeBytes(byte[],int)
   */
  public abstract void readBytes(byte[] b, int offset, int len)
    throws IOException;

  /**
   * Reads a specified number of bytes into an array at the
   * specified offset with control over whether the read
   * should be buffered (callers who have their own buffer
   * should pass in "false" for useBuffer).
   * @param b the array to read bytes into
   * @param offset the offset in the array to start storing bytes
   * @param len the number of bytes to read
   * @param useBuffer set to false if the caller will handle
   * buffering.
   * @see DataOutput#writeBytes(byte[],int)
   */
  public void readBytes(byte[] b, int offset, int len, boolean useBuffer)
    throws IOException
  {
    // Default to ignoring useBuffer entirely
    readBytes(b, offset, len);
  }

  /**
   * Reads two bytes and returns a short.
   * @see DataOutput#writeByte(byte)
   */
  public short readShort() throws IOException {
    return (short) (((readByte() & 0xFF) <<  8) |  (readByte() & 0xFF));
  }

  /**
   * Reads four bytes and returns an int.
   * @see DataOutput#writeInt(int)
   */
  public int readInt() throws IOException {
    return ((readByte() & 0xFF) << 24) | ((readByte() & 0xFF) << 16)
      | ((readByte() & 0xFF) <<  8) |  (readByte() & 0xFF);
  }

  /**
   * Reads eight bytes and returns a long.
   * @see DataOutput#writeLong(long)
   */
  public long readLong() throws IOException {
    return (((long)readInt()) << 32) | (readInt() & 0xFFFFFFFFL);
  }

  /**
   * Reads four bytes and returns a float.
   *  @see DataOutput#writeFloat(float)
   */
  public float readFloat() throws IOException {
    return Float.intBitsToFloat(readInt());
  }

  /**
   * Reads eight bytes and returns a double.
   *  @see DataOutput#writeDouble(double)
   */
  public double readDouble() throws IOException {
    return Double.longBitsToDouble(readLong());
  }

  /**
   * Reads a byte and returns a boolean.
   *  @see DataOutput#writeBoolean(boolean)
   */
  public boolean readBoolean() throws IOException {
    int ch = readByte();
    if (ch < 0)
      throw new EOFException();
    return (ch != 0);
  }

  /**
   * Reads a string.
   * @see DataOutput#writeString(String)
   */
  public String readString() throws IOException {
    int length = readInt();
    final byte[] bytes = new byte[length];
    readBytes(bytes, 0, length);
    return new String(bytes, 0, length, StandardCharsets.UTF_8);
  }

  /**
   * Reads a single byte to int
   * @see DataOutput#writeIntAsByte(int)
   */
  public int readByteAsInt() throws IOException {
    return readByte() & 0xFF;
  }

  /**
   * Returns a clone of this stream.
   *
   * <p>Clones of a stream access the same data, and are positioned at the same
   * point as the stream they were cloned from.
   *
   * <p>Expert: Subclasses must ensure that clones may be positioned at
   * different points in the input from each other and from the stream they
   * were cloned from.
   */
  @Override
  public DataInput clone() {
    try {
      return (DataInput) super.clone();
    } catch (CloneNotSupportedException e) {
      throw new Error("This cannot happen: Failing to clone DataInput");
    }
  }

  /**
   * Reads a Map&lt;String,String&gt; previously written
   * with {@link DataOutput#writeMapOfStrings(Map)}.
   * @return An immutable map containing the written contents.
   */
  public Map<String,String> readMapOfStrings() throws IOException {
    int count = readInt();
    if (count == 0) {
      return Collections.emptyMap();
    } else if (count == 1) {
      return Collections.singletonMap(readString(), readString());
    } else {
      Map<String,String> map = count > 10 ? new HashMap<>() : new TreeMap<>();
      for (int i = 0; i < count; i++) {
        final String key = readString();
        final String val = readString();
        map.put(key, val);
      }
      return Collections.unmodifiableMap(map);
    }
  }

  /**
   * Reads a Set&lt;String&gt; previously written
   * with {@link DataOutput#writeSetOfStrings(Set)}.
   * @return An immutable set containing the written contents.
   */
  public Set<String> readSetOfStrings() throws IOException {
    int count = readInt();
    if (count == 0) {
      return Collections.emptySet();
    } else if (count == 1) {
      return Collections.singleton(readString());
    } else {
      Set<String> set = count > 10 ? new HashSet<>() : new TreeSet<>();
      for (int i = 0; i < count; i++) {
        set.add(readString());
      }
      return Collections.unmodifiableSet(set);
    }
  }

  /**
   * Skip over <code>numBytes</code> bytes. The contract on this method is that it
   * should have the same behavior as reading the same number of bytes into a
   * buffer and discarding its content. Negative values of <code>numBytes</code>
   * are not supported.
   */
  public void skipBytes(final long numBytes) throws IOException {
    if (numBytes < 0) {
      throw new IllegalArgumentException("numBytes must be >= 0, got " + numBytes);
    }
    if (skipBuffer == null) {
      skipBuffer = new byte[SKIP_BUFFER_SIZE];
    }
    assert skipBuffer.length == SKIP_BUFFER_SIZE;
    for (long skipped = 0; skipped < numBytes; ) {
      final int step = (int) Math.min(SKIP_BUFFER_SIZE, numBytes - skipped);
      readBytes(skipBuffer, 0, step, false);
      skipped += step;
    }
  }

}
