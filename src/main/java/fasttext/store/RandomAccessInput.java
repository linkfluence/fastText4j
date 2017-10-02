package fasttext.store;

import java.io.IOException;

/**
 * Random Access Resource API.
 * Unlike {@link ResourceInput}, this has no concept of file position, all reads
 * are absolute. However, like ResourceInput, it is only intended for use by a single thread.
 *
 * <p>From Lucene RandomAccessInput.
 */
public interface RandomAccessInput {

  /**
   * Reads a byte at the given position in the file
   * @see DataInput#readByte
   */
  byte readByte(long pos) throws IOException;
  /**
   * Reads a short at the given position in the file
   * @see DataInput#readShort
   */
  short readShort(long pos) throws IOException;
  /**
   * Reads an integer at the given position in the file
   * @see DataInput#readInt
   */
  int readInt(long pos) throws IOException;
  /**
   * Reads a long at the given position in the file
   * @see DataInput#readLong
   */
  long readLong(long pos) throws IOException;
}

