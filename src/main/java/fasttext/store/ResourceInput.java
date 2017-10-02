package fasttext.store;

import java.io.Closeable;
import java.io.IOException;

/** 
 * Abstract base class for input from a file. A random-access input stream.
 *
 * <p>{@code ResourceInput} may only be used from one thread, because it is not
 * thread safe (it keeps internal state like file position). To allow
 * multithreaded use, every {@code ResourceInput} instance must be cloned before
 * it is used in another thread. Subclasses must therefore implement {@link #clone()},
 * returning a new {@code ResourceInput} which operates on the same underlying
 * resource, but positioned independently. 
 * 
 * <p><b>Warning:</b> FastText4j never closes cloned
 * {@code ResourceInput}s, it will only call {@link #close()} on the original object.
 * 
 * <p>If you access the cloned ResourceInput after closing the original object,
 * any <code>readXXX</code> methods will throw {@link AlreadyClosedException}.
 *
 * <p>From Lucene IndexInput.
 */
public abstract class ResourceInput extends DataInput implements Cloneable,Closeable {

  private final String resourceDescription;

  /** resourceDescription should be a non-null, opaque string
   *  describing this resource; it's returned from
   *  {@link #toString}. */
  protected ResourceInput(String resourceDescription) {
    if (resourceDescription == null) {
      throw new IllegalArgumentException("resourceDescription must not be null");
    }
    this.resourceDescription = resourceDescription;
  }

  /** Closes the stream to further operations. */
  @Override
  public abstract void close() throws IOException;

  /** Returns the current position in this file, where the next read will
   * occur.
   * @see #seek(long)
   */
  public abstract long getFilePointer();

  /** Sets current position in this file, where the next read will occur.  If this is
   *  beyond the end of the file then this will throw {@code EOFException} and then the
   *  stream is in an undetermined state.
   *
   * @see #getFilePointer()
   */
  public abstract void seek(long pos) throws IOException;

  /** The number of bytes in the file. */
  public abstract long length();

  @Override
  public String toString() {
    return resourceDescription;
  }

  /** {@inheritDoc}
   *
   * <p><b>Warning:</b> FastText never closes cloned
   * {@code ResourceInput}s, it will only call {@link #close()} on the original object.
   *
   * <p>If you access the cloned ResourceInput after closing the original object,
   * any <code>readXXX</code> methods will throw {@link AlreadyClosedException}.
   *
   * <p>This method is NOT thread safe, so if the current {@code ResourceInput}
   * is being used by one thread while {@code clone} is called by another,
   * disaster could strike.
   */
  @Override
  public ResourceInput clone() {
    return (ResourceInput) super.clone();
  }

  /**
   * Creates a slice of this index input, with the given description, offset, and length.
   * The slice is seeked to the beginning.
   */
  public abstract ResourceInput slice(String sliceDescription, long offset, long length) throws IOException;

  /** Subclasses call this to get the String for resourceDescription of a slice of this {@code ResourceInput}. */
  protected String getFullSliceDescription(String sliceDescription) {
    if (sliceDescription == null) {
      // Clones pass null sliceDescription:
      return toString();
    } else {
      return toString() + " [slice=" + sliceDescription + "]";
    }
  }

  /**
   * Creates a random-access slice of this index input, with the given offset and length.
   * <p>
   * The default implementation calls {@link #slice}, and it doesn't support random access,
   * it implements absolute reads as seek+read.
   */
  public RandomAccessInput randomAccessSlice(long offset, long length) throws IOException {
    final ResourceInput slice = slice("randomaccess", offset, length);
    if (slice instanceof RandomAccessInput) {
      // slice() already supports random access
      return (RandomAccessInput) slice;
    } else {
      // return default impl
      return new RandomAccessInput() {
        @Override
        public byte readByte(long pos) throws IOException {
          slice.seek(pos);
          return slice.readByte();
        }

        @Override
        public short readShort(long pos) throws IOException {
          slice.seek(pos);
          return slice.readShort();
        }

        @Override
        public int readInt(long pos) throws IOException {
          slice.seek(pos);
          return slice.readInt();
        }

        @Override
        public long readLong(long pos) throws IOException {
          slice.seek(pos);
          return slice.readLong();
        }

        @Override
        public String toString() {
          return "RandomAccessInput(" + ResourceInput.this.toString() + ")";
        }
      };
    }
  }
}
