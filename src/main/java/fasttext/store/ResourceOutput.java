package fasttext.store;

import java.io.Closeable;
import java.io.IOException;

/**
 * Abstract base class for output to a file.  A random-access output stream.
 *
 * <p>{@code ResourceOutput} may only be used from one thread, because it is not
 * thread safe (it keeps internal state like file position).
 *
 * <p>From Lucene IndexOutput.
 *
 * @see ResourceInput
 */
public abstract class ResourceOutput extends DataOutput implements Closeable {

  /** Full description of this output, e.g. which class such as {@code FSResourceOutput},
   * and the full path to the file
   */
  private final String resourceDescription;

  /** Just the name part from {@code resourceDescription} */
  private final String name;

  /** Sole constructor.  resourceDescription should be non-null, opaque string
   *  describing this resource; it's returned from {@link #toString}.
   */
  protected ResourceOutput(String resourceDescription, String name) {
    if (resourceDescription == null) {
      throw new IllegalArgumentException("resourceDescription must not be null");
    }
    this.resourceDescription = resourceDescription;
    this.name = name;
  }

  /** Returns the name used to create this {@code ResourceOutput}.*/
  public String getName() {
    return name;
  }

  /** Closes this stream to further operations. */
  @Override
  public abstract void close() throws IOException;

  /** Returns the current position in this file, where the next write will
   * occur.
   */
  public abstract long getFilePointer();

  /** Returns the current checksum of bytes written so far */
  public abstract long getChecksum() throws IOException;

  @Override
  public String toString() {
    return resourceDescription;
  }

}
