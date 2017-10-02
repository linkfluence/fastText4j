package fasttext.store;

import fasttext.store.util.Constants;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;

import java.util.Locale;

/**
 * Implementation for access to resources through mmap for reading
 * and {@link OutputStreamResourceOutput} for writing.
 *
 * <p><b>NOTE</b>: memory mapping uses up a portion of the
 * virtual memory address space in your process equal to the
 * size of the file being mapped.  Before using this class,
 * be sure your have plenty of virtual address space, e.g. by
 * using a 64 bit JRE, or a 32 bit JRE with indexes that are
 * guaranteed to fit within the address space.
 * If you get an OutOfMemoryException, it is recommended
 * to reduce the chunk size, until it works.
 *
 * <p> Inspired by Lucene's MMapDirectory.
 */
public class MMapFile {

  private boolean preload;
  private Path path;

  public static final int DEFAULT_MAX_CHUNK_SIZE = Constants.JRE_IS_64BIT ? (1 << 30) : (1 << 28);
  final int chunkSizePower;

  public MMapFile(Path path) throws IOException {
    this(path, DEFAULT_MAX_CHUNK_SIZE);
  }

  public MMapFile(Path path, int maxChunkSize) throws IOException {
    if (maxChunkSize <= 0) {
      throw new IllegalArgumentException("Maximum chunk size for mmap must be >0");
    }
    this.path = path;
    this.chunkSizePower = 31 - Integer.numberOfLeadingZeros(maxChunkSize);
    assert this.chunkSizePower >= 0 && this.chunkSizePower <= 30;
  }

  /**
   * Set to {@code true} to ask mapped pages to be loaded
   * into physical memory on init. The behavior is best-effort
   * and operating system dependent.
   * @see MappedByteBuffer#load
   */
  public void setPreload(boolean preload) {
    this.preload = preload;
  }

  /**
   * Returns {@code true} if mapped pages should be loaded.
   * @see #setPreload
   */
  public boolean getPreload() {
    return preload;
  }


  static final int OUTPUT_CHUNK_SIZE = 8192;

  public ResourceOutput createOutput(int bufferSize, OpenOption... options) throws IOException {
    return new OutputStreamResourceOutput(path.toString(), new FilterOutputStream(Files.newOutputStream(path, options)), bufferSize);
  }

  public ResourceOutput createOutput(OpenOption... options) throws IOException {
    return createOutput(OUTPUT_CHUNK_SIZE, options);
  }

  public ResourceOutput createOutput(int bufferSize) throws IOException {
    return createOutput(bufferSize, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
  }

  public ResourceOutput createOutput() throws IOException {
    return createOutput(OUTPUT_CHUNK_SIZE, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
  }


  /** Creates an IndexInput for the file with the given name. */
  public ResourceInput openInput() throws IOException {
    try (FileChannel c = FileChannel.open(path, StandardOpenOption.READ)) {
      final String resourceDescription = "MMapIndexInput(path=\"" + path.toString() + "\")";
      return ByteBufferResourceInput.newInstance(resourceDescription,
        map(resourceDescription, c, 0, c.size()),
        c.size(), chunkSizePower);
    }
  }

  /** Maps a file into a set of buffers */
  final ByteBuffer[] map(String resourceDescription, FileChannel fc, long offset, long length) throws IOException {
    if ((length >>> chunkSizePower) >= Integer.MAX_VALUE)
      throw new IllegalArgumentException("RandomAccessFile too big for chunk size: " + resourceDescription);

    final long chunkSize = 1L << chunkSizePower;

    // we always allocate one more buffer, the last one may be a 0 byte one
    final int nrBuffers = (int) (length >>> chunkSizePower) + 1;

    ByteBuffer buffers[] = new ByteBuffer[nrBuffers];

    long bufferStart = 0L;
    for (int bufNr = 0; bufNr < nrBuffers; bufNr++) {
      int bufSize = (int) ( (length > (bufferStart + chunkSize))
        ? chunkSize
        : (length - bufferStart)
      );
      MappedByteBuffer buffer;
      try {
        buffer = fc.map(FileChannel.MapMode.READ_ONLY, offset + bufferStart, bufSize);
      } catch (IOException ioe) {
        throw convertMapFailedIOException(ioe, resourceDescription, bufSize);
      }
      if (preload) {
        buffer.load();
      }
      buffers[bufNr] = buffer;
      bufferStart += bufSize;
    }

    return buffers;
  }

  private IOException convertMapFailedIOException(IOException ioe, String resourceDescription, int bufSize) {
    final String originalMessage;
    final Throwable originalCause;
    if (ioe.getCause() instanceof OutOfMemoryError) {
      // nested OOM confuses users, because it's "incorrect", just print a plain message:
      originalMessage = "Map failed";
      originalCause = null;
    } else {
      originalMessage = ioe.getMessage();
      originalCause = ioe.getCause();
    }
    final String moreInfo;
    if (!Constants.JRE_IS_64BIT) {
      moreInfo = "MMapDirectory should only be used on 64bit platforms, because the address space on 32bit operating systems is too small. ";
    } else if (Constants.WINDOWS) {
      moreInfo = "Windows is unfortunately very limited on virtual address space. If your index size is several hundred Gigabytes, consider changing to Linux. ";
    } else if (Constants.LINUX) {
      moreInfo = "Please review 'ulimit -v', 'ulimit -m' (both should return 'unlimited'), and 'sysctl vm.max_map_count'. ";
    } else {
      moreInfo = "Please review 'ulimit -v', 'ulimit -m' (both should return 'unlimited'). ";
    }
    final IOException newIoe = new IOException(String.format(Locale.ENGLISH,
      "%s: %s [this may be caused by lack of enough unfragmented virtual address space "+
        "or too restrictive virtual memory limits enforced by the operating system, "+
        "preventing us to map a chunk of %d bytes." +
      originalMessage, resourceDescription, bufSize, moreInfo), originalCause);
    newIoe.setStackTrace(ioe.getStackTrace());
    return newIoe;
  }

  public Path getPath() {
    return path;
  }

}
