package fasttext.store;

/**
 * This exception is thrown when there is an attempt to
 * access something that has already been closed.
 */
public class AlreadyClosedException extends IllegalStateException {
  public AlreadyClosedException(String message) {
    super(message);
  }

  public AlreadyClosedException(String message, Throwable cause) {
    super(message, cause);
  }
}
