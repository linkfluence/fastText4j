package fasttext.store.util;

public class ByteUtils {

  /**
   * Gets an array of primitive ints from a byte-array with ints written
   * in lower-order first.
   */
  public static int[] getIntArray(byte[] b, int offset, int length) {
    int intArr[] = new int[length];
    int arrOffset = offset;
    for(int i = 0; i < intArr.length; i++) {
      intArr[i] = (b[3 + arrOffset] & 0xFF) | ((b[2 + arrOffset] & 0xFF) << 8)
        | ((b[1 + arrOffset] & 0xFF) << 16) | ((b[arrOffset] & 0xFF) << 24);
      arrOffset += 4;
    }
    return intArr;
  }

}
