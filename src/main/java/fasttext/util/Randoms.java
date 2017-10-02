package fasttext.util;

import java.util.Random;

public class Randoms {

  public static float randomFloat(Random rnd, float lower, float upper) {
    assert(lower <= upper);
    if (lower == upper) {
      return lower;
    }
    return (rnd.nextFloat() * (upper - lower)) + lower;
  }

}
