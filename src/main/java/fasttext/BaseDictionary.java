package fasttext;

import com.google.common.base.Preconditions;
import com.google.common.primitives.UnsignedLong;
import fasttext.util.Randoms;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.*;
import java.util.function.Predicate;

/** Base class for fastText dictionary implementation */
public abstract class BaseDictionary implements Cloneable, Closeable {

  public enum EntryType {

    WORD(0), LABEL(1);
    protected int value;

    EntryType(int value) {
      this.value = value;
    }

    public static EntryType fromValue(int value) throws IllegalArgumentException {
      try {
        return EntryType.values()[value];
      } catch (ArrayIndexOutOfBoundsException e) {
        throw new IllegalArgumentException("Unknown entry_type enum value :" + value);
      }
    }

    public int getValue() { return value; }

    @Override
    public String toString() {
      return value == 0 ? "word" : value == 1 ? "label" : "unknown";
    }

  }

  public static class Entry implements Comparable<Entry> {

    String word;
    long count;
    List<Integer> subwords = null;
    EntryType type;

    public Entry() {}

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("entry [word=");
      sb.append(word);
      sb.append(", count=");
      sb.append(count);
      sb.append(", type=");
      sb.append(type);
      sb.append(", subwords=(");
      for (int i = 0; i < subwords.size(); i++) {
        sb.append(i);
        if (i != subwords.size() - 1) {
          sb.append(", ");
        }
      }
      sb.append(")");
      sb.append("]");
      return sb.toString();
    }

    public int compareTo(Entry e) {
      if (this.type != e.type) {
        return this.type.compareTo(e.type);
      } else if (this.count == e.count) {
        return 1;
      } else if (this.count > e.count) {
        return 1;
      } else {
        return -1;
      }
    }

    public void setWord(String w) {
      word = w;
    }

    public String word() {
      return word;
    }

    public void setCount(long c) {
      count = c;
    }

    public long count() {
      return count;
    }

    public void setType(EntryType t) {
      type = t;
    }

    public EntryType type() {
      return type;
    }

    public void setSubwords(List<Integer> sw) {
      subwords = sw;
    }

    public List<Integer> subwords() {
      return subwords;
    }

  }

  static class EntryComparator implements Comparator<Entry> {

    public int compare(Entry e1, Entry e2) {
      return e1.compareTo(e2);
    }

  }

  public static final int MAX_VOCAB_SIZE = 30000000;
  public static final int MAX_LINE_SIZE = 1024;
  public static final Integer WORD_ID_DEFAULT = -1;

  private static final String EOS = "</s>";
  private static final String BOW = "<";
  private static final String EOW = ">";

  protected final Args args;

  protected final long nTokens;
  protected final int nLabels;
  protected final int nWords;
  protected final int size;
  protected final int pruneIdxSize;

  protected double[] pDiscard;

  protected BaseDictionary(Args args,
                           int size,
                           int nWords,
                           int nLabels,
                           long nTokens,
                           int pruneIdxSize) {
    this.args = args;
    this.size = size;
    this.nWords = nWords;
    this.nLabels = nLabels;
    this.nTokens = nTokens;
    this.pruneIdxSize = pruneIdxSize;
  }

  public abstract Entry[] getEntries();

  public abstract Entry getEntry(int id);

  public abstract String getWord(int id);

  public abstract String getLabel(int lid);

  public abstract long getCount(int id);

  public EntryType getType(String w) {
    return w.startsWith(args.getLabelPrefix()) ? EntryType.LABEL : EntryType.WORD;
  }

  public abstract EntryType getType(int id);

  public abstract List<Integer> getSubwords(int id);

  public List<Integer> getSubwords(String word) {
    int id = getId(word);
    if (id != WORD_ID_DEFAULT) {
      return getSubwords(id);
    } else {
      List<Integer> ngrams = new ArrayList<>();
      computeSubwords(BOW + word + EOW, ngrams);
      return ngrams;
    }
  }

  public List<Integer> getSubwords(String word, List<Integer> ngrams, List<String> substrings) {
    int id = getId(word);
    ngrams.clear();
    substrings.clear();
    if (id == WORD_ID_DEFAULT) {
      ngrams.add(WORD_ID_DEFAULT);
      substrings.add(word);
    } else {
      ngrams.add(id);
      substrings.add(word);
    }
    computeSubwords(BOW + word + EOW, ngrams, substrings);
    return ngrams;
  }

  protected abstract int hashToId(long h);

  protected abstract int getPruning(int id);

  public int getId(String w) {
    return hashToId(find(w));
  }

  protected int getId(String w, long h) {
    return hashToId(find(w, h));
  }

  public int size() {
    return this.size;
  }

  public long nTokens() {
    return nTokens;
  }

  public int nLabels() {
    return nLabels;
  }

  public int nWords() {
    return nWords;
  }

  protected void initSubwords() {
    for (int i = 0; i < size; i++) {
      Entry e = getEntry(i);
      String word = BOW + e.word + EOW;
      if (e.subwords == null) {
        e.subwords = new ArrayList<>();
      }
      e.subwords.add(i);
      computeSubwords(word, e.subwords);
    }
  }

  protected void initTableDiscard() {
    pDiscard = new double[size];
    for (int i = 0; i < size; i++) {
      double d = (double) (getCount(i) / nTokens);
      pDiscard[i] = Math.sqrt(args.getSamplingTreshold() / d) + args.getSamplingTreshold() / d;
    }
  }

  protected boolean idNotFound(String w, long h) {
    int id = hashToId(h);
    return !(id == WORD_ID_DEFAULT) && !(getEntry(id).word.equals(w));
  }

  protected long find(String w, long hw) {
    long h = hw % MAX_VOCAB_SIZE;
    while (idNotFound(w, h)) {
      h = (h + 1) % MAX_VOCAB_SIZE;
    }
    return h;
  }

  protected long find(String w) {
    return find(w, hash(w));
  }

  /**
   * String FNV-1a 32 bits Hash
   * @param str
   * @return
   */
  public long hash(final String str) {
    int h = (int) 2166136261L;      // 0xffffffc5;
    for (byte strByte : str.getBytes()) {
      h = (h ^ strByte) * 16777619; // FNV-1a
    }
    return h & 0xffffffffL;
  }

  public boolean contains(String w) {
    return hashToId(find(w)) >= 0;
  }

  protected boolean discard(int id, double rand) {
    Preconditions.checkArgument(id >= 0);
    Preconditions.checkArgument(id < nWords);
    if (args.getModel() == Args.ModelName.SUP) {
      return false;
    } else {
      return rand > pDiscard[id];
    }
  }

  protected boolean charMatches(char c) {
    return (c & 0xC0) == 0x80;
  }

  protected void computeSubwords(String word, List<Integer> ngrams, List<String> substrings) {
    for(int i = 0; i < word.length(); i++) {
      StringBuilder ngram = new StringBuilder();
      if (!charMatches(word.charAt(i))) {
        for (int j = i, n = 1; j < word.length() && n <= args.getMaxn(); n++) {
          ngram.append(word.charAt(j++));
          while (j < word.length() && charMatches(word.charAt(j))) {
            ngram.append(word.charAt(j++));
          }
          if (n >= args.getMinn() && !(n == 1 && (i == 0 || j == word.length()))) {
            UnsignedLong h = UnsignedLong.valueOf(hash(ngram.toString()));
            h = h.mod(UnsignedLong.valueOf(args.getBucketNumber()));
            ngrams.add(nWords + h.intValue());
            substrings.add(ngram.toString());
          }
        }
      }
    }
  }

  protected void computeSubwords(String word, List<Integer> ngrams) {
    for(int i = 0; i < word.length(); i++) {
      StringBuilder ngram = new StringBuilder();
      if (!charMatches(word.charAt(i))) {
        for (int j = i, n = 1; j < word.length() && n <= args.getMaxn(); n++) {
          ngram.append(word.charAt(j++));
          while (j < word.length() && charMatches(word.charAt(j))) {
            ngram.append(word.charAt(j++));
          }
          if (n >= args.getMinn() && !(n == 1 && (i == 0 || j == word.length()))) {
            UnsignedLong h = UnsignedLong.valueOf(hash(ngram.toString()));
            h = h.mod(UnsignedLong.valueOf(args.getBucketNumber()));
            ngrams.add(nWords + h.intValue());
          }
        }
      }
    }
  }

  private static UnsignedLong U64_START = UnsignedLong.valueOf("18446744069414584320");

  protected UnsignedLong toUnsignedLong64(long l) {
    if (l > Integer.MAX_VALUE) {
      return U64_START.plus(UnsignedLong.valueOf(l));
    } else {
      return UnsignedLong.valueOf(l);
    }
  }

  protected void addWordNGrams(List<Integer> line, List<Long> hashes, int n) {
    if (pruneIdxSize == 0) {
      return;
    }
    for (int i = 0; i < hashes.size(); i++) {
      UnsignedLong h = toUnsignedLong64(hashes.get(i));
      for (int j = i + 1; j < hashes.size() && j < i + n; j++) {
        long h2 = (long) hashes.get(j).intValue();
        long coeff = 116049371L;
        if (h2 >= 0) {
          h = h.times(UnsignedLong.valueOf(coeff)).plus(UnsignedLong.valueOf(h2));
        } else {
          h = h.times(UnsignedLong.valueOf(coeff)).minus(UnsignedLong.valueOf(-h2));
        }
        int id = h.mod(UnsignedLong.valueOf(args.getBucketNumber())).intValue();
        if (pruneIdxSize > 0) {
          int pruneId = getPruning(id);
          if (pruneId >= 0) {
            id = pruneId;
          } else {
            continue;
          }
        }
        line.add(nWords + id);
      }
    }
  }

  private static Predicate<Integer> SPACEBREAK_CODEPOINT = (cp ->
    (cp == 0x00A0) ||                  // no-break space
      (cp == 0x0009) ||                  // horizontal tabulation
      (cp >= 0x000A && cp <= 0x000D) ||  // line feed, line tabulation, form feed, carriage return
      (cp == 0x0020) ||                  // space
      (cp == 0x0085) ||                  // next line
      (cp == 0x1680) ||                  // Ogham space mark
      (cp >= 0x2028 && cp <= 0x2029) ||  // line separator Zl block or paragraph separator Zp block
      (cp >= 0x2000 && cp <= 0x200A) ||  // space separator Zs
      (cp == 0x202F) ||                  // space separator Zs: narrow no-break space
      (cp == 0x205F) ||                  // space separator Zs: medium mathematical space
      (cp == 0x3000)                     // space separator Zs:ideographic space
  );

  public List<String> readLineTokens(List<String> tokens) {
    List<String> lineTokens = new ArrayList<>(tokens);
    lineTokens.add(EOS);
    return lineTokens;
  }

  public List<String> readLineTokens(String line) {
    List<String> lineTokens = new ArrayList<>();
    StringBuilder token = new StringBuilder();
    for (int cp : line.codePoints().toArray()) {
      if (SPACEBREAK_CODEPOINT.test(cp)) {
        if (token.length() > 0) {
          lineTokens.add(token.toString());
          token.setLength(0);
        }
      } else {
        token.appendCodePoint(cp);
      }
    }
    if (token.length() != 0) {
      lineTokens.add(token.toString());
    }
    lineTokens.add(EOS);
    return lineTokens;
  }

  public List<Long> getCounts(EntryType type) {
    List<Long> counts = new ArrayList<>();
    for (int i = 0; i < size; i++) {
      Entry e = getEntry(i);
      if (e.type == type) {
        counts.add(e.count);
      }
    }
    return counts;
  }

  protected int getDictLine(List<String> tokens,
                            List<Integer> lineWords,
                            List<Long> wordHashes,
                            List<Integer> labels,
                            Random rng) {
    lineWords.clear();
    labels.clear();
    wordHashes.clear();
    int nTokens = 0;
    for (int i = 0; i < tokens.size(); i++) {
      String token = tokens.get(i);
      long h = hash(token);
      int wid = getId(token, h);
      if (wid < 0) {
        if (getType(token) == EntryType.WORD) {
          wordHashes.add(h);
        }
        continue;
      }
      EntryType type = getType(wid);
      nTokens++;
      if (type == EntryType.WORD && !discard(wid, Randoms.randomFloat(rng, 0, 1))) {
        lineWords.add(wid);
        wordHashes.add(h);
      }
      if (type == EntryType.LABEL) {
        labels.add(wid - nWords);
      }
      if (token.equals(EOS)) {
        break;
      }
      if (nTokens > MAX_LINE_SIZE && args.getModel() != Args.ModelName.SUP) {
        break;
      }
    }
    return nTokens;
  }

  protected int getDictLine(List<String> tokens,
                            List<Integer> words,
                            List<Integer> labels,
                            Random rng) {
    List<Long> wordHashes = new ArrayList<>();
    int nTokens = getDictLine(tokens, words, wordHashes, labels, rng);
    if (args.getModel() == Args.ModelName.SUP) {
      addWordNGrams(words, wordHashes, args.getWordNGrams());
    }
    return nTokens;
  }

  public int getLine(List<String> tokens,
                     List<Integer> words,
                     List<Integer> labels,
                     Random rng) {
    return getDictLine(readLineTokens(tokens), words, labels, rng);
  }

  public int getLine(List<String> tokens,
                     List<Integer> words,
                     List<Long> wordHashes,
                     List<Integer> labels,
                     Random rng) {
    return getDictLine(readLineTokens(tokens), words, wordHashes, labels, rng);
  }

  public int getLine(String line,
                     List<Integer> words,
                     List<Long> wordHashes,
                     List<Integer> labels,
                     Random rng) {
    return getDictLine(readLineTokens(line), words, wordHashes, labels, rng);
  }

  public int getLine(String line,
                     List<Integer> words,
                     List<Integer> labels,
                     Random rng) {
    return getDictLine(readLineTokens(line), words, labels, rng);
  }

  public abstract void saveToMMap(OutputStream os) throws IOException;

  @Override
  public BaseDictionary clone() throws CloneNotSupportedException {
    return (BaseDictionary) super.clone();
  }

  public abstract void close() throws IOException;

}
