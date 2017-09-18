package fasttext;

import com.google.common.primitives.UnsignedLong;
import fasttext.util.Utils;
import org.apache.log4j.Logger;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;
import java.util.function.Predicate;

import static fasttext.util.io.IOUtils.*;

public class Dictionary {

  public enum EntryType {

    WORD(0), LABEL(1);
    private int value;

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

    @Override
    public String toString() {
      return value == 0 ? "word" : value == 1 ? "label" : "unknown";
    }

  }

  public static class Entry implements Comparable<Entry> {

    private String word;
    private long count;
    private List<Integer> subwords = null;
    private EntryType type;

    public Entry() {}

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append("entry [word =");
      sb.append(word);
      sb.append(", count =");
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

  }

  static class EntryComparator implements Comparator<Entry> {

    public int compare(Entry e1, Entry e2) {
      return e1.compareTo(e2);
    }

  }

  private static final int MAX_VOCAB_SIZE = 30000000;
  private static final int MAX_LINE_SIZE = 1024;
  private static final Integer WORD_ID_DEFAULT = -1;
  private static final Charset CHARSET = Charset.forName("UTF-8");

  private static final String EOS = "</s>";
  private static final String BOW = "<";
  private static final String EOW = ">";

  private Args args;

  private Charset charset = CHARSET;

  private Entry[] words;
  private Map<Long, Integer> word2int;
  private double[] pDiscard;

  private long nTokens;
  private int nLabels;
  private int nWords;
  private int size;
  private long pruneIdxSize = -1;
  private Map<Integer, Integer> pruneIdx;

  public Dictionary(Args args) {
    this.args = args;
    this.word2int = new HashMap<>();
    this.size = 0;
    this.nWords = 0;
    this.nLabels = 0;
    this.nTokens = 0L;
  }

  private void initNGrams() {
    for (int i = 0; i < size; i++) {
      String word = BOW + words[i].word + EOW;
      Entry e = words[i];
      if (e.subwords == null) {
        e.subwords = new ArrayList<>();
      }
      e.subwords.add(i);
      computeNGrams(word, e.subwords);
    }
  }

  private void initTableDiscard() {
    pDiscard = new double[size];
    for (int i = 0; i < size; i++) {
      double d = (double) (words[i].count / nTokens);
      pDiscard[i] = Math.sqrt(args.getSamplingTreshold() / d) + args.getSamplingTreshold() / d;
    }
  }

  public int getSize() {
    return this.size;
  }

  private long find(String w) {
    long h = hash(w) % MAX_VOCAB_SIZE;
    while (
      !word2int.getOrDefault(h, WORD_ID_DEFAULT).equals(WORD_ID_DEFAULT) &&
      !words[word2int.getOrDefault(h, WORD_ID_DEFAULT)].word.equals(w)) {
      h = (h + 1) % MAX_VOCAB_SIZE;
    }
    return h;
  }

  public boolean contains(String w) {
    return word2int.get(find(w)) != null;
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

  public int getId(String w) {
    long h = find(w);
    return word2int.getOrDefault(h, WORD_ID_DEFAULT);
  }

  public EntryType getType(String w) {
    return w.startsWith(args.getLabelPrefix()) ? EntryType.LABEL : EntryType.WORD;
  }

  public EntryType getType(int id) {
    assert(id >= 0);
    assert(id < nWords);
    return words[id].type;
  }

  public String getWord(int id) {
    assert(id >= 0);
    assert(id < nWords);
    return words[id].word;
  }

  public List<Integer> getNGrams(int id) {
    assert(id >= 0);
    assert(id < nWords);
    return words[id].subwords;
  }

  public List<Integer> getNGrams(String word) {
    int id = getId(word);
    if (id != WORD_ID_DEFAULT) {
      return getNGrams(id);
    } else {
      List<Integer> ngrams = new ArrayList<>();
      computeNGrams(BOW + word + EOW, ngrams);
      return ngrams;
    }
  }

  public List<Integer> getNGrams(String word, List<Integer> ngrams, List<String> substrings) {
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
    computeNGrams(BOW + word + EOW, ngrams, substrings);
    return ngrams;
  }

  public boolean discard(int id, double rand) {
    assert(id >= 0);
    assert(id < nWords);
    if (args.getModel() == Args.ModelName.SUP) {
      return false;
    } else {
      return rand > pDiscard[id];
    }
  }

  private boolean charMatches(char c) {
    return (c & 0xC0) == 0x80;
  }

  private void computeNGrams(String word, List<Integer> ngrams, List<String> substrings) {
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

  private void computeNGrams(String word, List<Integer> ngrams) {
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

  private static UnsignedLong U64_START = UnsignedLong.valueOf("18446744069414584320");

  private UnsignedLong toUnsignedLong64(long l) {
    if (l > Integer.MAX_VALUE) {
      return U64_START.plus(UnsignedLong.valueOf(l));
    } else {
      return UnsignedLong.valueOf(l);
    }
  }

 private void addNGrams(List<Integer> line, List<Long> hashes, int n) {
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
          if (pruneIdx.get(id) != null) {
            id = pruneIdx.get(id);
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

  public String getLabel(int lid) {
    assert(lid >= 0);
    assert(lid < nLabels);
    return words[lid + nWords].word;
  }

  public List<Long> getCounts(EntryType type) {
    List<Long> counts = new ArrayList<>();
    for (Entry w : words) {
      if (w.type == type) {
        counts.add(w.count);
      }
    }
    return counts;
  }

  private int getDictLine(List<String> tokens, List<Integer> words, List<Long> wordHashes, List<Integer> labels, Random rng) {
    words.clear();
    labels.clear();
    wordHashes.clear();
    int nTokens = 0;
    for (int i = 0; i < tokens.size(); i++) {
      String token = tokens.get(i);
      long h = find(token);
      int wid = word2int.getOrDefault(h, -1);
      if (wid < 0) {
        if (getType(token) == EntryType.WORD) {
          wordHashes.add(hash(token));
        }
        continue;
      }
      EntryType type = getType(wid);
      nTokens++;
      if (type == EntryType.WORD && !discard(wid, Utils.randomFloat(rng, 0, 1))) {
        words.add(wid);
        wordHashes.add(hash(token));
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

  private int getDictLine(List<String> tokens, List<Integer> words, List<Integer> labels, Random rng) {
    List<Long> wordHashes = new ArrayList<>();
    int nTokens = getDictLine(tokens, words, wordHashes, labels, rng);
    if (args.getModel() == Args.ModelName.SUP) {
      addNGrams(words, wordHashes, args.getWordNGrams());
    }
    return nTokens;
  }

  public int getLine(List<String> tokens, List<Integer> words, List<Integer> labels, Random rng) {
    return getDictLine(readLineTokens(tokens), words, labels, rng);
  }

  public int getLine(List<String> tokens, List<Integer> words, List<Long> wordHashes, List<Integer> labels, Random rng) {
    return getDictLine(readLineTokens(tokens), words, wordHashes, labels, rng);
  }

  public int getLine(String line, List<Integer> words, List<Long> wordHashes, List<Integer> labels, Random rng) {
    return getDictLine(readLineTokens(line), words, wordHashes, labels, rng);
  }

  public int getLine(String line, List<Integer> words, List<Integer> labels, Random rng) {
    return getDictLine(readLineTokens(line), words, labels, rng);
  }

  private void setCharset(String charsetName) {
    this.charset = Charset.forName(charsetName);
  }

  void save(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  void load(InputStream is) throws IOException {
    size = readInt(is);
    nWords = readInt(is);
    nLabels = readInt(is);
    nTokens = readLong(is);
    pruneIdxSize = readLong(is);

    word2int = new HashMap<>(size);
    words = new Entry[size];

    for (int i = 0; i < size; i++) {
      Entry e = new Entry();
      e.word = readString(is, charset.name());
      e.count = readLong(is);
      e.type = EntryType.fromValue(readByte(is));
      words[i] = e;
      word2int.put(find(e.word), i);
    }

    pruneIdx = new HashMap<>(Math.max(0, (int) pruneIdxSize));
    if (pruneIdxSize != -1) {
      for (int i = 0; i < pruneIdxSize; i++) {
        int first = readInt(is);
        int second = readInt(is);
        pruneIdx.put(first, second);
      }
    }

    initTableDiscard();
    initNGrams();
  }

}
