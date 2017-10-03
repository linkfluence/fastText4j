package fasttext;

import com.google.common.base.Preconditions;
import fasttext.store.InputStreamFastTextInput;
import fasttext.store.OutputStreamFastTextOutput;
import fasttext.store.OutputStreamResourceOutput;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Implementation class of {@link BaseDictionary} using fastText model */
public class Dictionary extends BaseDictionary {

  private final Entry[] words;
  private final Map<Long, Integer> word2int;
  private final Map<Integer, Integer> pruneIdx;

  private Dictionary(Args args,
                     int size,
                     int nWords,
                     int nLabels,
                     long nTokens,
                     int pruneIdxSize,
                     Entry[] words,
                     Map<Long, Integer> word2int,
                     Map<Integer, Integer> pruneIdx) {
    super(args, size, nWords, nLabels, nTokens, pruneIdxSize);
    this.words = words;
    this.word2int = word2int;
    this.pruneIdx = pruneIdx;
    initWord2int();
    initTableDiscard();
    initSubwords();
  }

  @Override
  protected int hashToId(long h) {
    return word2int.getOrDefault(h, WORD_ID_DEFAULT);
  }

  @Override
  protected int getPruning(int id) {
    return pruneIdx.getOrDefault(id, -1);
  }

  @Override
  public Entry[] getEntries() { return this.words; }

  @Override
  public Entry getEntry(int id) {
    Preconditions.checkPositionIndex(id, size);
    return words[id];
  }

  @Override
  public EntryType getType(int id) {
    Preconditions.checkPositionIndex(id, size);
    return words[id].type;
  }

  @Override
  public String getWord(int id) {
    Preconditions.checkPositionIndex(id, nWords);
    return words[id].word;
  }

  @Override
  public String getLabel(int lid) {
    Preconditions.checkPositionIndex(lid, nLabels);
    return words[lid + nWords].word;
  }

  @Override
  public long getCount(int id) {
    Preconditions.checkPositionIndex(id, size);
    return words[id].count;
  }

  @Override
  public List<Integer> getSubwords(int id) {
    Preconditions.checkPositionIndex(id, nWords);
    return words[id].subwords;
  }

  private void initWord2int() {
    for (int i = 0; i < size; i++) {
      Entry e = getEntry(i);
      word2int.put(find(e.word), i);
    }
  }

  public static Dictionary load(Args args, InputStreamFastTextInput is) throws IOException {
    int size = is.readInt();
    int nWords = is.readInt();
    int nLabels = is.readInt();
    long nTokens = is.readLong();
    int pruneIdxSize = (int) is.readLong();
    Entry[] words = new Entry[size];
    Map<Long, Integer> word2int = new HashMap<>(size);

    for (int i = 0; i < size; i++) {
      Entry e = new Entry();
      e.word = is.readString();
      e.count = is.readLong();
      e.type = EntryType.fromValue(is.readByteAsInt());
      words[i] = e;
    }

    Map<Integer, Integer> pruneIdx = new HashMap<>(Math.max(0, pruneIdxSize));
    if (pruneIdxSize != -1) {
      for (int i = 0; i < pruneIdxSize; i++) {
        int first = is.readInt();
        int second = is.readInt();
        pruneIdx.put(first, second);
      }
    }
    return new Dictionary(args, size, nWords, nLabels, nTokens,
      pruneIdxSize, words, word2int, pruneIdx);
  }

  public void save(OutputStreamFastTextOutput os) throws IOException {
    os.writeInt(size);
    os.writeInt(nWords);
    os.writeInt(nLabels);
    os.writeLong(nTokens);
    os.writeLong(pruneIdxSize);
    for (int i = 0; i < size; i++) {
      Entry e = words[i];
      os.writeString(e.word);
      os.writeLong(e.count);
      os.writeIntAsByte(e.type.getValue());
    }
    for (Map.Entry<Integer, Integer> pair : pruneIdx.entrySet()) {
      os.writeInt(pair.getKey());
      os.writeInt(pair.getValue());
    }
  }

  public void saveToMMap(OutputStream os) throws IOException {
    List<Pair<Long, Integer>> orderedWord2int = new ArrayList<>(size);
    for (Map.Entry<Long, Integer> w2i : word2int.entrySet()) {
      orderedWord2int.add(new Pair<>(w2i.getKey(), w2i.getValue()));
    }
    orderedWord2int.sort(Comparator.comparing(Pair::first));

    List<Pair<Integer, Integer>> orderedPruneIdx = new ArrayList<>(pruneIdxSize);
    for (Map.Entry<Integer, Integer> p : pruneIdx.entrySet()) {
      orderedPruneIdx.add(new Pair<>(p.getKey(), p.getValue()));
    }
    orderedPruneIdx.sort(Comparator.comparing(Pair::first));

    // find byte array size for words and subwords
    int maxStringLength = Integer.MIN_VALUE;
    int maxSubwordsSize = Integer.MIN_VALUE;
    for (int i = 0; i < size; i++) {
      Entry e = words[i];
      if (e.word().getBytes(StandardCharsets.UTF_8).length > maxStringLength) {
        maxStringLength = e.word().getBytes(StandardCharsets.UTF_8).length;
      }
      if (e.subwords().size() > maxSubwordsSize) {
        maxSubwordsSize = e.subwords().size();
      }
    }
    int wordByteArrayLength = maxStringLength;
    int subwordsByteArrayLength = Integer.BYTES * maxSubwordsSize;


    int entryByteArraySize = (
      wordByteArrayLength +     // word
      Integer.BYTES +           // word length
      Byte.BYTES +              // entryType
      Long.BYTES +              // count
      subwordsByteArrayLength + // subwords array
      Integer.BYTES             // subwords array length
    );
    int headerSize = 36 + 8 * pruneIdxSize + 12 * size;
    int bufferSize = headerSize + entryByteArraySize * words.length;

    try (OutputStreamResourceOutput fos =
           new OutputStreamResourceOutput("dictionary", os, bufferSize)) {
      // dictionary utilities
      fos.writeInt(wordByteArrayLength);
      fos.writeInt(subwordsByteArrayLength);
      // dictionary meta data
      fos.writeInt(size);
      fos.writeInt(nWords);
      fos.writeInt(nLabels);
      fos.writeLong(nTokens);
      fos.writeLong(pruneIdxSize);
      for (Pair<Integer, Integer> pair : orderedPruneIdx) {
        fos.writeInt(pair.first());
      }
      for (Pair<Integer, Integer> pair : orderedPruneIdx) {
        fos.writeInt(pair.last());
      }
      for (Pair<Long, Integer> pair : orderedWord2int) {
        fos.writeLong(pair.first());
      }
      for (Pair<Long, Integer> pair : orderedWord2int) {
        fos.writeInt(pair.last());
      }
      // dictionary entries
      for (int i = 0; i < size; i++) {
        Entry e = words[i];
        // write word
        ByteBuffer wordBuffer = ByteBuffer.allocate(wordByteArrayLength);
        byte[] wordBytes = e.word().getBytes(StandardCharsets.UTF_8);
        wordBuffer.put(wordBytes);
        wordBuffer.flip();
        fos.writeInt(wordBytes.length);
        fos.writeBytes(wordBuffer.array(), wordByteArrayLength);
        // write count
        fos.writeLong(e.count());
        // write type
        fos.writeIntAsByte(e.type().getValue());
        // write subwords
        ByteBuffer subwordsBuffer = ByteBuffer.allocate(subwordsByteArrayLength);
        for (int subword : e.subwords()) {
          subwordsBuffer.putInt(subword);
        }
        subwordsBuffer.flip();
        fos.writeInt(e.subwords().size());
        fos.writeBytes(subwordsBuffer.array(), subwordsByteArrayLength);
      }
    }
  }

  @Override
  public Dictionary clone() throws CloneNotSupportedException {
    return (Dictionary) super.clone();
  }

  public void close() {}

}
