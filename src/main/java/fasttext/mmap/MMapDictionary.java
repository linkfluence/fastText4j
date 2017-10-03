package fasttext.mmap;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import fasttext.*;
import fasttext.store.MMapFile;
import fasttext.store.ResourceInput;
import fasttext.store.util.ByteUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Memory-mapped dictionary implementation of {@link BaseDictionary} */
public class MMapDictionary extends BaseDictionary {

  private final MMapFile mmapFile;
  private final long entriesPositionOffset;
  private final int wordByteArrayLength;
  private final int subwordsByteArrayLength;

  protected final long[] wordHashes;
  protected final int[] ids;

  protected final int[] pruneKeys;
  protected final int[] pruneValues;

  private ResourceInput in;

  private MMapDictionary(Args args,
                         int size,
                         int nWords,
                         int nLabels,
                         long nTokens,
                         int pruneIdxSize,
                         MMapFile mmapFile,
                         ResourceInput in,
                         long entriesPositionOffset,
                         int wordByteArrayLength,
                         int subwordsByteArrayLength,
                         long[] wordHashes,
                         int[] ids,
                         int[] pruneKeys,
                         int[] pruneValues) {
    super(args, size, nWords, nLabels, nTokens, pruneIdxSize);
    this.mmapFile = mmapFile;
    this.in = in;
    this.entriesPositionOffset = entriesPositionOffset;
    this.wordByteArrayLength = wordByteArrayLength;
    this.subwordsByteArrayLength = subwordsByteArrayLength;
    this.wordHashes = wordHashes;
    this.ids = ids;
    this.pruneKeys = pruneKeys;
    this.pruneValues = pruneValues;
    // ngrams are already initialized
    initTableDiscard();
  }

  private int entryByteArrayLength() {
    return wordByteArrayLength + subwordsByteArrayLength + Integer.BYTES + Byte.BYTES + Long.BYTES + Integer.BYTES;
  }

  /**
   * Position of an entry based on its id. Returns the position.
   */
  private long entryPosition(int id) {
    return entriesPositionOffset + (entryByteArrayLength()) * id;
  }

  /**
   * Position of an entry field based on the entry's id and the field's offset.
   */
  private long entryFieldPosition(int id, int offset) {
    return entryPosition(id) + offset;
  }

  /**
   * Offset to access to the word in the entry byte array.
   * Word is the first element of the byte array.
   */
  private long wordOffset() {
    return 0L;
  }

  /**
   * Offset to access to the entry type in the entry byte array.
   * Consists in: Integer.BYTES word length + word length
   */
  private int typeOffset() {
    return Integer.BYTES + wordByteArrayLength;
  }

  /**
   * Offset to access to the count value in the entry byte array
   * Consists in: Integer.BYTES word length + word length + Byte.BYTES type encoding
   */
  private int countOffset() {
    return Integer.BYTES + Long.BYTES + wordByteArrayLength;
  }

  /**
   * Offset to access to the subwords array in the entry byte array
   * Consists in: Integer.BYTES word length + word length + Byte.BYTES type encoding + Long.BYTES count
   */
  private int subwordsOffset() {
    return Integer.BYTES + wordByteArrayLength + Byte.BYTES + Long.BYTES;
  }

  private void position(long pos) {
    try {
      in.seek(pos);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not seek position " + pos);
    }
  }

  private EntryType readType() {
    try {
      return EntryType.fromValue(in.readByteAsInt());
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read bytes to EntryType");
    }
  }

  private String readWord() {
    try {
      int currWordLength = in.readInt();
      byte[] barr = new byte[wordByteArrayLength];
      in.readBytes(barr, 0, wordByteArrayLength);
      return new String(barr, 0, currWordLength, StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read bytes to String");
    }
  }

  private long readCount() {
    try {
      return in.readLong();
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read bytes to long");
    }
  }

  private int[] readSubwords() {
    try {
      int currSubwordsSize = in.readInt();
      byte[] barr = new byte[subwordsByteArrayLength];
      in.readBytes(barr, 0, subwordsByteArrayLength);
      return ByteUtils.getIntArray(barr, 0, currSubwordsSize);
    } catch (IOException ex) {
      throw new IllegalArgumentException("Could not read bytes to array of ints");
    }
  }

  private Entry readEntry() {
    Entry e = new Entry();
    e.setWord(readWord());
    e.setCount(readCount());
    e.setType(readType());
    e.setSubwords(Ints.asList(readSubwords()));
    return e;
  }

  @Override
  protected int hashToId(long h) {
    int idx;
    idx = Arrays.binarySearch(wordHashes, h);
    if (idx >= 0) {
      return ids[idx];
    }
    return WORD_ID_DEFAULT;
  }

  @Override
  protected int getPruning(int id) {
    int idx = Arrays.binarySearch(pruneKeys, id);
    if (idx >= 0) {
      return pruneValues[idx];
    }
    return -1;
  }

  @Override
  public Entry getEntry(int id) {
    Preconditions.checkPositionIndex(id, size);
    position(entryPosition(id));
    return readEntry();
  }

  @Override
  public EntryType getType(int id) {
    Preconditions.checkPositionIndex(id, size);
    position(entryFieldPosition(id, typeOffset()));
    return readType();
  }

  @Override
  public String getWord(int id) {
    Preconditions.checkPositionIndex(id, nWords);
    position(entryPosition(id));
    return readWord();
  }

  @Override
  public String getLabel(int lid) {
    Preconditions.checkPositionIndex(lid, nLabels);
    position(entryPosition(lid + nWords));
    return readWord();
  }

  @Override
  public long getCount(int id) {
    Preconditions.checkPositionIndex(id, size);
    position(entryFieldPosition(id, countOffset()));
    return readCount();
  }

  @Override
  public List<Integer> getSubwords(int id) {
    Preconditions.checkPositionIndex(id, size);
    position(entryFieldPosition(id, subwordsOffset()));
    return Ints.asList(readSubwords());
  }

  @Override
  public Entry[] getEntries() {
    Entry[] words = new Entry[size];
    for (int id = 0; id < size; id++) {
      words[id] = getEntry(id);
    }
    return words;
  }

  public static MMapDictionary load(Args args, MMapFile mmap) throws IOException {
    ResourceInput in = mmap.openInput();

    // dictionary mmap utilities
    int wordByteArrayLength = in.readInt();
    int subwordsByteArrayLength = in.readInt();

    // dictionary meta data
    int size = in.readInt();
    int nWords = in.readInt();
    int nLabels = in.readInt();
    long nTokens = in.readLong();
    int pruneIdxSize = (int) in.readLong();

    int[] pruneKeys = new int[pruneIdxSize];
    int[] pruneValues = new int[pruneIdxSize];

    for (int i = 0; i < pruneIdxSize; i++) {
      pruneKeys[i] = in.readInt();
    }
    for (int i = 0; i < pruneIdxSize; i++) {
      pruneValues[i] = in.readInt();
    }

    // word2int
    long[] wordHashes = new long[size];
    int[] ids = new int[size];
    for (int i = 0; i < size; i++) {
      wordHashes[i] = in.readLong();
    }
    for (int i = 0; i < size; i++) {
      ids[i] = in.readInt();
    }

    int entriesPositionOffset = 36 + 8 * pruneIdxSize + 12 * size;

    return new MMapDictionary(args, size, nWords, nLabels, nTokens, pruneIdxSize,
      mmap, in, entriesPositionOffset, wordByteArrayLength, subwordsByteArrayLength,
      wordHashes, ids, pruneKeys, pruneValues);
  }

  @Override
  public MMapDictionary clone() throws CloneNotSupportedException {
    MMapDictionary d = (MMapDictionary) super.clone();
    d.in = in.clone();
    return d;
  }

  public void close() throws IOException {
    in.close();
  }

  public void saveToMMap(OutputStream os) throws IOException {
    throw new UnsupportedOperationException("Not implemented yet");
  }

}
