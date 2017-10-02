package fasttext;

import com.google.common.collect.MinMaxPriorityQueue;
import com.google.common.primitives.Booleans;
import com.google.common.primitives.Ints;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class Model {

  private static final int SIGMOID_TABLE_SIZE = 512;
  private static final int MAX_SIGMOID = 8;
  private static final int LOG_TABLE_SIZE = 512;

  private static final int NEGATIVE_TABLE_SIZE = 10000000;

  public class Node {
    int parent;
    int left;
    int right;
    long count;
    boolean binary;
  }

  private final ReadableMatrix wi;
  private final Matrix wo;

  private final ReadableQMatrix qwi;
  private final QMatrix qwo;

  private final Args args;

  private final Vector hidden;
  private final Vector output;
  private final Vector grad;
  private final boolean quant;
  private final int hsz;
  private final int osz;
  private final float loss;
  private final int nexamples;
  private float[] tSigmoid;
  private float[] tLog;


  private int[] negatives;
  private int negpos;

  private int[][] paths;
  private boolean[][] codes;
  private Node[] tree;

  private transient Random rng;

  public Model(Args args,
               int seed,
               ReadableMatrix wi,
               Matrix wo,
               boolean quant,
               boolean qout,
               ReadableQMatrix qwi,
               QMatrix qwo) {
    this.hidden = new Vector(args.getDimension());
    this.output = new Vector(wo.m());
    this.grad = new Vector(args.getDimension());
    this.rng = new Random(seed);
    this.wi = wi;
    this.wo = wo;
    this.args = args;
    this.hsz = args.getDimension();
    this.negpos = 0;
    this.loss = 0.0f;
    this.nexamples = 1;
    this.quant = quant;
    this.qwi = qwi;
    this.qwo = qwo;
    if (qout) {
      this.osz = qwo.m();
    } else {
      this.osz = wo.m();
    }
    initSigmoid();
    initLog();
  }

  public Random rng() {
    return this.rng;
  }

  public float binaryLogistic(int target, boolean label, float lr) {
    float score = sigmoid(wo.dotRow(hidden, target));
    float slabel = label ? 1.0f : 0.0f;
    float alpha = lr * (slabel - score);
    grad.addRow(wo, target, alpha);
    wo.addRow(hidden, target, alpha);
    if (label) {
      return -log(score);
    } else {
      return -log(1.0f - score);
    }
  }

  public float negativeSampling(int target, float lr) {
    float loss = 0.0f;
    grad.zero();
    for (int n = 0; n < args.getNeg(); n++) {
      if (n == 0) {
        loss += binaryLogistic(target, true, lr);
      } else {
        loss += binaryLogistic(getNegative(target), false, lr);
      }
    }
    return loss;
  }

  public float hierarchicalSoftmax(int target, float lr) {
    float loss = 0.0f;
    grad.zero();
    boolean[] binaryCodes = codes[target];
    int[] pathToRoot = paths[target];
    for (int i = 0; i < pathToRoot.length; i++) {
      loss += binaryLogistic(pathToRoot[i], binaryCodes[i], lr);
    }
    return loss;
  }

  public void computeOutputSoftmax(Vector hidden, Vector output) {
    if (quant && args.getQOut()) {
      output.mul(qwo, hidden);
    } else {
      output.mul(wo, hidden);
    }
    double max = output.at(0), z = 0.0f;
    for (int i = 0; i < osz; i++) {
      max = Math.max(output.at(i), max);
    }
    for (int i = 0; i < osz; i++) {
      float p = (float) Math.exp(output.data[i] - max);
      z += p;
      output.set(i, p);
    }
    for (int i = 0; i < osz; i++) {
      output.data[i] /= z;
    }
  }

  public void computeOutputSoftmax() {
    computeOutputSoftmax(hidden, output);
  }

  public float softmax(int target, float lr) {
    grad.zero();
    computeOutputSoftmax();
    for (int i = 0; i < osz; i++) {
      float label = (i == target) ? 1.0f : 0.0f;
      float alpha = lr * (label - output.at(i));
      grad.addRow(wo, i, alpha);
      wo.addRow(hidden, i, alpha);
    }
    return -log(output.at(target));
  }

  public void computeHidden(int[] input, Vector hidden) {
    assert(hidden.size() == hsz);
    hidden.zero();
    for(int it : input) {
      if (quant) {
        hidden.addRow(qwi, it);
      } else {
        hidden.addRow(wi, it);
      }
    }
    hidden.mul(1.0f / input.length);
  }

  static class HeapComparator<T> implements Comparator<Pair<Float, T>> {
    @Override
    public int compare(Pair<Float, T> p1, Pair<Float, T> p2) {
      if (p1.first().equals(p2.first())) {
        return 0;
      } else if (p1.first() < p2.first()) {
        return 1;
      } else {
        return -1;
      }
    }
  }

  public void predict(int[] input, int k, MinMaxPriorityQueue<Pair<Float, Integer>> heap) {
    assert(k > 0);
    computeHidden(input, hidden);
    if (args.getLoss().equals(Args.LossName.HS)) {
      dfs(k, 2 * osz - 2, 0.0f, heap, hidden);
    } else {
      findKBest(k, heap, hidden, output);
    }
  }

  public void predict(int[] input, int k, MinMaxPriorityQueue<Pair<Float, Integer>> heap, Vector hidden, Vector output) {
    assert(k > 0);
    computeHidden(input, hidden);
    if (args.getLoss().equals(Args.LossName.HS)) {
      dfs(k, 2 * osz - 2, 0.0f, heap, hidden);
    } else {
      findKBest(k, heap, hidden, output);
    }
  }

  public void findKBest(int k, MinMaxPriorityQueue<Pair<Float, Integer>> heap, Vector hidden, Vector output) {
    computeOutputSoftmax(hidden, output);
    for (int i = 0; i < osz; i++) {
      if (heap.size() == k && log(output.data[i]) < heap.peekFirst().first()) {
        continue;
      }
      heap.add(new Pair<>(log(output.data[i]), i));
    }
    while (heap.size() > k) {
      heap.pollLast();
    }
  }

  public void dfs(int k, int node, float score, MinMaxPriorityQueue<Pair<Float, Integer>> heap, Vector hidden) {
    if (heap.size() == k && score < heap.peekFirst().first()) {
      return;
    }
    if (tree[node].left == -1 && tree[node].right == -1) {
      heap.add(new Pair<>(score, node));
      while (heap.size() > k) {
        heap.pollLast();
      }
      return;
    }
    float f;
    if (quant && args.getQOut()) {
      f = sigmoid(qwo.dotRow(hidden, node - osz));
    } else {
      f = sigmoid(wo.dotRow(hidden, node - osz));
    }
    dfs(k, tree[node].left, score + log(1.0f - f), heap, hidden);
    dfs(k, tree[node].right, score + log(f), heap, hidden);
  }

  public void update(int[] input, int target, float lr) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void setTargetCounts(long[] counts) {
    assert(counts.length == osz);
    if (args.getLoss() == Args.LossName.NS) {
      initTableNegative(counts);
    } else if (args.getLoss() == Args.LossName.HS) {
      buildTree(counts);
    }
  }

  public void initTableNegative(long[] counts) {
    float z = 0.0f;
    List<Integer> negs = new ArrayList<>();
    for (int i = 0; i < counts.length; i++) {
      z += (float) Math.pow(counts[i], 0.5);
    }
    for (int i = 0; i < counts.length; i++) {
      float c = (float) Math.pow(counts[i], 0.5);
      for (int j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
        negs.add(i);
      }
    }
    negatives = negs.stream().mapToInt(i->i).toArray();
  }

  public int getNegative(long target) {
    int negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.length;
    while (target == negative) {
      negative = negatives[negpos];
      negpos = (negpos + 1) % negatives.length;
    }
    return negative;
  }

  public void buildTree(long[] counts) {
    tree = new Node[2 * osz - 1];
    paths = new int[osz][];
    codes = new boolean[osz][];
    for (int i = 0; i < 2 * osz - 1; i++) {
      Node n = new Node();
      n.parent = -1;
      n.left = -1;
      n.right = -1;
      n.count = 1000000000000000L; // 1e15f
      n.binary = false;
      tree[i] = n;
    }
    for (int i = 0; i < osz; i++) {
      tree[i].count = counts[i];
    }
    int leaf = osz - 1;
    int node = osz;
    for (int i = osz; i < 2 * osz - 1; i++) {
      int[] mini = new int[2];
      for (int j = 0; j < 2; j++) {
        if (leaf >= 0 && tree[leaf].count < tree[node].count) {
          mini[j] = leaf--;
        } else {
          mini[j] = node++;
        }
      }
      tree[i].left = mini[0];
      tree[i].right = mini[1];
      tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
      tree[mini[0]].parent = i;
      tree[mini[1]].parent = i;
      tree[mini[1]].binary = true;
    }
    for (int i = 0; i < osz; i++) {
      List<Integer> path = new ArrayList<>();
      List<Boolean> code = new ArrayList<>();
      int j = i;
      while (tree[j].parent != -1) {
        path.add(tree[j].parent - osz);
        code.add(tree[j].binary);
        j = tree[j].parent;
      }
      paths[i] = Ints.toArray(path);
      codes[i] = Booleans.toArray(code);
    }
  }

  public float getLoss() {
    return loss / nexamples;
  }

  private void initSigmoid() {
    tSigmoid = new float[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
      float x = (i * 2 * MAX_SIGMOID) / (float) SIGMOID_TABLE_SIZE - MAX_SIGMOID;
      tSigmoid[i] = (float) (1.0 / (1.0 + Math.exp(-x)));
    }
  }

  private void initLog() {
    tLog = new float[LOG_TABLE_SIZE + 1];
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
      float x =  (i + 1e-5f) / LOG_TABLE_SIZE;
      tLog[i] = (float) Math.log(x);
    }
  }

  public float log(float x) {
    if (x > 1.0) {
      return 0.0f;
    } else {
      int i = (int) (x * LOG_TABLE_SIZE);
      return tLog[i];
    }
  }

  public float sigmoid(float x) {
    if (x < -MAX_SIGMOID) {
      return 0.0f;
    } else if (x > MAX_SIGMOID) {
      return 1.0f;
    } else {
      int i = (int) ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
      return tSigmoid[i];
    }
  }

}
