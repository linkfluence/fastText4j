package fasttext;

public class FastTextPrediction {

  private final String label;
  private final double logProbability;

  public FastTextPrediction(String label, double logProbability) {
    this.label = label;
    this.logProbability = logProbability;
  }

  public String label() {
    return this.label;
  }

  public double logProbability() {
    return this.logProbability;
  }

  public double probability() {
    return Math.exp(this.logProbability);
  }

}
