import java.io.Serializable;

public class Edge implements Serializable {
    private static final long serialVersionUID = 1L;
    String name;
    String leftNeuron;
    String rightNeuron;
    double weight;
    double chnageInWeight = 0;
    double changeInDelta;
    double prevChangeInDelta;



    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getLeftNeuron() {
        return leftNeuron;
    }

    public void setLeftNeuron(String leftNeuron) {
        this.leftNeuron = leftNeuron;
    }

    public String getRightNeuron() {
        return rightNeuron;
    }

    public void setRightNeuron(String rightNeuron) {
        this.rightNeuron = rightNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getChnageInWeight() {
        return chnageInWeight;
    }

    public void setChnageInWeight(double chnageInWeight) {
        this.chnageInWeight = chnageInWeight;
    }
}
