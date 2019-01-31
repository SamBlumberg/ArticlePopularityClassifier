import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Neuron implements Serializable {
    private static final long serialVersionUID = 1L;
    String name;
    private double input;
    private double output;
    private List<Edge> frontEdges = new ArrayList<>();
    private List<Edge> backEdge = new ArrayList<>();
    private double bias;
    double changeInBias;
    double prevChangeInBias;
    double changeInDelta;
    double prevChnageInDelta;
    private double expectedOutput; // This is only useful if this neuron is an output neuron
    private double error;

    public double getChangeInBias() {
        return changeInBias;
    }

    public void setChangeInBias(double changeInBias) {
        this.changeInBias = changeInBias;
    }

    public double getPrevChangeInBias() {
        return prevChangeInBias;
    }

    public void setPrevChangeInBias(double prevChangeInBias) {
        this.prevChangeInBias = prevChangeInBias;
    }

    public double getChangeInDelta() {
        return changeInDelta;
    }

    public void setChangeInDelta(double changeInDelta) {
        this.changeInDelta = changeInDelta;
    }

    public double getPrevChnageInDelta() {
        return prevChnageInDelta;
    }

    public void setPrevChnageInDelta(double prevChnageInDelta) {
        this.prevChnageInDelta = prevChnageInDelta;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getInput() {
        return input;
    }

    public void setInput(double input) {
        this.input = input;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public List<Edge> getFrontEdges() {
        return frontEdges;
    }

    public void setFrontEdges(List<Edge> frontEdges) {
        this.frontEdges = frontEdges;
    }

    public List<Edge> getBackEdge() {
        return backEdge;
    }

    public void setBackEdge(List<Edge> backEdge) {
        this.backEdge = backEdge;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getExpectedOutput() {
        return expectedOutput;
    }

    public void setExpectedOutput(double expectedOutput) {
        this.expectedOutput = expectedOutput;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }
}
