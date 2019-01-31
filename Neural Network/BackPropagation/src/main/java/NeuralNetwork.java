
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Random;

class NN  implements Serializable {
    HashMap<String, Neuron> inputLayer;
    HashMap<String, Neuron> hiddenLayer;
    HashMap<String, Neuron> outPutLayer;
    HashMap<String, Edge> edges;

    public NN(HashMap<String, Neuron> inputLayer, HashMap<String, Neuron> hiddenLayer,
              HashMap<String, Neuron> outPutLayer, HashMap<String, Edge> edges) {
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outPutLayer = outPutLayer;
        this.edges = edges;
    }

    public HashMap<String, Neuron> getInputLayer() {
        return inputLayer;
    }

    public HashMap<String, Neuron> getHiddenLayer() {
        return hiddenLayer;
    }

    public HashMap<String, Neuron> getOutPutLayer() {
        return outPutLayer;
    }

    public HashMap<String, Edge> getEdges() {
        return edges;
    }
}

public class NeuralNetwork implements Serializable {
    private HashMap<String, Neuron> inputLayer;
    private HashMap<String, Neuron> hiddenLayer;
    private HashMap<String, Neuron> outPutLayer;
    private HashMap<String, Edge> edges;
    private static String fileName;

    NeuralNetwork(String filename){
        inputLayer = new HashMap<>();
        hiddenLayer =  new HashMap<>();
        outPutLayer = new HashMap<>();
        edges = new HashMap<>();
        this.fileName = filename;
    }

    NeuralNetwork(HashMap<String, Neuron> inputLayer, HashMap<String, Neuron> hiddenLayer,
                  HashMap<String, Neuron> outPutLayer, HashMap<String, Edge> edges) {
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outPutLayer = outPutLayer;
        this.edges = edges;
    }

    public void reset(){
        inputLayer = new HashMap<>();
        hiddenLayer =  new HashMap<>();
        outPutLayer = new HashMap<>();
        edges = new HashMap<>();
    }

    public HashMap<String, Neuron> getInputLayer() {
        return inputLayer;
    }


    public HashMap<String, Neuron> getHiddenLayer() {
        return hiddenLayer;
    }


    public HashMap<String, Neuron> getOutPutLayer() {
        return outPutLayer;
    }


    public HashMap<String, Edge> getEdges() {
        return edges;
    }


    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }


    public void buildNetworkTopology() throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        // Reads in network topology as a JSON object
        JSONObject network = (JSONObject) parser.parse(new FileReader(fileName));
        JSONArray input_layerJson = (JSONArray)network.get("input_layer");
        JSONArray hidden_layerJson = (JSONArray)network.get("hidden_layer");
        JSONArray output_layerJson = (JSONArray)network.get("output_layer");

        // Program start time
        long start = System.currentTimeMillis();
        //Build network Topology from JSOn object
        //Start with input layer Topology in parallel
        for(Object node: input_layerJson){
            JSONObject obj = (JSONObject)node;
            Neuron unit = new Neuron();
            unit.setName((String) obj.get("name"));
            JSONArray edgesObj = (JSONArray) obj.get("front_edges");
            // Create connected edges for this unit
            // input layer only have front edges
            for(Object edge: edgesObj){
                JSONObject edgeObj = (JSONObject)edge;
                // create new edge
                Edge front_edge = new Edge();
                // assign  left and right nodes to the edge
                front_edge.setLeftNeuron(unit.getName());
                front_edge.setRightNeuron((String)edgeObj.get("name"));
                front_edge.setName(unit.getName() + "-" + edgeObj.get("name"));
                // Initialized weight to randomly generate number (Range is -1.0 to 1.0)
                double wt = -1.0 + new Random().nextDouble() * (1.0 - (-1.0));
                front_edge.setWeight(wt);
                //front_edge.setWeight(ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
                unit.getFrontEdges().add(front_edge);
                // Add edge to Neural Network topology
                edges.put(front_edge.getName(), front_edge);
            }

            // Add unit to Neural Network topology
            inputLayer.putIfAbsent(unit.getName(), unit);
        }

        // Build Hidden layer Topology in parallel
        for(Object node: hidden_layerJson){
            JSONObject obj = (JSONObject)node;
            Neuron unit = new Neuron();
            unit.setName((String) obj.get("name"));
            // Initialized unit bias to randomly generate number (Range is -1.0 to 1.0)
            double wt = -1.0 + new Random().nextDouble() * (1.0 - (-1.0));
            unit.setBias(wt);
            //unit.setBias(ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
            JSONArray front_edges = (JSONArray) obj.get("front_edges");
            JSONArray back_edges = (JSONArray) obj.get("back_edges");

            // Create connected edges for this unit
            // Hidden layer has both front and back edge (This is a personal design choice)
            for(Object edge: front_edges){
                JSONObject edgeObj = (JSONObject)edge;
                if(!hiddenLayer.containsKey((String)edgeObj.get("name"))){
                    // create new edge
                    Edge front_edge = new Edge();
                    // assign  left and right nodes to the edge
                    front_edge.setLeftNeuron(unit.getName());
                    front_edge.setRightNeuron((String)edgeObj.get("name"));
                    front_edge.setName(unit.getName() + "-" + edgeObj.get("name"));
                    // Initialized weight to randomly generate number (Range is -1.0 to 1.0)
                    wt = -1.0 + new Random().nextDouble() * (1.0 - (-1.0));
                    front_edge.setWeight(wt);
                    //front_edge.setWeight(ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
                    unit.getFrontEdges().add(front_edge);
                    // Add edge to Neural Network topology
                    edges.put(front_edge.getName(), front_edge);
                }
                else{
                    String name = unit.getName() + "-" + edgeObj.get("name");
                    Edge front_edge = edges.get(name);
                    unit.getFrontEdges().add(front_edge);
                }
            }


            for(Object edge: back_edges){
                JSONObject edgeObj = (JSONObject)edge;
                if(!hiddenLayer.containsKey((String)edgeObj.get("name"))
                        && !inputLayer.containsKey((String)edgeObj.get("name")) ){
                    computeBackEdges(unit, edgeObj);
                }
                else{
                    String name = edgeObj.get("name") + "-" + unit.getName();
                    Edge back_edge = edges.get(name);
                    unit.getBackEdge().add(back_edge);
                }
            }

            // Add unit to Neural Network topology
            hiddenLayer.putIfAbsent(unit.getName(), unit);
        }

        // Build Output layer Topology in parallel
        for(Object node: output_layerJson){
            JSONObject obj = (JSONObject) node;
            Neuron unit = new Neuron();
            unit.setName((String) obj.get("name"));
            // Initialized unit bias to randomly generate number (Range is -1.0 to 1.0)
            double wt = -1.0 + new Random().nextDouble() * (1.0 - (-1.0));
            unit.setBias(wt);
            //unit.setBias(ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
            JSONArray back_edges = (JSONArray) obj.get("back_edges");

            for(Object edge: back_edges) {
                JSONObject edgeObj = (JSONObject) edge;
                if (!hiddenLayer.containsKey((String) edgeObj.get("name"))){
                    computeBackEdges(unit, edgeObj);
                }
                else{
                    String name = edgeObj.get("name") + "-" + unit.getName();
                    Edge back_edge = edges.get(name);
                    unit.getBackEdge().add(back_edge);
                }
            }

            // Add unit to Neural Network topology
            outPutLayer.putIfAbsent(unit.getName(), unit);
        }

        // Program end time
        long end = System.currentTimeMillis();
        NumberFormat formatter = new DecimalFormat("#0.00000");
        System.out.print("Execution time is " + formatter.format((end - start) / 1000d) + " seconds");
        System.out.println();




    }

    public void computeBackEdges(Neuron unit, JSONObject edgeObj){
        Edge back_edge = new Edge();
        // assign  left and right nodes to the edge
        back_edge.setLeftNeuron((String)edgeObj.get("name"));
        back_edge.setRightNeuron(unit.getName());
        back_edge.setName(edgeObj.get("name") + "-" + unit.getName());
        // Initialized weight to randomly generate number (Range is -1.0 to 1.0)
        double wt = -1.0 + new Random().nextDouble() * (1.0 - (-1.0));
        back_edge.setWeight(wt);
        //back_edge.setWeight(ThreadLocalRandom.current().nextDouble(-1.0, 1.0));
        unit.getBackEdge().add(back_edge);
        // Add edge to Neural Network topology
        edges.put(back_edge.getName(), back_edge);
    }

}

