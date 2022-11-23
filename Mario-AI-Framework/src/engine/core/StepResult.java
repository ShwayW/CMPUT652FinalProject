package engine.core;

public class StepResult {
    public int[][] observation;
    public float reward;
    public boolean done;
    public float[] info;

    public StepResult(int[][] observation, float reward, boolean done, float[] info) {
        this.observation = observation;
        this.reward = reward; // Note - dont use this anymore, instead calculate reward in python for ease of development
        this.done = done;
        this.info = info;

    }
}
