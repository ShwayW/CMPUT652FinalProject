package engine.core;

public class StepResult {
    public int[][] observation;
    public float reward;
    public boolean done;
    public String info;

    public StepResult(int[][] observation, float reward, boolean done, String info) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;

    }
}
