import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioResult;
import engine.helper.GameStatus;
import engine.core.StepResult;


/*

Helper code to control the game in the same way Python code would if formatted as an OpenAI Gym environment.

Must implement a way to reset(), step(), and take an observation() from the game at any timestep

*/

public class PythonController {
    
	static MarioGame game;

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }
    
    public static StepResult reset(boolean render, String level, int timer) {
    	// Start new game and return first observation
    	game = new MarioGame();
    	MarioResult result = game.buildGame(new agents.robinBaumgarten.Agent(), getLevel(level), timer, 0, render);
    	return game.stepGame(new boolean[] {false, false, false, false, false}); // Just step the game once with a no-op
    }
    
    public static StepResult step(boolean[] action) {
        // Step the environment by one game tick with the provided action
    	return game.stepGame(action);
    }
    
    public static StepResult step() {
    	return game.stepGame(); // if step without action, use default A* agent's action
    }
    
    public static void close() {
    	game.killGame();
    }

    public static void sample() {
        // Sample code which runs the game twice, once with holding right the entire time and second time with an A* agent

        MarioGame game = new MarioGame();

        MarioResult result = game.buildGame(new agents.random.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, true);
        while (result.getGameStatus() == GameStatus.RUNNING) {
            game.stepGame(new boolean[] {false, true, false, false, false});
        }
        game.killGame(); // reset()

        result = game.buildGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, true);
        while (result.getGameStatus() == GameStatus.RUNNING) {
            game.stepGame();
        }
    }
    
    public static MarioResult collectTrajectory() {
    	// Use this to collect trajectories from human-input
    	MarioGame game = new MarioGame();
    	MarioResult result = game.runGame(new agents.human.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, true);
    	game.killGame();
    	return result;
    }
}
