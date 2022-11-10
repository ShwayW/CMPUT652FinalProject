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
    
    public static StepResult reset(boolean render) {
    	
    	game = new MarioGame();
    	MarioResult result = game.buildGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, render);
    	return game.stepGame(new boolean[] {false, false, false, false, false});
    }
    
    public static StepResult step(boolean[] action) {
    	return game.stepGame(action);
    }
    
    public static void close() {
    	game.killGame();
    }

    public static void sample() {
        MarioGame game = new MarioGame();
        // printResults(game.playGame(getLevel("../levels/original/lvl-1.txt"), 200, 0));
        MarioResult result = game.buildGame(new agents.random.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, true);
        while (result.getGameStatus() == GameStatus.RUNNING) {
            game.stepGame(new boolean[] {false, true, false, false, false});
        }
        game.killGame(); // reset()


        // printResults(game.playGame(getLevel("../levels/original/lvl-1.txt"), 200, 0));
        result = game.buildGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-1.txt"), 20, 0, true);
        while (result.getGameStatus() == GameStatus.RUNNING) {
            game.stepGame();
        }
        // game.killGame(); // reset()
    }
}
